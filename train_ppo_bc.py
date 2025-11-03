# train_ppo_with_bc.py
import numpy as np
import torch
import pickle
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from fiveg_env import FiveGEnv

# Note: The BehavioralCloningLogger is good as-is.

class SafeBehavioralCurriculum:
    """
    An adaptive behavioral curriculum that regulates difficulty by advancing
    when the agent is proficient and retreating when it struggles.
    REFINED VERSION.
    """
    def __init__(self, action_dim, 
                 initial_range=(0.9, 1.0),
                 final_range=(0.4, 0.7),
                 hold_episodes=50,
                 compliance_threshold=0.98,
                 patience_to_advance=15,
                 progression_step=0.01,
                 patience_to_regress=25,
                 regression_step=0.005):
        self.action_dim = action_dim
        self.initial_low, self.initial_high = initial_range
        self.final_low, self.final_high = final_range
        self.hold_episodes = hold_episodes
        self.compliance_threshold = compliance_threshold
        self.patience_to_advance = patience_to_advance
        self.progression_step = progression_step
        self.patience_to_regress = patience_to_regress
        self.regression_step = regression_step

        self.episode_count = 0
        self.consecutive_compliant = 0
        self.consecutive_failed = 0
        self.current_low, self.current_high = initial_range
        
    def update_curriculum(self, episode_compliance_rate: float) -> dict:
        self.episode_count += 1
        is_compliant = episode_compliance_rate >= self.compliance_threshold
        
        if is_compliant:
            self.consecutive_compliant += 1
            self.consecutive_failed = 0
        else:
            self.consecutive_compliant = 0
            self.consecutive_failed += 1

        if self.episode_count <= self.hold_episodes:
            return {'status': 'HOLD', 'reason': 'Initial hold phase'}

        if self.consecutive_compliant >= self.patience_to_advance:
            if self.current_high > self.final_high:
                # --- FIX 2: Incremental, not absolute, progression ---
                self.current_low = max(self.current_low - self.progression_step, self.final_low)
                self.current_high = max(self.current_high - self.progression_step, self.final_high)
                self.consecutive_compliant = 0  # Reset patience after progressing
                print(f"🎓 CURRICULUM ADVANCED! New Power: [{self.current_low:.3f}, {self.current_high:.3f}]")
                return {'status': 'ADVANCED', 'reason': 'Patience met'}
            else:
                return {'status': 'HOLD', 'reason': 'At final range'}

        if self.consecutive_failed >= self.patience_to_regress:
            if self.current_high < self.initial_high:
                self.current_low = min(self.current_low + self.regression_step, self.initial_low)
                self.current_high = min(self.current_high + self.regression_step, self.initial_high)
                self.consecutive_failed = 0
                print(f"⏪ CURRICULUM REGRESSED! New Power: [{self.current_low:.3f}, {self.current_high:.3f}]")
                return {'status': 'REGRESSED', 'reason': 'Consecutive failures'}

        return {'status': 'HOLD', 'reason': 'Building patience'}

    def sample_action(self):
        return np.random.uniform(self.current_low, self.current_high, size=self.action_dim)

def create_training_env(config, n_envs=4, is_eval=False):
    """Creates and wraps the training environment."""
    def make_env(rank):
        def _init():
            env_conf = config.copy()
            env_conf['seed'] = config.get('seed', 42) + rank
            env = FiveGEnv(env_conf)
            env = Monitor(env)
            return env
        return _init
    
    if n_envs > 1 and not is_eval: # Use SubprocVecEnv for parallel training
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else: # Use DummyVecEnv for data generation, eval, or single-core training
        env = DummyVecEnv([make_env(0)])
    
    norm_reward = not is_eval
    env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=10., gamma=0.99)
    return env

def generate_expert_dataset(config, behavioral_policy, num_episodes=150):
    """
    Generate expert dataset using a SINGLE, non-parallel env for correctness and simplicity.
    """
    print(f"\n{'='*70}\nGENERATING EXPERT DATASET (Single Environment)\n{'='*70}")
    
    # --- FIX 1: Use a single, temporary environment for clean data generation ---
    temp_env = Monitor(FiveGEnv(config))
    
    expert_observations = []
    expert_actions = []
    
    for ep in range(num_episodes):
        obs, _ = temp_env.reset()
        done = False
        compliant_steps = 0
        total_steps = 0
        while not done:
            action = behavioral_policy.sample_action()
            next_obs, reward, terminated, truncated, info = temp_env.step(action)
            done = terminated or truncated
            
            expert_observations.append(obs)
            expert_actions.append(action)
            
            if info.get('reward_info', {}).get('constraints_satisfied', False):
                compliant_steps += 1
            total_steps += 1
            obs = next_obs

        compliance_rate = compliant_steps / total_steps if total_steps > 0 else 0
        behavioral_policy.update_curriculum(compliance_rate)

        if (ep + 1) % 10 == 0:
            low, high = behavioral_policy.current_low, behavioral_policy.current_high
            print(f"Episode {ep+1}/{num_episodes} | Compliance: {compliance_rate:.1%} | Power Range: [{low:.3f}, {high:.3f}]")

    print(f"\n✅ Dataset generation complete. Total samples: {len(expert_actions)}\n")
    return np.array(expert_observations), np.array(expert_actions), behavioral_policy

def pretrain_with_behavioral_cloning(model, observations, actions, epochs=25, 
                                   batch_size=256, validation_split=0.15):
    """
    Enhanced behavioral cloning with correct shuffling, validation, and early stopping.
    """
    print(f"\n{'='*70}\nBEHAVIORAL CLONING PRE-TRAINING\n{'='*70}")
    
    # --- FIX 3: Shuffle data BEFORE splitting to prevent data leakage ---
    indices = np.random.permutation(len(observations))
    shuffled_obs = observations[indices]
    shuffled_acts = actions[indices]

    n_val = int(len(shuffled_obs) * validation_split)
    train_obs, val_obs = shuffled_obs[:-n_val], shuffled_obs[-n_val:]
    train_acts, val_acts = shuffled_acts[:-n_val], shuffled_acts[-n_val:]
    
    optimizer = model.policy.optimizer
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        model.policy.train()
        train_loss = 0
        n_batches = 0
        
        for i in range(0, len(train_obs), batch_size):
            obs_tensor = obs_as_tensor(train_obs[i:i+batch_size], model.device)
            acts_tensor = torch.tensor(train_acts[i:i+batch_size], dtype=torch.float32, device=model.device)
            
            _, log_prob, _ = model.policy.evaluate_actions(obs_tensor, acts_tensor)
            loss = -torch.mean(log_prob)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / n_batches if n_batches > 0 else 0
        
        model.policy.eval()
        with torch.no_grad():
            val_obs_tensor = obs_as_tensor(val_obs, model.device)
            val_acts_tensor = torch.tensor(val_acts, dtype=torch.float32, device=model.device)
            val_distribution = model.policy.get_distribution(val_obs_tensor)
            val_log_prob = val_distribution.log_prob(val_acts_tensor)
            val_loss = -torch.mean(val_log_prob).item()
        
        print(f"BC Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"🛑 Early stopping after {epoch+1} epochs due to no improvement in validation loss.")
            break
    
    print("✅ Behavioral cloning complete!")

def main():
    # --- 1. Configuration ---
    TOTAL_TIMESTEPS = 1_000_000
    PRETRAIN_EPISODES = 150
    PRETRAIN_EPOCHS = 30 # Increased for more robust cloning
    N_ENVS = 8
    
    config = {'simTime': 500, 'timeStep': 1, 'numSites': 4, 'numUEs': 100}
    
    os.makedirs('sb3_models', exist_ok=True); os.makedirs('expert_data', exist_ok=True)
    
    # --- 2. Phase 1: Generate Expert Dataset ---
    behavioral_policy = SafeBehavioralCurriculum(action_dim=FiveGEnv(config).action_space.shape[0])
    expert_obs, expert_acts, adapted_policy = generate_expert_dataset(config, behavioral_policy, num_episodes=PRETRAIN_EPISODES)
    
    with open('expert_data/expert_data.pkl', 'wb') as f:
        pickle.dump({'observations': expert_obs, 'actions': expert_acts, 'policy_state': adapted_policy}, f)
    
    # --- 3. Environment Setup for Training ---
    print("\nInitializing parallel environments for PPO training...")
    env = create_training_env(config, n_envs=N_ENVS)
    
    # --- 4. Normalize Expert Data ---
    print("Normalizing expert data using training environment statistics...")
    normalized_expert_obs = env.normalize_obs(expert_obs)

    # --- 5. Initialize PPO Model ---
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4, # Slightly lower for stability with large batches
        n_steps=2048,       # --- FIX 5: Keep n_steps high for stability ---
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log="sb3_logs/PPO_BC/",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    )
    
    # --- 6. Phase 2: Behavioral Cloning ---
    pretrain_with_behavioral_cloning(
        model, normalized_expert_obs, expert_acts, epochs=PRETRAIN_EPOCHS
    )
    
    # --- 7. Phase 3: Reinforcement Learning ---
    print(f"\n{'='*70}\nSTARTING REINFORCEMENT LEARNING FINE-TUNING\n{'='*70}")
    
    eval_env = create_training_env(config, n_envs=1, is_eval=True)
    eval_callback = EvalCallback(
        eval_env, best_model_save_path='sb3_models/best_model/',
        log_path='sb3_logs/PPO_BC/', eval_freq=max(10000 // N_ENVS, 1)
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // N_ENVS, 1), save_path='sb3_models/', name_prefix='ppo_bc'
    )
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=False # Continue timestep count from 0, but policy is pre-trained
    )
    
    # --- 8. Save Final Model ---
    model.save("sb3_models/ppo_bc_final")
    env.save("sb3_models/vec_normalize_final.pkl")
    print(f"\n🎉 TRAINING COMPLETE! Model saved.")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()