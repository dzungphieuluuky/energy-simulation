# train_ppo_with_bc.py
import numpy as np
import torch
import pickle
import os
import random
import json
import sys
import multiprocessing
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.monitor import Monitor

from fiveg_env import FiveGEnv

# Note: The BehavioralCloningLogger and SafeBehavioralCurriculum classes are good as-is.

class SafeBehavioralCurriculum:
    """
    An adaptive behavioral curriculum that regulates difficulty by advancing
    when the agent is proficient and retreating when it struggles.
    REFINED VERSION.
    """
    def __init__(self, action_dim, 
                 initial_range=(0.8, 0.95),
                 final_range=(0.5, 0.75),
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
                self.current_low = max(self.current_low - self.progression_step, self.final_low)
                self.current_high = max(self.current_high - self.progression_step, self.final_high)
                self.consecutive_compliant = 0
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

def load_scenario_configs(scenario_folder: str) -> list:
    """Loads all .json configuration files from a given folder."""
    if not os.path.isdir(scenario_folder):
        raise FileNotFoundError(f"Scenario folder not found: {scenario_folder}")
    
    config_paths = [os.path.join(scenario_folder, f) for f in os.listdir(scenario_folder) if f.endswith('.json')]
    if not config_paths:
        raise FileNotFoundError(f"No .json scenario files found in '{scenario_folder}'")
        
    configs = []
    print("\n--- Loading Scenario Configurations ---")
    for path in sorted(config_paths):
        with open(path, 'r') as f:
            config = json.load(f)
            config['name'] = config.get('name', os.path.basename(path))
            configs.append(config)
            print(f"  - Loaded '{config['name']}'")
    print("-------------------------------------\n")
    return configs


def create_training_env(configs: list, n_envs=4, is_eval=False):
    """Creates a vectorized environment that cycles through the provided configs."""
    def make_env(rank):
        def _init():
            config_for_this_env = configs[rank % len(configs)].copy()
            config_for_this_env['seed'] = config_for_this_env.get('seed', 42) + rank
            
            env = FiveGEnv(config_for_this_env)
            env = Monitor(env)
            return env
        return _init
    
    vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = vec_env_cls([make_env(i) for i in range(n_envs)])
    
    # We will normalize the reward only for the training environment
    norm_reward = not is_eval
    env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=10., gamma=0.99)
    return env

def run_episode_worker(args):
    """
    A worker function that runs a single episode for a given config.
    Designed to be used with multiprocessing.Pool.
    """
    config, behavioral_policy = args
    
    # Each worker needs its own environment instance
    # Using Monitor to get episode info, but it's not strictly necessary here
    env = Monitor(FiveGEnv(config))
    
    obs_list, act_list = [], []
    
    obs, _ = env.reset()
    done = False
    while not done:
        action = behavioral_policy.sample_action()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        obs_list.append(obs)
        act_list.append(action)
        obs = next_obs
        
    env.close()
    return obs_list, act_list

def generate_expert_dataset_parallel(configs: list, behavioral_policy, num_episodes_per_config=50):
    """
    Generates a mixed expert dataset in parallel using all available CPU cores.
    Includes a robust mechanism to handle KeyboardInterrupt (Ctrl+C).
    """
    print(f"\n{'='*70}\nGENERATING MIXED EXPERT DATASET (IN PARALLEL)\n{'='*70}")
    
    tasks = []
    for config in configs:
        for _ in range(num_episodes_per_config):
            tasks.append((config.copy(), behavioral_policy))

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} parallel processes. Press Ctrl+C to interrupt.")
    
    expert_observations = []
    expert_actions = []

    # Manual pool management for robust interrupt handling
    pool = multiprocessing.Pool(processes=num_processes)
    try:
        # Use tqdm to create a progress bar for the parallel tasks
        results_iterator = pool.imap(run_episode_worker, tasks)
        
        # Use tqdm's manual control to iterate
        with tqdm(total=len(tasks), desc="Generating Episodes") as pbar:
            for result in results_iterator:
                obs_list, act_list = result
                expert_observations.extend(obs_list)
                expert_actions.extend(act_list)
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n\n🛑 KeyboardInterrupt detected! Terminating worker processes...")
        # Forcefully terminate all worker processes
        pool.terminate()
        # Wait for the processes to be fully cleaned up
        pool.join()
        print("Workers terminated. Exiting script.")
        # Exit the script with an error code
        sys.exit(1)
    
    finally:
        # This block ensures the pool is closed cleanly on normal completion
        pool.close()
        pool.join()

    print(f"\n✅ Mixed dataset generation complete. Total samples: {len(expert_actions)}\n")
    return np.array(expert_observations), np.array(expert_actions)

def generate_expert_dataset(configs: list, behavioral_policy, num_episodes_per_config=50):
    """
    Generates a mixed expert dataset using unnormalized environments.
    """
    print(f"\n{'='*70}\nGENERATING MIXED EXPERT DATASET\n{'='*70}")
    
    expert_observations = []
    expert_actions = []
    
    total_episodes = num_episodes_per_config * len(configs)
    current_episode = 0

    for config in configs:
        print(f"\n--- Generating data for scenario: {config.get('name', 'N/A')} ---")
        # CRITICAL: Use a simple, unnormalized Monitor env for data generation
        temp_env = Monitor(FiveGEnv(config))
        
        for ep in range(num_episodes_per_config):
            obs, _ = temp_env.reset()
            done = False
            while not done:
                action = behavioral_policy.sample_action()
                next_obs, reward, terminated, truncated, info = temp_env.step(action)
                done = terminated or truncated
                
                expert_observations.append(obs)
                expert_actions.append(action)
                obs = next_obs
            
            current_episode += 1
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1}/{num_episodes_per_config} complete. "
                      f"Total progress: {current_episode}/{total_episodes} episodes.")
    
    temp_env.close()
    print(f"\n✅ Mixed dataset generation complete. Total samples: {len(expert_actions)}\n")
    return np.array(expert_observations), np.array(expert_actions)


def pretrain_with_behavioral_cloning(model, observations: np.array, actions: np.array, epochs: int = 25, 
                                   batch_size: int = 256, validation_split: float = 0.15):
    """
    Enhanced behavioral cloning with correct shuffling, validation, and early stopping.
    """
    print(f"\n{'='*70}\nBEHAVIORAL CLONING PRE-TRAINING\n{'='*70}")
    
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
            _, val_log_prob, _ = model.policy.evaluate_actions(val_obs_tensor, val_acts_tensor)
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
    TOTAL_TIMESTEPS = 1_500_000
    PRETRAIN_EPISODES_PER_CONFIG = 100
    PRETRAIN_EPOCHS = 50
    N_ENVS = 9
    SCENARIO_FOLDER = "scenarios"
    
    os.makedirs('sb3_models', exist_ok=True); 
    os.makedirs('expert_data', exist_ok=True)
    
    # --- 2. Load All Scenario Configs ---
    try:
        scenario_configs = load_scenario_configs(SCENARIO_FOLDER)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}"); return

    # --- 3. Generate Expert Dataset ---
    temp_env_for_dims = FiveGEnv(scenario_configs[0])
    behavioral_policy = SafeBehavioralCurriculum(action_dim=temp_env_for_dims.action_space.shape[0])
    del temp_env_for_dims

    expert_obs, expert_acts = generate_expert_dataset_parallel(
        scenario_configs, behavioral_policy, num_episodes_per_config=PRETRAIN_EPISODES_PER_CONFIG
    )
    
    with open('expert_data/mixed_expert_data.pkl', 'wb') as f:
        pickle.dump({'observations': expert_obs, 'actions': expert_acts}, f)
        
    # --- 4. Create the Final Training and Evaluation Environments ---
    print("\nInitializing normalized, parallel environments for PPO training...")
    env = create_training_env(scenario_configs, n_envs=N_ENVS)
    
    # The evaluation env should use the same normalization stats as the training env
    print("Initializing normalized evaluation environment...")
    eval_env = create_training_env([scenario_configs[0]], n_envs=1, is_eval=True)

    # --- 5. CRITICAL: Normalize Expert Data Using Training Env Statistics ---
    print("Normalizing expert data using the training environment's statistics...")
    # This step is vital. It ensures the BC pre-training sees data in the same
    # format that the RL agent will see during fine-tuning.
    normalized_expert_obs = env.normalize_obs(expert_obs)

    # --- 6. Initialize PPO Model ---
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, # Standard learning rate for BC pre-training
        n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, # Standard entropy for BC
        verbose=1, tensorboard_log="sb3_logs/PPO_MixedBC/",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # --- 7. Phase 2: Behavioral Cloning on NORMALIZED Mixed Data ---
    pretrain_with_behavioral_cloning(
        model, normalized_expert_obs, expert_acts, epochs=PRETRAIN_EPOCHS
    )
    
    # --- 8. Phase 3: Reinforcement Learning Fine-Tuning ---
    print(f"\n{'='*70}\nSTARTING REINFORCEMENT LEARNING FINE-TUNING\n{'='*70}")
    
    # *** CRITICAL FIX: Transfer normalization stats to the eval env ***
    # This ensures that evaluation is performed on the same scale as training.
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    
    eval_callback = EvalCallback(
        eval_env, best_model_save_path='sb3_models/best_mixed_model/',
        log_path='sb3_logs/PPO_BC/', eval_freq=max(15000 // N_ENVS, 1),
        deterministic=True, render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // N_ENVS, 1), save_path='sb3_models/', name_prefix='ppo_mixed_bc'
    )
    
    # *** CRITICAL FIX: Lower learning rate and entropy for fine-tuning ***
    model.learning_rate = 5e-6  # Much smaller learning rate
    model.ent_coef = 0.001     # Lower entropy to encourage exploitation

    # The .learn() call will now start from the BC-trained policy
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=False # IMPORTANT: Do not reset timesteps after BC
    )
    
    # --- 9. Save Final Model and Normalization Stats ---
    model.save("sb3_models/ppo_mixed_bc_final")
    env.save("sb3_models/vec_normalize_mixed_final.pkl") # This saves the normalization stats
    print(f"\n🎉 GENERALIZED TRAINING COMPLETE! Model and normalization stats saved.")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()