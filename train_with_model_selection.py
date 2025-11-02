# train_sb3.py
import json
import torch
import os
import pandas as pd
from typing import Callable, List
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import numpy as np

from fiveg_env_chatgpt import FiveGEnv
from custom_models_sb3 import EnhancedAttentionNetwork

class AlgorithmComparisonCallback(BaseCallback):
    """Callback to track training progress and compare algorithm performance."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # Track episode rewards
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_reward += sum(self.locals['rewards'])
            
        dones = self.locals.get('dones', [])
        if any(dones):
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
            if len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {len(self.episode_rewards)}, Last 10 eps mean reward: {mean_reward:.3f}")
        
        return True

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env_thunk(env_config, max_cells):
    """Creates a thunk (a zero-argument function) for SubprocVecEnv."""
    def _thunk():
        env = FiveGEnv(env_config=env_config, max_cells=max_cells)
        env = Monitor(env)
        return env
    return _thunk

def create_model(algorithm, env, policy_kwargs, device, learning_rate):
    """Create model based on algorithm choice."""
    if algorithm.lower() == 'sac':
        return SAC(
            policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs,
            learning_rate=learning_rate, buffer_size=1_000_000,
            batch_size=256, tau=0.005, gamma=0.99,
            train_freq=1, gradient_steps=1,
            ent_coef='auto', target_entropy='auto',
            verbose=1, tensorboard_log="sb3_logs/", device=device
        )
    
    elif algorithm.lower() == 'ppo':
        return PPO(
            policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs,
            learning_rate=learning_rate, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, verbose=1, 
            tensorboard_log="sb3_logs/", device=device
        )
    
    elif algorithm.lower() == 'td3':
        return TD3(
            policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs,
            learning_rate=learning_rate, buffer_size=1_000_000,
            batch_size=100, gamma=0.99, tau=0.005,
            train_freq=1, gradient_steps=1, policy_delay=2,
            verbose=1, tensorboard_log="sb3_logs/", device=device
        )
    
    elif algorithm.lower() == 'ddpg':
        return DDPG(
            policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs,
            learning_rate=learning_rate, buffer_size=1_000_000,
            batch_size=100, gamma=0.99, tau=0.005,
            train_freq=1, gradient_steps=1,
            verbose=1, tensorboard_log="sb3_logs/", device=device
        )
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def benchmark_algorithms(config, max_cells, num_cpu, total_timesteps=50000):
    """Quick benchmark of different algorithms to find the best one."""
    print("\n=== ALGORITHM BENCHMARK ===")
    
    algorithms = ['sac', 'ppo', 'td3']
    results = {}
    
    for algo in algorithms:
        print(f"\n--- Testing {algo.upper()} ---")
        
        # Create environment
        env_thunks = [make_env_thunk(config, max_cells) for _ in range(num_cpu)]
        env = SubprocVecEnv(env_thunks)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # Create model
        policy_kwargs = dict(
            features_extractor_class=EnhancedAttentionNetwork,
            features_extractor_kwargs=dict(
                features_dim=256,
                max_cells=max_cells,
                n_cell_features=25,
            ),
        )
        
        model = create_model(algo, env, policy_kwargs, 'cpu', linear_schedule(3e-4))
        
        # Train briefly
        callback = AlgorithmComparisonCallback()
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
        
        # Store results
        results[algo] = {
            'final_reward': callback.episode_rewards[-1] if callback.episode_rewards else 0,
            'mean_reward': np.mean(callback.episode_rewards) if callback.episode_rewards else 0,
            'max_reward': np.max(callback.episode_rewards) if callback.episode_rewards else 0
        }
        
        env.close()
    
    # Print results
    print("\n=== BENCHMARK RESULTS ===")
    for algo, res in results.items():
        print(f"{algo.upper()}: Mean={res['mean_reward']:.3f}, Max={res['max_reward']:.3f}")
    
    best_algo = max(results.items(), key=lambda x: x[1]['mean_reward'])[0]
    print(f"\n🎯 RECOMMENDED ALGORITHM: {best_algo.upper()}")
    
    return best_algo, results

if __name__ == "__main__":
    print("Starting Stable Baselines3 5G Agent Training...")
    
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_CELLS_SYSTEM_WIDE = 57
    num_cpu = 5
    
    print(f"Using device: {device}")
    print(f"Max cells: {MAX_CELLS_SYSTEM_WIDE}, Parallel envs: {num_cpu}")

    # --- Load Scenarios ---
    scenario_folder = "scenarios/"
    scenario_files = [f for f in os.listdir(scenario_folder) if f.endswith('.json')]
    scenario_configs = []
    
    for sf in sorted(scenario_files):
        with open(os.path.join(scenario_folder, sf), 'r') as f:
            config = json.load(f)
            scenario_configs.append(config)
            print(f"Loaded: {config.get('name', sf)}")

    # --- Algorithm Selection ---
    print("\n" + "="*50)
    print("ALGORITHM SELECTION")
    print("="*50)
    print("Available algorithms:")
    print("1. SAC (Recommended) - Best for continuous control, sample efficient")
    print("2. PPO  - Stable, good performance but less sample efficient")  
    print("3. TD3  - Good for continuous control, but may be less stable")
    print("4. DDPG - Older, less stable than SAC/TD3")
    
    choice = input("\nChoose algorithm (1=SAC, 2=PPO, 3=TD3, 4=DDPG, 5=Benchmark): ").strip()
    
    if choice == '1':
        algorithm = 'sac'
    elif choice == '2':
        algorithm = 'ppo'
    elif choice == '3':
        algorithm = 'td3'
    elif choice == '4':
        algorithm = 'ddpg'
    elif choice == '5':
        # Run benchmark
        benchmark_config = scenario_configs[0]  # Use first scenario for benchmark
        algorithm, benchmark_results = benchmark_algorithms(
            benchmark_config, MAX_CELLS_SYSTEM_WIDE, min(2, num_cpu), 50000
        )
    else:
        algorithm = 'sac'  # Default to SAC
        print(f"Defaulting to SAC")

    print(f"\nSelected algorithm: {algorithm.upper()}")

    # --- Training Setup ---
    log_dir, model_dir = "sb3_logs/", "sb3_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create environments
    env_thunks = []
    for i in range(num_cpu):
        config = scenario_configs[i % len(scenario_configs)]
        env_thunks.append(make_env_thunk(config, MAX_CELLS_SYSTEM_WIDE))

    # Initialize model and environment
    model, env = None, None
    continue_training = input("\nContinue training from checkpoint? (y/n): ").strip().lower() == 'y'

    try:
        if continue_training:
            # Load existing model
            print("\nAvailable checkpoints:")
            for f in sorted(os.listdir(model_dir)):
                if f.endswith('.zip'): print(f"  - {f}")
            model_path = input("Enter model checkpoint path: ").strip()

            print("\nAvailable normalization stats:")
            for f in sorted(os.listdir(model_dir)):
                if f.endswith('.pkl'): print(f"  - {f}")
            stats_path = input("Enter normalization stats path: ").strip()

            base_env = SubprocVecEnv(env_thunks)
            env = VecNormalize.load(os.path.join(model_dir, stats_path), base_env)
            
            if algorithm == 'sac':
                model = SAC.load(os.path.join(model_dir, model_path), env=env, device=device)
            elif algorithm == 'ppo':
                model = PPO.load(os.path.join(model_dir, model_path), env=env, device=device,
                                 learning_rate=linear_schedule(1e-4))
            elif algorithm == 'td3':
                model = TD3.load(os.path.join(model_dir, model_path), env=env, device=device)
            elif algorithm == 'ddpg':
                model = DDPG.load(os.path.join(model_dir, model_path), env=env, device=device)
                
            print(f"\n{algorithm.upper()} model loaded successfully.")
            
        else:
            raise ValueError("Starting new training session.")
            
    except (ValueError, FileNotFoundError) as e:
        if continue_training:
            print(f"Error loading: {e}")
        print("\nStarting new training session...")
        
        env = SubprocVecEnv(env_thunks)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        policy_kwargs = dict(
            features_extractor_class=EnhancedAttentionNetwork,
            features_extractor_kwargs=dict(
                features_dim=256,
                max_cells=MAX_CELLS_SYSTEM_WIDE,
                n_cell_features=25,
            ),
        )
        
        # Adjust learning rate based on algorithm
        if algorithm == 'sac':
            lr = 3e-4
        elif algorithm == 'ppo':
            lr = 2.5e-4  
        else:  # td3, ddpg
            lr = 1e-3
            
        model = create_model(algorithm, env, policy_kwargs, device, linear_schedule(lr))

    # --- Training ---
    total_timesteps_str = input(f"\nEnter total timesteps for {algorithm.upper()} (default 1,000,000): ").strip()
    try:
        total_timesteps = int(float(total_timesteps_str.replace(',', '')))
    except ValueError:
        total_timesteps = 1_000_000

    run_name_prefix = f"{algorithm}_mixed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // num_cpu, 1), 
        save_path=model_dir, 
        name_prefix=run_name_prefix, 
        verbose=1
    )
    progress_callback = AlgorithmComparisonCallback()
    
    print(f"\n{'='*60}")
    print(f"TRAINING {algorithm.upper()} FOR {total_timesteps:,} TIMESTEPS")
    print(f"{'='*60}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, progress_callback],
        progress_bar=True,
        reset_num_timesteps=not continue_training
    )

    # --- Save Final Model ---
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = os.path.join(model_dir, f"{algorithm}_mixed_{timestamp}.zip")
    model.save(final_model_path)

    stats_path = os.path.join(model_dir, f"vec_normalize_{timestamp}.pkl")
    env.save(stats_path)
    
    print(f"\n✅ Training complete!")
    print(f"Model saved: {final_model_path}")
    print(f"Stats saved: {stats_path}")

    # --- Quick Evaluation ---
    print("\n--- Running Quick Evaluation ---")
    test_config = scenario_configs[0]
    test_env = FiveGEnv(test_config, max_cells=MAX_CELLS_SYSTEM_WIDE)
    
    obs, _ = test_env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 50:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        total_reward += reward
        steps += 1
    
    print(f"Test: {steps} steps, Total reward: {total_reward:.3f}")
    
    if total_reward < 0:
        print("⚠️  Model may need more training or hyperparameter tuning")
    else:
        print("🎉 Model shows promising performance!")