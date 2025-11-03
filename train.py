# train_unified.py
import json
import torch
import os
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any

# Stable Baselines3 components
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.base_class import BaseAlgorithm

# Custom environment and components
from fiveg_env import FiveGEnv
from custom_models_sb3 import EnhancedAttentionNetwork
# Assuming these custom callbacks exist in a file named callback.py
from callback import AlgorithmComparisonCallback, ConstraintMonitorCallback 

# =================================================================================
# --- 1. SCRIPT CONFIGURATION & HYPERPARAMETERS ---
# =================================================================================

# --- System-wide constants ---
MAX_CELLS_SYSTEM_WIDE = 57
STATE_DIM_PER_CELL = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CPU_CORES = 10  # Number of parallel environments

# --- Centralized Hyperparameters for Easy Tuning ---
# Policy network architecture is shared across models
POLICY_KWARGS = dict(
    features_extractor_class=EnhancedAttentionNetwork,
    features_extractor_kwargs=dict(
        features_dim=256,
        max_cells=MAX_CELLS_SYSTEM_WIDE,
        n_cell_features=STATE_DIM_PER_CELL,
    ),
    net_arch=dict(pi=[256, 256], qf=[256, 256]) # For actor-critic models
)

HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    'sac': {
        'model_class': SAC,
        'params': {
            'learning_rate': 3e-4,
            'buffer_size': 1_000_000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'ent_coef': 'auto',
            'target_entropy': 'auto',
        }
    },
    'ppo': {
        'model_class': PPO,
        'params': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
        }
    },
    'td3': {
        'model_class': TD3,
        'params': {
            'learning_rate': 1e-3,
            'buffer_size': 1_000_000,
            'batch_size': 100,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_delay': 2,
        }
    },
    'ddpg': {
        'model_class': DDPG,
        'params': {
            'learning_rate': 1e-3,
            'buffer_size': 1_000_000,
            'batch_size': 100,
            'gamma': 0.99,
            'tau': 0.005,
        }
    }
}

# =================================================================================
# --- 2. HELPER FUNCTIONS ---
# =================================================================================

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env_thunk(env_config: Dict[str, Any], max_cells: int, seed: int) -> Callable:
    """Creates a thunk (a zero-argument function) for SubprocVecEnv."""
    def _thunk() -> Monitor:
        env_config_copy = env_config.copy()
        env_config_copy['seed'] = seed
        env = FiveGEnv(env_config=env_config_copy, max_cells=max_cells)
        env = Monitor(env)
        return env
    return _thunk

def create_model(algorithm: str, env: VecNormalize, device: torch.device) -> BaseAlgorithm:
    """Factory function to create a model based on the chosen algorithm."""
    if algorithm not in HYPERPARAMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
        
    config = HYPERPARAMS[algorithm]
    model_class = config['model_class']
    params = config['params'].copy()
    
    # Apply linear schedule to learning rate
    params['learning_rate'] = linear_schedule(params['learning_rate'])
    
    return model_class(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=POLICY_KWARGS,
        device=device,
        tensorboard_log="sb3_logs/",
        verbose=1,
        **params
    )

def benchmark_algorithms(configs: list, max_cells: int, num_cpu: int, timesteps: int = 50_000):
    """Quickly benchmark different algorithms to find the most promising one."""
    print("\n" + "="*50)
    print("--- RUNNING ALGORITHM BENCHMARK ---")
    print("="*50)
    
    benchmark_config = configs[0] # Use the first scenario for a consistent benchmark
    algorithms_to_test = ['sac', 'ppo', 'td3']
    results = {}
    
    for algo in algorithms_to_test:
        print(f"\n--- Testing {algo.upper()} ---")
        try:
            env_thunks = [make_env_thunk(benchmark_config, max_cells, seed=i) for i in range(num_cpu)]
            env = SubprocVecEnv(env_thunks)
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
            
            model = create_model(algo, env, DEVICE)
            
            callback = AlgorithmComparisonCallback()
            model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
            
            results[algo] = {
                'mean_reward': np.mean(callback.episode_rewards) if callback.episode_rewards else -np.inf,
                'max_reward': np.max(callback.episode_rewards) if callback.episode_rewards else -np.inf
            }
            env.close()
        except Exception as e:
            print(f"  ERROR testing {algo.upper()}: {e}")
            results[algo] = {'mean_reward': -np.inf, 'max_reward': -np.inf}
    
    print("\n" + "="*50)
    print("--- BENCHMARK RESULTS (Mean Reward) ---")
    print("="*50)
    for algo, res in results.items():
        print(f"  {algo.upper():<5}: {res['mean_reward']:.3f}")
    
    best_algo = max(results, key=lambda k: results[k]['mean_reward'])
    print(f"\n🎯 Recommended Algorithm: {best_algo.upper()}")
    
    return best_algo


# =================================================================================
# --- 3. MAIN TRAINING SCRIPT ---
# =================================================================================

if __name__ == "__main__":
    print("--- Starting Unified 5G Agent Training Script ---")
    print(f"Using device: {DEVICE}")
    print(f"Max cells: {MAX_CELLS_SYSTEM_WIDE}, Parallel envs: {NUM_CPU_CORES}")

    # --- Load Scenarios ---
    scenario_folder = "scenarios/"
    try:
        scenario_files = [f for f in os.listdir(scenario_folder) if f.endswith('.json')]
        scenario_configs = []
        print("\nLoading scenarios:")
        for sf in sorted(scenario_files):
            with open(os.path.join(scenario_folder, sf), 'r') as f:
                config = json.load(f)
                scenario_configs.append(config)
                print(f"  - Loaded: {config.get('name', sf)}")
    except FileNotFoundError:
        print(f"ERROR: Scenario folder '{scenario_folder}' not found. Exiting.")
        exit()

    # --- Algorithm Selection ---
    print("\n" + "="*50 + "\n1. ALGORITHM SELECTION\n" + "="*50)
    print("1. SAC (Recommended for continuous control)")
    print("2. PPO (Stable on-policy alternative)")
    print("3. TD3 (Alternative to SAC)")
    print("4. DDPG (Older off-policy algorithm)")
    print("5. Run Benchmark (briefly trains SAC, PPO, TD3 to recommend the best)")
    
    choice = input("\nChoose algorithm or benchmark (default: 1): ").strip()
    if choice == '2': algorithm = 'ppo'
    elif choice == '3': algorithm = 'td3'
    elif choice == '4': algorithm = 'ddpg'
    elif choice == '5': algorithm = benchmark_algorithms(scenario_configs, MAX_CELLS_SYSTEM_WIDE, min(4, NUM_CPU_CORES))
    else: algorithm = 'sac'
    
    print(f"\n--- Preparing to train with {algorithm.upper()} ---")

    # --- Directory Setup ---
    log_dir, model_dir = "sb3_logs/", "sb3_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- Environment and Model Initialization ---
    env_thunks = [make_env_thunk(scenario_configs[i % len(scenario_configs)], MAX_CELLS_SYSTEM_WIDE, seed=i) for i in range(NUM_CPU_CORES)]
    
    model = None
    env = None
    continue_training = input("\nContinue training from a checkpoint? (y/n): ").strip().lower() == 'y'

    try:
        if continue_training:
            print("\nAvailable model checkpoints:")
            [print(f"  - {f}") for f in sorted(os.listdir(model_dir)) if f.endswith('.zip')]
            model_path = os.path.join(model_dir, input("Enter path to model checkpoint: ").strip())

            print("\nAvailable normalization stats files:")
            [print(f"  - {f}") for f in sorted(os.listdir(model_dir)) if f.endswith('.pkl')]
            stats_path = os.path.join(model_dir, input("Enter path to VecNormalize stats file: ").strip())

            base_env = SubprocVecEnv(env_thunks)
            env = VecNormalize.load(stats_path, base_env)
            
            model_class = HYPERPARAMS[algorithm]['model_class']
            model = model_class.load(model_path, env=env, device=DEVICE)
            print(f"\nSuccessfully loaded {algorithm.upper()} model and stats. Resuming training...")
        else:
            raise ValueError("Starting a new training session.")
    except (ValueError, FileNotFoundError, KeyError) as e:
        if continue_training: print(f"Could not load files: {e}.")
        print("\nStarting a new training session from scratch...")
        env = SubprocVecEnv(env_thunks)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        model = create_model(algorithm, env, DEVICE)

    # --- Training Execution ---
    timesteps_str = input(f"\nEnter total timesteps for this session (default 1,000,000): ").strip()
    total_timesteps = int(float(timesteps_str.replace(',', ''))) if timesteps_str else 1_000_000
    
    run_name_prefix = f"{algorithm}_mixed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    
    # --- Callbacks for robust training ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // NUM_CPU_CORES, 1), save_path=model_dir, name_prefix=run_name_prefix
    )
    constraint_monitor_callback = ConstraintMonitorCallback()
    eval_env = VecNormalize(SubprocVecEnv([make_env_thunk(scenario_configs[0], MAX_CELLS_SYSTEM_WIDE, seed=99)]), training=False, norm_reward=False)
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=os.path.join(model_dir, f"best_model_{algorithm}"),
        log_path=log_dir, eval_freq=max(25000 // NUM_CPU_CORES, 1),
        deterministic=True, render=False
    )
    
    print(f"\n{'='*60}\n--- TRAINING {algorithm.upper()} FOR {total_timesteps:,} TIMESTEPS ---\n{'='*60}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, constraint_monitor_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=not continue_training
    )
    print("\n--- Training Session Complete ---")

    # --- Save Final Model and Stats ---
    kaggle_save_path = "/kaggle/working/"
    if not os.path.exists(kaggle_save_path): kaggle_save_path = model_dir # Fallback for local run

    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = os.path.join(kaggle_save_path, f"{algorithm}_final_{time_stamp}.zip")
    model.save(final_model_path)

    stats_path = os.path.join(kaggle_save_path, f"vec_normalize_final_{time_stamp}.pkl")
    env.save(stats_path)
    
    print(f"\n✅ Final generalized model saved to: {final_model_path}")
    print(f"   Normalization stats saved to: {stats_path}")

    # --- Final Quick Evaluation ---
    print("\n--- Running Quick Evaluation on Final Model ---")
    obs, _ = eval_env.reset()
    done = False
    total_reward, steps = 0, 0
    while not done and steps < 500: # 500 steps for a more thorough test
        action, _ = model.predict(eval_env.normalize_obs(obs), deterministic=True)
        obs, reward, done, _ = eval_env.step(action)
        total_reward += reward
        steps += 1
    
    print(f"Evaluation complete over {steps} steps. Average reward: {total_reward/steps:.3f}")
    if total_reward < 0:
        print("⚠️  Warning: Final model has negative average reward in evaluation.")
    else:
        print("🎉 Final model shows promising performance!")