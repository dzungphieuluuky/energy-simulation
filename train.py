# train_unified.py
import json
import torch
import os
import pandas as pd
import numpy as np
import argparse
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
    net_arch=dict(pi=[256, 256], qf=[256, 256], vf=[256, 256]) # For actor-critic models
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
            'n_steps': 4096,
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.995,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.005,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': 0.02,
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

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Unified training script for the 5G environment agent.")
    
    parser.add_argument(
        "--algorithm", "-a", type=str, default="sac",
        choices=['sac', 'ppo', 'td3', 'ddpg'],
        help="The reinforcement learning algorithm to train. Default: 'sac'."
    )
    parser.add_argument(
        "--total-timesteps", "-t", type=float, default=1_000_000,
        help="Total number of timesteps for the training session. Supports scientific notation (e.g., 1e6). Default: 1,000,000."
    )
    parser.add_argument(
        "--n-envs", type=int, default=os.cpu_count(),
        help=f"Number of parallel environments to use. Default: {os.cpu_count()}."
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="If set, run a short benchmark of SAC, PPO, and TD3 to find the best algorithm, then exit."
    )
    parser.add_argument(
        "--continue-from-model", type=str, default=None,
        help="Path to a .zip model file to continue training from."
    )
    parser.add_argument(
        "--continue-from-stats", type=str, default=None,
        help="Path to a .pkl VecNormalize stats file to use when continuing training. Required if --continue-from-model is set."
    )
    parser.add_argument(
        "--log-folder", type=str, default="sb3_logs/",
        help="Directory to save TensorBoard logs. Default: 'sb3_logs/'."
    )
    parser.add_argument(
        "--model-folder", type=str, default="sb3_models/",
        help="Directory to save model checkpoints and final models. Default: 'sb3_models/'."
    )
    
    args = parser.parse_args()
    
    # --- Argument validation ---
    if args.continue_from_model and not args.continue_from_stats:
        parser.error("--continue-from-stats is required when using --continue-from-model.")
    if args.continue_from_stats and not args.continue_from_model:
        parser.error("--continue-from-model is required when using --continue-from-stats.")
        
    return args


# =================================================================================
# --- 3. MAIN TRAINING SCRIPT ---
# =================================================================================

def main(args: argparse.Namespace):
    """Main function to run the training pipeline."""
    print("--- Starting Unified 5G Agent Training Script ---")
    print(f"Using device: {DEVICE}")
    print(f"Max cells: {MAX_CELLS_SYSTEM_WIDE}, Parallel envs: {args.n_envs}")

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

    # --- Algorithm Selection & Benchmarking ---
    if args.benchmark:
        benchmark_algorithms(scenario_configs, MAX_CELLS_SYSTEM_WIDE, min(4, args.n_envs))
        print("\nBenchmark complete. Exiting script.")
        return # Exit after benchmarking

    algorithm = args.algorithm
    print(f"\n--- Preparing to train with {algorithm.upper()} ---")

    # --- Directory Setup ---
    log_dir, model_dir = args.log_folder, args.model_folder
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- Environment and Model Initialization ---
    env_thunks = [make_env_thunk(scenario_configs[i % len(scenario_configs)], MAX_CELLS_SYSTEM_WIDE, seed=i) for i in range(args.n_envs)]
    
    continue_training = bool(args.continue_from_model)

    if continue_training:
        print(f"\nAttempting to load model from: {args.continue_from_model}")
        print(f"Attempting to load stats from: {args.continue_from_stats}")
        try:
            base_env = SubprocVecEnv(env_thunks)
            env = VecNormalize.load(args.continue_from_stats, base_env)
            
            model_class = HYPERPARAMS[algorithm]['model_class']
            model = model_class.load(args.continue_from_model, env=env, device=DEVICE)
            print(f"\nSuccessfully loaded {algorithm.upper()} model and stats. Resuming training...")
        except (FileNotFoundError, KeyError) as e:
            print(f"\nFATAL ERROR: Could not load files: {e}. Exiting.")
            return
    else:
        print("\nStarting a new training session from scratch...")
        env = SubprocVecEnv(env_thunks)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        model = create_model(algorithm, env, DEVICE)

    # --- Training Execution ---
    total_timesteps = int(args.total_timesteps)
    run_name_prefix = f"{algorithm}_mixed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    
    # --- Callbacks for robust training ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args.n_envs, 1), save_path=model_dir, name_prefix=run_name_prefix
    )
    constraint_monitor_callback = ConstraintMonitorCallback()

    eval_env = VecNormalize(SubprocVecEnv([make_env_thunk(scenario_configs[0], MAX_CELLS_SYSTEM_WIDE, seed=99)]), training=False, norm_reward=False)
    eval_env.obs_rms = env.obs_rms
    eval_env.training = False

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.join(model_dir, f"best_model_{algorithm}"),
        log_path=log_dir, 
        eval_freq=max(25000 // args.n_envs, 1),
        deterministic=True, 
        render=False
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
    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = os.path.join(model_dir, f"{algorithm}_final_{time_stamp}.zip")
    model.save(final_model_path)

    stats_path = os.path.join(model_dir, f"vec_normalize_final_{time_stamp}.pkl")
    env.save(stats_path)
    
    print(f"\n✅ Final generalized model saved to: {final_model_path}")
    print(f"   Normalization stats saved to: {stats_path}")

    # --- Final Quick Evaluation ---
    print("\n--- Running Quick Evaluation on Final Model ---")
    obs, _ = eval_env.reset()
    done = False
    total_reward, steps = 0, 0
    while not done and steps < 500: # 500 steps for a more thorough test
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = eval_env.step(action)
        total_reward += reward
        steps += 1
    
    print(f"Evaluation complete over {steps} steps. Average reward: {total_reward/steps:.3f}")
    if total_reward < 0:
        print("⚠️  Warning: Final model has negative average reward in evaluation.")
    else:
        print("🎉 Final model shows promising performance!")

if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)