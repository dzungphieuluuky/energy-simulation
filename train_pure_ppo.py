# train_ppo_pure.py
import numpy as np
import torch
import os
import json
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from fiveg_env import FiveGEnv

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
            # Assign a unique seed to each parallel environment
            config_for_this_env['seed'] = config_for_this_env.get('seed', 42) + rank
            
            env = FiveGEnv(config_for_this_env)
            env = Monitor(env)
            return env
        return _init
    
    vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = vec_env_cls([make_env(i) for i in range(n_envs)])
    
    # Normalize rewards only for the training environment
    norm_reward = not is_eval
    env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=10., gamma=0.99)
    return env

def main():
    # --- 1. Configuration ---
    TOTAL_TIMESTEPS = 2_000_000  # Give it enough time to learn from scratch
    N_ENVS = 10  # Use more parallel environments if you have the cores
    SCENARIO_FOLDER = "scenarios"
    
    # Create necessary directories
    os.makedirs('sb3_models', exist_ok=True)
    os.makedirs('sb3_logs', exist_ok=True)
    
    # --- 2. Load All Scenario Configs ---
    try:
        scenario_configs = load_scenario_configs(SCENARIO_FOLDER)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}"); return

    # --- 3. Create the Training and Evaluation Environments ---
    print("\nInitializing normalized, parallel environments for PPO training...")
    env = create_training_env(scenario_configs, n_envs=N_ENVS)
    
    print("Initializing normalized evaluation environment...")
    # Use only the first scenario for consistent evaluation
    eval_env = create_training_env([scenario_configs[0]], n_envs=1, is_eval=True)
    
    # CRITICAL: Freeze the eval env and sync its stats
    eval_env.training = False
    eval_env.obs_rms = env.obs_rms
    
    # --- 4. Initialize PPO Model for Training from Scratch ---
    # These hyperparameters are robust defaults for PPO.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,      # Standard learning rate for training from scratch
        n_steps=2048,            # Collect more data before each update
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,           # Encourage exploration initially
        verbose=1,
        tensorboard_log="sb3_logs/PPO_Mixed/",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # --- 5. Set up Callbacks for Saving and Evaluation ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='sb3_models_pure_ppo/best_model/',
        log_path='sb3_logs/PPO_Eval/',
        eval_freq=max(15000 // N_ENVS, 1), # Evaluate more frequently
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // N_ENVS, 1),
        save_path='sb3_models/',
        name_prefix='ppo_mixed'
    )
    
    # --- 6. Start Reinforcement Learning ---
    print(f"\n{'='*70}\nSTARTING PURE REINFORCEMENT LEARNING (FROM SCRATCH)\n{'='*70}")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # --- 7. Save Final Model and Normalization Stats ---
    model.save("sb3_models/ppo_mixed_final")
    env.save("sb3_models/vec_normalize_mixed_final.pkl")
    print(f"\n🎉 PURE PPO TRAINING COMPLETE! Model and normalization stats saved.")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    # This is still recommended for SubprocVecEnv on Windows/macOS
    multiprocessing.freeze_support()
    main()