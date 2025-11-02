# train_sb3.py
import json
import torch
import os
import pandas as pd
from typing import Callable, List
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import numpy as np

from fiveg_env_chatgpt import FiveGEnv
from custom_models_sb3 import EnhancedAttentionNetwork

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env_thunk(env_config, max_cells):
    """Creates a thunk (a zero-argument function) for SubprocVecEnv."""
    def _thunk():
        env = FiveGEnv(env_config=env_config, max_cells=max_cells)
        # Wrap with Monitor to track ep_rew_mean and other stats for logging
        env = Monitor(env)
        return env
    return _thunk

class TrainingProgressCallback(BaseCallback):
    """
    A custom callback to track training progress and update the environment if needed.
    The new FiveGEnv handles reward weights internally based on episode count.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # The new environment handles reward weights internally based on self.total_episodes
        # We don't need to manually set reward weights anymore
        return True

if __name__ == "__main__":
    print("Starting Stable Baselines3 5G Agent Training with Optimized Environment...")
    
    # --- 1. System and Training Parameters ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    
    MAX_CELLS_SYSTEM_WIDE = 57
    STATE_DIM_PER_CELL = 25
    num_cpu = 5
    
    print(f"\nTraining with a fixed maximum of {MAX_CELLS_SYSTEM_WIDE} cells.")
    print(f"Using {num_cpu} parallel environments.")

    # --- 2. Load ALL Scenario Configurations ---
    scenario_folder = "scenarios/"
    scenario_files = [f for f in os.listdir(scenario_folder) if f.endswith('.json')]
    if not scenario_files:
        raise FileNotFoundError(f"No scenario files found in '{scenario_folder}' directory.")
        
    scenario_configs = []
    print("\nLoading all scenarios for mixed training:")
    for sf in sorted(scenario_files):
        with open(os.path.join(scenario_folder, sf), 'r') as f:
            config = json.load(f)
            scenario_configs.append(config)
            print(f"  - Loaded: {config.get('name', sf)}")

    # --- 3. Network Architecture Selection ---
    policy_kwargs = dict(
        features_extractor_class=EnhancedAttentionNetwork,
        features_extractor_kwargs=dict(
            features_dim=256,
            max_cells=MAX_CELLS_SYSTEM_WIDE,
            n_cell_features=STATE_DIM_PER_CELL,
        ),
    )

    # --- 4. Environment and Model Initialization (with Continue Logic) ---
    log_dir, model_dir = "sb3_logs/", "sb3_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model, env = None, None
    continue_training = input("\nContinue training from a checkpoint? (y/n): ").strip().lower() == 'y'

    env_thunks = []
    for i in range(num_cpu):
        config = scenario_configs[i % len(scenario_configs)]
        env_thunks.append(make_env_thunk(config, MAX_CELLS_SYSTEM_WIDE))

    try:
        if continue_training:
            print("\nAvailable model checkpoints:")
            for f in sorted(os.listdir(model_dir)):
                if f.endswith('.zip'): print(f"  - {f}")
            model_path_name = input("Enter path to model checkpoint file: ").strip()

            print("\nAvailable normalization stats files:")
            for f in sorted(os.listdir(model_dir)):
                if f.endswith('.pkl'): print(f"  - {f}")
            env_path_name = input("Enter path to VecNormalize stats file: ").strip()

            base_env = SubprocVecEnv(env_thunks)
            env = VecNormalize.load(os.path.join(model_dir, env_path_name), base_env)
            
            model = SAC.load(
                os.path.join(model_dir, model_path_name),
                env=env, device=device,
                learning_rate=linear_schedule(1e-4) # Use a smaller LR for fine-tuning
            )
            print("\nModel and normalization stats loaded successfully. Resuming training...")
        else:
            raise ValueError("Starting a new training session.") 
    except (ValueError, FileNotFoundError) as e:
        if continue_training:
            print(f"Error loading files: {e}.")
        print("\nStarting a new training session from scratch...")
        env = SubprocVecEnv(env_thunks)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        model = SAC(
            policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(3e-4), 
            buffer_size=1_000_000, batch_size=512, 
            tau=0.005, gamma=0.99,
            train_freq=1, gradient_steps=1,
            ent_coef='auto', target_entropy='auto',
            target_update_interval=1, verbose=1, tensorboard_log="sb3_logs/", device=device
        )

    # --- 5. Training ---
    total_timesteps_str = input("Enter total timesteps for this training session (default 2,000,000): ").strip()
    try:
        total_timesteps = int(float(total_timesteps_str.replace(',', '')))
    except ValueError:
        total_timesteps = 2_000_000
    
    run_name_prefix = f"sac_mixed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // num_cpu, 1), save_path=model_dir, name_prefix=run_name_prefix, verbose=1
    )
    progress_callback = TrainingProgressCallback()
    
    print(f"\n--- Starting/Resuming Training for {total_timesteps:,} timesteps ---")
    print("Note: Environment now uses internal adaptive reward weighting based on episode count.")
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[checkpoint_callback, progress_callback], 
        progress_bar=True,
        reset_num_timesteps=not continue_training
    )
    print("\n--- Training Session Complete ---")

    # --- 6. Save Final Model and Stats ---
    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    network_name = policy_kwargs['features_extractor_class'].__name__
    final_model_path = os.path.join(model_dir, f"sac_mixed_{network_name}_{time_stamp}.zip")
    model.save(final_model_path)

    stats_path = os.path.join(model_dir, f"vec_normalize_{time_stamp}.pkl")
    env.save(stats_path)
    
    print(f"\nFinal generalized model saved to: {final_model_path}")
    print(f"Normalization stats saved to: {stats_path}")

