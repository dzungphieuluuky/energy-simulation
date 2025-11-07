# train.py (Offline Trainer)

from dataclasses import dataclass
import torch
import os
import argparse
import gymnasium as gym
import pickle
from typing import Dict, Any

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from custom_models_sb3 import LightweightAttentionNetwork
from initial import bias_policy_output

BUFFER_LOAD_PATH = "collected_replay_buffer.pkl"
MODEL_SAVE_NAME = "best_model" # Will be saved as best_model.zip

@dataclass
class TrainingConfig:
    max_cells: int = 57
    state_dim_per_cell: int = 12
    device: torch.device = None
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OfflineTrainer:
    def __init__(self, config: TrainingConfig, train_steps: int, load_model_path: str):
        self.config = config
        self.hyperparameters = self._setup_hyperparameters()
        self.train_steps = train_steps
        self.load_model_path = load_model_path

    def _setup_hyperparameters(self) -> Dict[str, Any]:
        policy_kwargs = {
            'features_extractor_class': LightweightAttentionNetwork,
            'features_extractor_kwargs': {'features_dim': 256},
            'net_arch': dict(pi=[256, 256], qf=[256, 256]),
            'share_features_extractor': True
        }
        return {
            'policy': "MlpPolicy", 
            'policy_kwargs': policy_kwargs, 
            'device': self.config.device,
            'learning_rate': 1e-6, 
            'buffer_size': 500000, 
            'batch_size': 512,
            'ent_coef': 'auto', 
            'gamma': 0.99, 
            'tau': 0.005, 
            'gradient_steps': 1,
            'learning_starts': 0 # We start learning immediately from the buffer
        }

    def train(self):
        print("="*50)
        print("OFFLINE TRAINING MODE ACTIVATED")
        print("="*50)

        # 1. Load the collected experience data
        if not os.path.exists(BUFFER_LOAD_PATH):
            print(f"FATAL: Replay buffer file not found at '{BUFFER_LOAD_PATH}'.")
            print("Please run the simulation first with rl_agent.py to collect data.")
            return
        print(f"Loading experiences from '{BUFFER_LOAD_PATH}'...")
        with open(BUFFER_LOAD_PATH, "rb") as f:
            collected_experiences = pickle.load(f)
        print(f"Loaded {len(collected_experiences)} transitions.")

        # 2. Setup a dummy environment and the SAC model
        # The environment is only a placeholder for the model's architecture
        dummy_env = gym.make("Pendulum-v1")
        
        if self.load_model_path and os.path.exists(self.load_model_path):
            print(f"Loading model to continue training from '{self.load_model_path}'")
            model = SAC.load(self.load_model_path, env=dummy_env, device=self.config.device)
        else:
            print("Creating a new SAC model for offline training.")
            model = SAC(env=dummy_env, **self.hyperparameters)
            # Apply bias only if creating a brand new model
            bias_policy_output(model, target_action=0.95)
        
        # 3. Populate the model's replay buffer with the loaded data
        print("Populating model's replay buffer...")
        for exp in collected_experiences:
            # The 'exp' is a tuple (obs, next_obs, action, reward, done, info)
            model.replay_buffer.add(*exp)
        print(f"Buffer populated. Current size: {model.replay_buffer.size()}/{model.replay_buffer.buffer_size}")
        
        # 4. Perform offline training for the specified number of gradient steps
        print(f"Starting offline training for {self.train_steps} gradient steps...")
        model.train(gradient_steps=self.train_steps, batch_size=self.hyperparameters['batch_size'])
        print("✅ Offline training complete.")

        # 5. Save the newly trained model
        self._save_model(model, MODEL_SAVE_NAME)

    def _save_model(self, model, name):
        model_path = f"{name}.zip"
        model.save(model_path)
        print("\n" + "="*50)
        print(f"✅ Trained model saved to '{model_path}'.")
        print("You can now copy this file to the collector agent for the next run.")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline SAC Trainer for 5G Energy Challenge")
    parser.add_argument("--train-steps", "-ts", type=int, default=200000, help="Total number of gradient steps to perform during offline training.")
    parser.add_argument("--load-model", type=str, default=f"{MODEL_SAVE_NAME}.zip", help="Path to a pre-trained model to continue training from.")
    args = parser.parse_args()
    
    config = TrainingConfig()
    trainer = OfflineTrainer(config, args.train_steps, args.load_model)
    trainer.train()