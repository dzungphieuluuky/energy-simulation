# enhanced_training_pipeline.py
"""
Enhanced 5G Network Training Pipeline
=====================================
A comprehensive training system with multiple parallel improvements
for safe and efficient 5G network optimization.
"""

import json
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import argparse
import gymnasium as gym
import multiprocessing
import torch_directml
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

# Stable Baselines3 components
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.base_class import BaseAlgorithm


# Custom components
from fiveg_env import FiveGEnv
from custom_models_sb3 import *
from callback import *
from wrapper import *
from initial import *

LAGRANGE_LR = 0.005

@dataclass
class TrainingConfig:
    max_cells: int = 57
    state_dim_per_cell: int = 12
    constraint_keys: List[str] = None
    device: torch.device = None
    
    # SAC-specific
    use_her: bool = False  # HER is for PPO, not SAC
    
    def __post_init__(self):
        if self.constraint_keys is None:
            self.constraint_keys = ['avg_drop_rate', 'avg_latency', 'cpu_violations', 'prb_violations']
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
class TrainingPipeline:
    """
    Comprehensive training pipeline with multiple improvement strategies.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.hyperparameters = self._setup_hyperparameters()
        self.setup_logging("training_log.log")
        
    def _setup_hyperparameters(self) -> Dict[str, Any]:
        """Setup algorithm-specific hyperparameters."""
        base_params = {
            'policy_kwargs': {
                'features_extractor_class': EnhancedAttentionNetwork,
                'features_extractor_kwargs': {
                    'features_dim': 256,
                    'max_cells': self.config.max_cells,
                    'n_cell_features': self.config.state_dim_per_cell,
                },
                'net_arch': dict(pi=[256, 256], qf=[256, 256], vf=[256, 256]),
                'share_features_extractor': True
            },
            'device': self.config.device,
            'tensorboard_log': "enhanced_logs/",
            'verbose': 1
        }
        
        algorithm_params = {
            'ppo': {
                'learning_rate': 1e-6,
                'n_steps': 8192,
                'batch_size': 512,
                'n_epochs': 5,
                'gamma': 0.995,
                'gae_lambda': 0.92,
                'clip_range': 0.1,
                'ent_coef': 0.001,
                'vf_coef': 0.8,
                'max_grad_norm': 0.5,
                'target_kl': 0.03
            },
            # In your _setup_hyperparameters function
            'sac': {
                'learning_rate': 1e-5,
                'buffer_size': 300000,       # Reduced
                'batch_size': 512,
                'ent_coef': 'auto',
                'gamma': 0.99,
                'tau': 0.01,
                'gradient_steps': 18,
                'learning_starts': 20000,    # Reduced
                'train_freq': 1,             # Update every step
                'policy_kwargs': {
                    'features_extractor_class': EnhancedAttentionNetwork,
                    'features_extractor_kwargs': {
                        'features_dim': 256,
                        'max_cells': self.config.max_cells,
                        'n_cell_features': self.config.state_dim_per_cell,
                    },
                    'net_arch': dict(pi=[256, 256], qf=[256, 256]),
                }
            }
        }
        return {algo: {**base_params, **params} for algo, params in algorithm_params.items()}
    
    def create_environment(self, env_config: Dict[str, Any], seed: int, name_env: str = "gated") -> Callable:
        """Create environment factory function."""
        create_environment_from_name = {
            "her": SimplifiedHERForPPO,
            "gated": GatedRewardWrapper,
            "strict": StrictConstraintWrapper,
            "lagrange": LagrangianRewardWrapper,
            "cost_lagrange": lambda env: LagrangianCostWrapper(
                env, 
                constraint_thresholds={
                    'avg_drop_rate': self.config.constraint_keys.get('avg_drop_rate', 1.0),
                    'avg_latency': 50.0,
                    'cpu_violations': 0.0,  # Violation count, not usage
                    'prb_violations': 0.0
                }
            )
        }
        
        def _make_env() -> gym.Env:
            base_env = FiveGEnv(env_config, self.config.max_cells)
            wrapper_class = create_environment_from_name.get(name_env)
            if wrapper_class:
                if callable(wrapper_class):
                    wrapped_env = wrapper_class(base_env)
                else:
                    wrapped_env = wrapper_class  # Lambda case
            else:
                wrapped_env = base_env
            monitored_env = Monitor(wrapped_env)
            return monitored_env
        return _make_env

    # --- CHANGE 1: Modified the 'train' method signature ---
    def train(self, algorithm: str, total_timesteps: int, n_envs: int = 4, name_env: str = "default", load_model_path: Optional[str] = None) -> BaseAlgorithm:
        """Execute the complete training pipeline."""
        print(f"🚀 Starting enhanced training with {algorithm.upper()}")
        
        scenarios = self._load_scenarios()
        self.logger.info(f"Loaded {len(scenarios)} training scenarios.")

        env_factories = [self.create_environment(scenarios[i % len(scenarios)], seed=i, name_env=name_env) 
                        for i in range(n_envs)]
        self.logger.info(f"Creating {n_envs} parallel environments.")
        self.logger.info(f"Using {name_env.upper()} environment wrapper.")
        subproc_vec_env = SubprocVecEnv(env_factories)
        vec_env = VecNormalize(subproc_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        self.logger.info(f"Normalized observation: {vec_env.norm_obs}")
        self.logger.info(f"Normalized reward: {vec_env.norm_reward}")
        
        model_class = {'ppo': PPO, 'sac': SAC, 'td3': TD3, 'ddpg': DDPG}[algorithm]

        # --- CHANGE 2: Added logic to load a model or create a new one ---
        if load_model_path:
            self.logger.info(f"Loading pre-trained model from: {load_model_path}")
            model = model_class.load(load_model_path, env=vec_env)
            
            stats_path = load_model_path.replace(".zip", "_vecnormalize.pkl")
            if os.path.exists(stats_path):
                self.logger.info(f"Loading VecNormalize stats from: {stats_path}")
                vec_env = VecNormalize.load(stats_path, vec_env)
                vec_env.training = True
            else:
                self.logger.warning(f"VecNormalize stats not found at {stats_path}. Model may underperform.")
        else:
            self.logger.info("Creating a new model from scratch.")
            model = model_class("MlpPolicy", vec_env, **self.hyperparameters[algorithm])
            
            print("\n" + "="*50)
            print("Applying custom policy initialization...")
            bias_policy_output(model, target_action=0.9)
            print("="*50 + "\n")

        self.model = model
        self.logger.info(f"Initialized {algorithm.upper()} model.")
        self.logger.info(f"Model architecture:\n {model.policy}")

        self.logger.info("Model hyperparameters:")
        # ... (rest of the function is the same)
        for key, value in self.hyperparameters[algorithm].items():
            if key == 'device':
                self.logger.info(f"  {key}: {value}")
            elif key == 'policy_kwargs':
                self.logger.info(f"  {key}:")
                self.logger.info(f"    features_extractor: {value['features_extractor_class'].__name__}")
                self.logger.info(f"    features_dim: {value['features_extractor_kwargs']['features_dim']}")
                self.logger.info(f"    net_arch: {value['net_arch']}")
            else:
                self.logger.info(f"  {key}: {value}")

        callbacks = self._setup_callbacks(load_model_path)
        self.logger.info(f"Using the following callbacks: {json.dumps([type(cb).__name__ for cb in callbacks], indent=4)}")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not load_model_path # If loading, don't reset timestep count
        )
        
        self._save_model(model, algorithm, vec_env)
        
        return model
    
    def _load_scenarios(self) -> List[Dict[str, Any]]:
        """Load training scenarios from configuration files."""
        # ... (function is the same)
        scenario_folder = "scenarios/"
        scenarios = []
        try:
            for filename in os.listdir(scenario_folder):
                if filename.endswith('.json'):
                    with open(os.path.join(scenario_folder, filename), 'r') as f:
                        scenario = json.load(f)
                        scenarios.append(scenario)
                        print(f"Loaded scenario: {scenario.get('name', filename)}")
        except FileNotFoundError:
            print(f"Scenario folder '{scenario_folder}' not found, using default")
            scenarios = [{'name': 'default', 'numUEs': 50, 'numSites': 3}]
        return scenarios
    
    def _setup_callbacks(self, load_model_path: Optional[str] = None) -> List:
        """Setup training `callbacks`."""
        # ... (function is the same)
        eval_env_factory = self.create_environment({}, seed=999)
        eval_env = DummyVecEnv([eval_env_factory])
        
        # When loading a model, we must use the same normalization stats for the eval env
        if load_model_path:
            stats_path = load_model_path.replace(".zip", "_vecnormalize.pkl")
            if os.path.exists(stats_path):
                print(f"Loading VecNormalize stats for evaluation from {stats_path}")
                eval_vec_env = VecNormalize.load(stats_path, eval_env)
                eval_vec_env.training = False # Do not update stats during evaluation
            else:
                eval_vec_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        else:
            eval_vec_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        callbacks = [
            CheckpointCallback(save_freq=50000, save_path='checkpoints/'),
            MetricsLoggerCallback(verbose=1),
            LoggedEvalCallback(
                eval_env=eval_vec_env,
                best_model_save_path='best_models/',
                log_path='eval_logs/',
                eval_freq=10000,
            ),
            ConstraintMonitorCallback(verbose=1),
        ]
        return callbacks
    
    # --- CHANGE 3: Modified the '_save_model' method to also save VecNormalize stats ---
    def _save_model(self, model: BaseAlgorithm, algorithm: str, vec_env: VecNormalize):
        """Save trained model and VecNormalize stats."""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs("trained_models/", exist_ok=True)
        model_path = f"trained_models/{algorithm}_final_{timestamp}.zip"
        model.save(model_path)
        print(f"Model saved to: {model_path}")

        stats_path = model_path.replace(".zip", "_vecnormalize.pkl")
        vec_env.save(stats_path)
        print(f"VecNormalize stats saved to: {stats_path}")

    def setup_logging(self, log_file):
        # ... (function is the same)
        self.logger = logging.getLogger('RLAgentTraining')
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler())


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Enhanced 5G Training Pipeline")
    parser.add_argument("--algorithm", "-a", type=str, default="ppo", 
                       choices=['ppo', 'sac', 'td3', 'ddpg'], help="RL algorithm to use")
    parser.add_argument("--timesteps", "-t", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--envs", "-e", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--name_env", "-n", type=str, default="gated", help="Environment wrapper to use")
    
    # --- CHANGE 4: Added the new command-line argument ---
    parser.add_argument("--load-model", type=str, default=None, 
                        help="Path to a pre-trained model to continue training from (e.g., 'best_models/best_model.zip')")
    
    args = parser.parse_args()
    
    config = TrainingConfig()
    pipeline = TrainingPipeline(config)

    print("Algorithm selected:", args.algorithm.upper())
    
    # --- CHANGE 5: Pass the new argument to the train method ---
    pipeline.train(
        algorithm=args.algorithm, 
        total_timesteps=args.timesteps, 
        n_envs=args.envs, 
        name_env=args.name_env,
        load_model_path=args.load_model
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()