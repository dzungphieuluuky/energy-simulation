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
                'features_extractor_class': LightweightAttentionNetwork,
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
                'learning_rate': 3e-4,       # Was 1e-6, which is too slow to learn effectively.
                'buffer_size': 300000,
                'batch_size': 256,
                'ent_coef': 'auto',      # Encourage exploration more initially.
                'gamma': 0.99,
                'tau': 0.005,
                'train_freq': (1, 'step'),   # More stable training.
                'gradient_steps': 1,
                'learning_starts': 10000,    # Start learning sooner.
            }
        }
        return {algo: {**base_params, **params} for algo, params in algorithm_params.items()}
    
    def create_environment(self, env_config: Dict[str, Any], seed: int, name_env = "gated") -> Callable:
        """Create environment factory function with mandatory normalization."""
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
            # 1. Start with the base environment
            base_env = FiveGEnv(env_config, self.config.max_cells)
            wrapper_class = create_environment_from_name.get(name_env)
            if wrapper_class:
                if callable(wrapper_class):
                    wrapped_env = wrapper_class(base_env)
                else:
                    wrapped_env = wrapper_class  # Lambda case
            else:
                wrapped_env = base_env

            # 3. Apply the mandatory state normalizer wrapper
            normalized_env = StateNormalizerWrapper(wrapped_env)
            
            # 4. Monitor the final environment
            monitored_env = Monitor(normalized_env)
            return monitored_env
        return _make_env

    # --- CHANGE 1: Modified the 'train' method signature ---
    def train(self, algorithm: str, total_timesteps: int, n_envs: int, load_model_path: Optional[str] = None, name_env = "gated"):
        print(f"🚀 Starting training with {algorithm.upper()} using StateNormalizer.")
        
        scenarios = self._load_scenarios()
        self.logger.info(f"Loaded {len(scenarios)} training scenarios.")

        env_factories = [self.create_environment(scenarios[i % len(scenarios)], seed=i, name_env=name_env) 
                        for i in range(n_envs)]
        self.logger.info(f"Creating {n_envs} parallel environments.")
        
        # --- CRITICAL CHANGE: REMOVED VecNormalize ---
        # We now use a standard SubprocVecEnv. The normalization is handled
        # inside each environment instance by the StateNormalizerWrapper.
        vec_env = SubprocVecEnv(env_factories)
        
        model_class = {'ppo': PPO, 'sac': SAC}[algorithm]

        if load_model_path:
            self.logger.info(f"Loading pre-trained model from: {load_model_path}")
            # The environment passed to load must have the same wrappers
            model = model_class.load(load_model_path, env=vec_env)
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

        callbacks = self._setup_callbacks(name_env=name_env)
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not load_model_path
        )
        
        self._save_model(model, algorithm)
        return model

    
    def _load_scenarios(self) -> List[Dict[str, Any]]:
        """Load training scenarios from configuration files."""
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

    def _setup_callbacks(self, name_env) -> List[BaseCallback]:
        """Setup callbacks, now without VecNormalize."""
        eval_env_factory = self.create_environment({}, seed=999, name_env=name_env)
        eval_env = DummyVecEnv([eval_env_factory])

        callbacks = [
            CheckpointCallback(save_freq=50000, save_path='checkpoints/', name_prefix='fiveg_rl_model'),
            MetricsLoggerCallback(verbose=1),
            LoggedEvalCallback(
                eval_env=eval_env, # Use the non-normalized eval env
                best_model_save_path='best_models/',
                log_path='eval_logs/',
                eval_freq=20000,
            ),
            ConstraintMonitorCallback(verbose=1),
        ]

        if name_env == "cost_lagrange":
            print("Lagrangian wrapper detected. Adding AdamLambdaUpdateCallback.")
            lagrangian_callback = ClaudeAdamLambdaUpdateCallback(
                 constraint_thresholds={'avg_drop_rate': 1.0, 
                                        'avg_latency': 50.0, 
                                        'cpu_violations': 0.0, 
                                        'prb_violations': 0.0},
                 initial_lambda_value=2.0,
                 lambda_lr=0.005,
                 update_freq=2048,
            )
            callbacks.append(lagrangian_callback)

        return callbacks
    
    def _save_model(self, model: BaseAlgorithm, algorithm: str):
        """Save trained model. No need to save stats anymore."""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs("trained_models/", exist_ok=True)
        model_path = f"trained_models/{algorithm}_final_{timestamp}.zip"
        model.save(model_path)
        print(f"Model saved to: {model_path}")


    def setup_logging(self, log_file):
        self.logger = logging.getLogger('RLAgentTraining')
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler())


def main():
    parser = argparse.ArgumentParser(description="5G Training Pipeline (StateNormalizer Compliant)")
    parser.add_argument("--algorithm", "-a", type=str, default="sac", choices=['ppo', 'sac'], help="RL algorithm to use")
    parser.add_argument("--timesteps", "-t", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--envs", "-e", type=int, default=multiprocessing.cpu_count(), help="Number of parallel environments")
    parser.add_argument("--load-model", type=str, default=None, help="Path to a pre-trained model to continue training")
    parser.add_argument("--name-env", "-n", type=str, default="gated", choices=['her', 'gated', 'strict', 'lagrange', 'cost_lagrange'], help="Type of environment wrapper to use")
    args = parser.parse_args()
    
    config = TrainingConfig()
    pipeline = TrainingPipeline(config)
    
    pipeline.train(
        algorithm=args.algorithm, 
        total_timesteps=args.timesteps, 
        n_envs=args.envs, 
        load_model_path=args.load_model,
        name_env=args.name_env
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
