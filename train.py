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
from custom_models_sb3 import EnhancedAttentionNetwork
from callback import *
from wrapper import *

LAGRANGE_LR = 0.005


def bias_policy_output(model: BaseAlgorithm, bias_value: float, weight_std: float = 0.01):
    """
    Initializes the output layer of an SB3 policy to a specific bias.
    This encourages the agent to start with a known "safe" action.

    Args:
        model: The SB3 model (PPO or SAC).
        bias_value: The target logit value (pre-tanh). A value of ~0.7 targets an action of 0.8.
        weight_std: The standard deviation for the final layer's weights. Should be small.
    """
    final_layer = None
    
    # --- PPO Case ---
    # For PPO's MlpPolicy, the final layer mapping features to actions is `action_net`.
    # This is a single nn.Linear layer, NOT a Sequential module.
    if hasattr(model.policy, "action_net") and isinstance(model.policy.action_net, nn.Linear):
        print("Biasing the final layer of PPO's action_net.")
        final_layer = model.policy.action_net

    # --- SAC Case ---
    # For SAC's MlpPolicy, the actor network ends with a `mu` network,
    # which IS a Sequential module. The final layer is the last element.
    elif hasattr(model.policy, "actor") and hasattr(model.policy.actor, "mu") and isinstance(model.policy.actor.mu, nn.Sequential):
        print("Biasing the final layer of SAC's actor (mu network).")
        final_layer = model.policy.actor.mu[-1]

    if final_layer is None or not isinstance(final_layer, nn.Linear):
        print("\nWARNING: Could not find a recognizable final nn.Linear policy layer to bias.")
        print("The agent will start with random actions. This may not be an error if using a custom policy.\n")
        return

    print(f"  - Original bias shape: {final_layer.bias.shape}")
    print(f"  - Setting final layer bias to {bias_value:.4f}")
    print(f"  - Setting final layer weights with std {weight_std:.4f}")

    # Initialize weights to be small to let the bias dominate the initial output
    torch.nn.init.normal_(final_layer.weight, mean=0.0, std=weight_std)
    
    # Initialize the bias to our target logit value
    torch.nn.init.constant_(final_layer.bias, bias_value)


@dataclass
class TrainingConfig:
    """Centralized configuration for training parameters."""
    max_cells: int = 57
    state_dim_per_cell: int = 25
    constraint_keys: List[str] = None
    device: torch.device = None
    
    def __post_init__(self):
        if self.constraint_keys is None:
            self.constraint_keys=['avg_drop_rate', 'avg_latency', 'cpu_violations', 'prb_violations']
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
                'net_arch': dict(pi=[256, 256], qf=[256, 256], vf=[256, 256])
            },
            'device': self.config.device,
            'tensorboard_log': "enhanced_logs/",
            'verbose': 1
        }
        
        algorithm_params = {
            'ppo': {
                'learning_rate': 3e-5,
                'n_steps': 4096,
                'batch_size': 256,
                'n_epochs': 20,
                'gamma': 0.995,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'target_kl': 0.02
            },
            'sac': {
                'learning_rate': 3e-4,
                'buffer_size': 1000000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'ent_coef': 'auto'
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
            "cost_lagrange": LagrangianCostWrapper
        }
        def _make_env() -> gym.Env:
            base_env = FiveGEnv(env_config, self.config.max_cells)
            wrapped_env = create_environment_from_name.get(name_env, lambda x: x)(base_env)
            monitored_env = Monitor(wrapped_env)
            return monitored_env
        return _make_env
    
    def train(self, algorithm: str, total_timesteps: int, n_envs: int = 4, name_env: str = "default") -> BaseAlgorithm:
        """Execute the complete training pipeline."""
        print(f"🚀 Starting enhanced training with {algorithm.upper()}")
        
        # Load training scenarios
        scenarios = self._load_scenarios()
        self.logger.info(f"Loaded {len(scenarios)} training scenarios.")

        # Create vectorized environment
        env_factories = [self.create_environment(scenarios[i % len(scenarios)], seed=i, name_env=name_env) 
                        for i in range(n_envs)]
        self.logger.info(f"Creating {n_envs} parallel environments.")
        self.logger.info(f"Using {name_env.upper()} environment wrapper.")
        subproc_vec_env = SubprocVecEnv(env_factories)
        vec_env = VecNormalize(subproc_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        self.logger.info(f"Normalized observation: {vec_env.norm_obs}")
        self.logger.info(f"Normalized reward: {vec_env.norm_reward}")
        
        # Create model
        model_class = {'ppo': PPO, 'sac': SAC, 'td3': TD3, 'ddpg': DDPG}[algorithm]
        model = model_class("MlpPolicy", vec_env, **self.hyperparameters[algorithm])
        self.model = model
        self.logger.info(f"Initialized {algorithm.upper()} model.")
        self.logger.info(f"Model architecture:\n {model.policy}")

        #------------Apply Bias Initialization------------#
        print("\n" + "="*50)
        print("Applying custom policy initialization...")

        # Bias to 1.1 power per action in the final layer
        bias_policy_output(model, bias_value=1.1)
        print("="*50 + "\n")


        #-----------------------------------------#
        # Log hyperparameters in a readable format
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

        # Setup callbacks
        callbacks = self._setup_callbacks()
        self.logger.info(f"Using the following callbacks: {json.dumps([type(cb).__name__ for cb in callbacks], indent=4)}")
        
        # Train model
        self.logger.info("Number of timesteps: {}".format(total_timesteps))
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
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
                        print(f"📁 Loaded scenario: {scenario.get('name', filename)}")
        except FileNotFoundError:
            print(f"⚠️  Scenario folder '{scenario_folder}' not found, using default")
            scenarios = [{'name': 'default', 'numUEs': 50, 'numSites': 3}]
        
        return scenarios
    
    def _setup_callbacks(self) -> List:
        """Setup training `callbacks`."""
        eval_env = DummyVecEnv([self.create_environment({}, seed=999)])
        vec_eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        callbacks = [
            CheckpointCallback(save_freq=50000, save_path='checkpoints/'),
            MetricsLoggerCallback(verbose=1),
            LoggedEvalCallback(
                eval_env=vec_eval_env,
                best_model_save_path='best_models/',
                log_path='eval_logs/',
                eval_freq=10000,
            ),
            ConstraintMonitorCallback(verbose=1),
        ]
        return callbacks
    
    def _save_model(self, model: BaseAlgorithm, algorithm: str):
        """Save trained model with metadata."""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs("trained_models/", exist_ok=True)
        model_path = f"trained_models/{algorithm}_final_{timestamp}.zip"
        model.save(model_path)
        print(f"💾 Model saved to: {model_path}")
    
    def setup_logging(self, log_file):
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
    parser.add_argument("--name_env", "-n", type=str, default="default", help="Environment type: 'default' or 'her'")
    
    args = parser.parse_args()
    
    # Initialize training configuration
    config = TrainingConfig()
    
    # Create and execute training pipeline
    pipeline = TrainingPipeline(config)

    # Display selected settings
    print("Algorithm selected:", args.algorithm.upper())
    pipeline.train(args.algorithm, args.timesteps, args.envs, args.name_env)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()