# train_enhanced.py
import json
import torch
import os
import pandas as pd
import numpy as np
import argparse
from typing import Callable, Dict, Any
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.base_class import BaseAlgorithm

from fiveg_env import FiveGEnv
from custom_models_sb3 import EnhancedAttentionNetwork

class CurriculumLearningCallback(BaseCallback):
    """
    Callback for automatic curriculum learning - advances training stages
    based on performance metrics.
    """
    
    def __init__(self, eval_freq: int = 10000, compliance_threshold: float = 0.85, 
                 min_steps_in_stage: int = 50000, verbose: int = 1):
        super(CurriculumLearningCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.compliance_threshold = compliance_threshold
        self.min_steps_in_stage = min_steps_in_stage
        self.last_eval_step = 0
        self.current_stage = "early"
        
    def _on_step(self) -> bool:
        # Check if it's time to evaluate for stage advancement
        if self.n_calls - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.n_calls
            
            # Get compliance rate from environments
            compliance_rates = []
            for env in self.training_env.envs:
                if hasattr(env, 'env') and hasattr(env.env, 'reward_computer'):
                    stats = env.env.reward_computer.get_stats()
                    compliance_rates.append(stats.get('compliance_rate', 0))
            
            if compliance_rates:
                avg_compliance = np.mean(compliance_rates) / 100.0  # Convert from percentage
                
                # Check if we should advance to next stage
                if (self.n_calls >= self.min_steps_in_stage and 
                    avg_compliance >= self.compliance_threshold):
                    
                    self._advance_training_stage()
        
        return True
    
    def _advance_training_stage(self):
        """Advance all environments to next training stage."""
        stages = ["early", "medium", "stable"]
        current_index = stages.index(self.current_stage)
        
        if current_index < len(stages) - 1:
            new_stage = stages[current_index + 1]
            self.current_stage = new_stage
            
            # Advance stage in all environments
            for env in self.training_env.envs:
                if hasattr(env, 'env') and hasattr(env.env, 'reward_computer'):
                    env.env.reward_computer.advance_training_stage()
            
            if self.verbose >= 1:
                print(f"\n🎓 CURRICULUM: Advanced to {new_stage} training stage at step {self.n_calls}")

def make_env_thunk(env_config: Dict[str, Any], max_cells: int, seed: int, training_stage: str = "stable") -> Callable:
    """Creates a thunk with training stage support."""
    def _thunk() -> Monitor:
        env_config_copy = env_config.copy()
        env_config_copy['seed'] = seed
        env_config_copy['training_stage'] = training_stage  # Add training stage
        env = FiveGEnv(env_config=env_config_copy, max_cells=max_cells)
        env = Monitor(env)
        return env
    return _thunk

def create_enhanced_model(algorithm: str, env: VecNormalize, device: torch.device, 
                         training_stage: str = "stable") -> BaseAlgorithm:
    """Factory function with stage-specific hyperparameters."""
    
    # Stage-specific hyperparameters
    stage_hyperparams = {
        "early": {
            "learning_rate": 5e-4,  # Higher learning rate for faster initial learning
            "gamma": 0.98,  # Shorter horizon for early training
            "ent_coef": 0.01,  # Higher exploration
        },
        "medium": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "ent_coef": 0.005,
        },
        "stable": {
            "learning_rate": 1e-4,  # Lower learning rate for fine-tuning
            "gamma": 0.995,  # Longer horizon
            "ent_coef": 0.001,  # Lower exploration
        }
    }
    
    if algorithm not in HYPERPARAMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
        
    config = HYPERPARAMS[algorithm]
    model_class = config['model_class']
    params = config['params'].copy()
    
    # Apply stage-specific adjustments
    stage_params = stage_hyperparams.get(training_stage, {})
    params.update(stage_params)
    
    # Apply linear schedule to learning rate
    params['learning_rate'] = linear_schedule(params['learning_rate'])
    
    return model_class(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=POLICY_KWARGS,
        device=device,
        tensorboard_log="enhanced_logs/",
        verbose=1,
        **params
    )

# Enhanced Training Protocol
def train_with_curriculum(args):
    """Main training function with curriculum learning."""
    print("--- Starting Enhanced 5G Training with Curriculum Learning ---")
    
    # Load scenarios
    scenario_folder = "scenarios/"
    scenario_files = [f for f in os.listdir(scenario_folder) if f.endswith('.json')]
    scenario_configs = []
    for sf in sorted(scenario_files):
        with open(os.path.join(scenario_folder, sf), 'r') as f:
            config = json.load(f)
            scenario_configs.append(config)
    
    # Set up curriculum stages
    curriculum_stages = [
        {"stage": "early", "timesteps": 200000, "compliance_threshold": 0.75},
        {"stage": "medium", "timesteps": 300000, "compliance_threshold": 0.85},
        {"stage": "stable", "timesteps": 500000, "compliance_threshold": 0.90}
    ]
    
    current_stage_index = 0
    total_trained_steps = 0
    
    for stage_config in curriculum_stages:
        stage_name = stage_config["stage"]
        stage_timesteps = stage_config["timesteps"]
        
        print(f"\n🎯 Starting {stage_name.upper()} training stage ({stage_timesteps:,} timesteps)")
        
        # Create environments for current stage
        env_thunks = [
            make_env_thunk(
                scenario_configs[i % len(scenario_configs)], 
                MAX_CELLS_SYSTEM_WIDE, 
                seed=i, 
                training_stage=stage_name
            ) for i in range(args.n_envs)
        ]
        
        if current_stage_index == 0:
            # First stage - create new model
            env = SubprocVecEnv(env_thunks)
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
            model = create_enhanced_model(args.algorithm, env, DEVICE, stage_name)
        else:
            # Continue with existing model but new environments
            env = SubprocVecEnv(env_thunks)
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
            # Use the same model but update environment
            model.set_env(env)
        
        # Set up callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=max(50000 // args.n_envs, 1), 
            save_path="enhanced_models/", 
            name_prefix=f"{args.algorithm}_{stage_name}"
        )
        
        curriculum_callback = CurriculumLearningCallback(
            eval_freq=10000,
            compliance_threshold=stage_config["compliance_threshold"],
            min_steps_in_stage=50000
        )
        
        # Train for this stage
        model.learn(
            total_timesteps=stage_timesteps,
            callback=[checkpoint_callback, curriculum_callback],
            reset_num_timesteps=False,
            tb_log_name=f"{args.algorithm}_curriculum"
        )
        
        total_trained_steps += stage_timesteps
        current_stage_index += 1
        
        # Save stage completion
        model.save(f"enhanced_models/{args.algorithm}_{stage_name}_completed.zip")
        env.save(f"enhanced_models/vec_normalize_{stage_name}.pkl")
        
        print(f"✅ Completed {stage_name} stage - Total steps: {total_trained_steps:,}")
    
    print(f"\n🎓 Curriculum training complete! Total: {total_trained_steps:,} timesteps")
    
    # Final evaluation
    final_evaluation(model, scenario_configs[0])
    
    return model

def final_evaluation(model, scenario_config):
    """Run comprehensive final evaluation."""
    print("\n--- Running Final Evaluation ---")
    
    eval_env = SubprocVecEnv([
        make_env_thunk(scenario_config, MAX_CELLS_SYSTEM_WIDE, seed=999, training_stage="stable")
    ])
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
    
    # Run multiple episodes for statistics
    n_episodes = 10
    episode_rewards = []
    compliance_rates = []
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        
        # Get compliance statistics
        if hasattr(eval_env.envs[0], 'env') and hasattr(eval_env.envs[0].env, 'reward_computer'):
            stats = eval_env.envs[0].env.reward_computer.get_stats()
            compliance_rates.append(stats.get('compliance_rate', 0))
    
    eval_env.close()
    
    print(f"\n📊 Final Evaluation Results ({n_episodes} episodes):")
    print(f"   Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   Average Compliance Rate: {np.mean(compliance_rates):.1f}%")
    print(f"   Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"   Worst Episode Reward: {np.min(episode_rewards):.2f}")

if __name__ == "__main__":
    # Reuse the same argument parsing from your original train.py
    from train import parse_args, HYPERPARAMS, POLICY_KWARGS, MAX_CELLS_SYSTEM_WIDE, DEVICE, linear_schedule
    
    args = parse_args()
    
    # Override to use enhanced training
    if hasattr(args, 'curriculum') and args.curriculum:
        train_with_curriculum(args)
    else:
        # Fall back to original training but with enhanced reward
        from train import main
        main(args)