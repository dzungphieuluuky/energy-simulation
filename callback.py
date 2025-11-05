from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
class ConstraintMonitorCallback(BaseCallback):
    """Enhanced monitoring with violation tracking and adaptive penalties."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.violation_history = []
        self.compliance_history = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Check if episode just finished
        if self.locals.get('dones', [False])[0]:
            # Get final metrics from info
            info = self.locals.get('infos', [{}])[0]
            
            # Track violations
            kpi_violations = info.get('kpi_violations', 0)
            self.violation_history.append(kpi_violations)
            
            # Track compliance
            drop_ok = info.get('avg_drop_rate', 999) <= self.training_env.get_attr('sim_params')[0].dropCallThreshold
            latency_ok = info.get('avg_latency', 999) <= self.training_env.get_attr('sim_params')[0].latencyThreshold
            cpu_ok = info.get('cpu_violations', 999) == 0
            prb_ok = info.get('prb_violations', 999) == 0
            
            all_ok = drop_ok and latency_ok and cpu_ok and prb_ok
            self.compliance_history.append(all_ok)
            
            # Log every 10 episodes
            if len(self.violation_history) % 10 == 0:
                recent_compliance = np.mean(self.compliance_history[-10:]) * 100
                recent_violations = np.mean(self.violation_history[-10:])
                
                print(f"\n[Step {self.num_timesteps}] Last 10 Episodes:")
                print(f"  Compliance Rate: {recent_compliance:.1f}%")
                print(f"  Avg Violations: {recent_violations:.2f}")
                
                if recent_compliance < 50:
                    print("  ⚠️  WARNING: Low compliance rate! Agent struggling with constraints.")
                elif recent_compliance > 80:
                    print("  ✅ Good compliance! Agent learning constraint satisfaction.")
        
        return True

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
    
class LambdaUpdateCallback(BaseCallback):
    """
    Performs gradient descent to update the Lagrange multipliers (lambdas).
    
    The update rule for each lambda is: λ_new = max(0, λ_old + learning_rate * cost).
    This is a gradient ascent step on the cost, which minimizes the Lagrangian objective.
    """
    def __init__(self, lambda_lr: float = 0.01, update_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.lambda_lr = lambda_lr
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        # Update lambdas only at a specified frequency to ensure stability
        if self.n_calls % self.update_freq == 0:
            # --- 1. Collect Costs from all parallel environments ---
            all_costs = {key: [] for key in self.training_env.get_attr('constraint_keys')[0]}
            for info in self.locals.get("infos", []):
                if 'lagrangian_costs' in info:
                    for key, cost in info['lagrangian_costs'].items():
                        all_costs[key].append(cost)
            
            # --- 2. Average Costs Across the collected batch ---
            mean_costs = {key: np.mean(values) for key, values in all_costs.items()}
            
            # --- 3. Apply the Gradient Update Rule ---
            current_lambdas = self.training_env.get_attr('lambdas')[0]
            new_lambdas = {}
            for key, cost in mean_costs.items():
                # The core of the dual update: λ_t+1 = λ_t + α * C(s_t)
                new_lambda = current_lambdas[key] + self.lambda_lr * cost
                # Lambdas must always be non-negative
                new_lambdas[key] = max(0, new_lambda)
            
            # --- 4. Update Lambdas in All Parallel Environments ---
            self.training_env.env_method('update_lambdas', new_lambdas)
            
            # --- 5. Log everything to TensorBoard for monitoring ---
            for key, value in new_lambdas.items():
                self.logger.record(f'lagrangian/lambda_{key}', value)
            for key, value in mean_costs.items():
                self.logger.record(f'lagrangian/cost_{key}', value)
        return True

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

class LoggedEvalCallback(EvalCallback):
    """EvalCallback with enhanced logging for evaluation results."""
    
    def __init__(self, eval_env, log_path: str = 'eval_logs/', eval_freq: int = 10000, 
                 best_model_save_path: str = 'best_models/', verbose: int = 1):
        super(LoggedEvalCallback, self).__init__(
            eval_env=eval_env,
            log_path=log_path,
            eval_freq=eval_freq,
            best_model_save_path=best_model_save_path,
            verbose=verbose
        )
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Additional logging
        if self.n_calls % self.eval_freq == 0:
            mean_reward = np.mean(self.last_mean_reward)
            std_reward = np.std(self.last_mean_reward)
            self.logger.info(f"[Evaluation at step {self.num_timesteps}] Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return result