from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional, List
import logging
import torch

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
    """Callback to update Lagrange multipliers."""
    
    def __init__(self, lambda_lr: float = 0.01, update_freq: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.lambda_lr = lambda_lr
        self.update_freq = update_freq
        self.lambda_history = defaultdict(list)
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Update lambda values at the end of each rollout."""
        # Get constraint violations from all environments
        if hasattr(self.training_env, 'get_attr'):
            # VecEnv case
            try:
                # Get lambdas and violations from the first environment
                env_lambdas = self.training_env.get_attr('lambdas')[0]
                env_violations = self.training_env.get_attr('constraint_violations')[0]
                
                # Update lambdas based on violations
                for key in env_lambdas.keys():
                    violation = env_violations.get(key, 0.0)
                    old_lambda = env_lambdas[key]
                    new_lambda = max(0.0, old_lambda + self.lambda_lr * violation)
                    
                    # Set updated lambda to all environments
                    for env_idx in range(self.training_env.num_envs):
                        self.training_env.env_method('set_lambda', key, new_lambda, indices=[env_idx])
                    
                    # Store history
                    self.lambda_history[key].append(new_lambda)
                
                # Log the updates
                if self.verbose > 0 and hasattr(self, 'logger') and self.logger is not None:
                    latest_values = {f"lambda/{key}": val[-1] for key, val in self.lambda_history.items()}
                    log_string = self._format_log_string(latest_values)
                    self.logger.info(f"Lambda Update: {log_string}")
                    
            except (AttributeError, IndexError) as e:
                if self.verbose > 0:
                    print(f"Warning: Could not update lambdas: {e}")
    
    def _format_log_string(self, values: dict[str, float]) -> str:
        """Format lambda values as a readable string."""
        if not isinstance(values, dict):
            return str(values)
        
        parts = []
        for key, value in values.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

class AdamLambdaUpdateCallback(BaseCallback):
    """
    Updates Lagrange multipliers (lambdas) using the Adam optimizer.
    This provides momentum and adaptive learning rates for more stable updates.

    :param constraint_keys: A list of keys for the constraints (e.g., ['drop_rate', 'latency']).
    :param initial_lambdas: A dictionary of initial lambda values.
    :param lambda_lr: The learning rate for the Adam optimizer.
    :param update_freq: How often to update the lambdas (in steps).
    """
    def __init__(self, 
                 constraint_keys: list[str],
                 initial_lambda_value: float = 1.0,
                 lambda_lr: float = 0.01, 
                 update_freq: int = 1000, 
                 verbose: int = 1):
        super().__init__(verbose)
        self.lambda_lr = lambda_lr
        self.update_freq = update_freq
        if constraint_keys is None:
            self.constraint_keys = ['drop_rate', 'latency', 'cpu_usage', 'prb_usage']
        else:
            self.constraint_keys = constraint_keys

        # --- 1. Set up the Lambdas as PyTorch Parameters ---
        # We treat the lambdas as learnable parameters so we can use a PyTorch optimizer.
        # We store the log of the lambdas for numerical stability and to ensure they remain non-negative.
        initial_log_lambdas = {key: torch.tensor(np.log(initial_lambda_value), dtype=torch.float32) 
                               for key in self.constraint_keys}
        self.log_lambdas = torch.nn.ParameterDict(initial_log_lambdas)

        # --- 2. Initialize the Adam Optimizer ---
        # The optimizer will manage the updates for our log_lambdas parameters.
        self.optimizer = torch.optim.Adam(self.log_lambdas.parameters(), lr=self.lambda_lr)

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            # --- 3. Collect and Average Costs ---
            all_costs = {key: [] for key in self.constraint_keys}
            for info in self.locals.get("infos", []):
                if 'lagrangian_costs' in info:
                    for key, cost in info['lagrangian_costs'].items():
                        all_costs[key].append(cost)
            
            mean_costs = {key: np.mean(values) if values else 0.0 
                          for key, values in all_costs.items()}
            
            # --- 4. Perform the Gradient Update with Adam ---
            self.optimizer.zero_grad(set_to_none=True)
            
            # The "loss" for our dual problem is `- (lambda * cost)`.
            # To perform gradient ascent (maximize this), we minimize the negative of it.
            # This is equivalent to `d_lambda = cost`.
            loss = 0
            for key in self.constraint_keys:
                # We work with log_lambdas, so we exponentiate to get the actual lambda value.
                # This ensures lambda is always >= 0.
                lambda_val = torch.exp(self.log_lambdas[key])
                loss -= lambda_val * mean_costs[key]
            
            # Backpropagate to compute gradients (d_loss / d_lambda)
            loss.backward()
            
            # Adam takes a step to update the log_lambdas
            self.optimizer.step()
            
            # --- 5. Update Lambdas in the Environment and Log ---
            with torch.no_grad():
                current_lambdas = {key: torch.exp(self.log_lambdas[key]).item() 
                                   for key in self.constraint_keys}
            
            self.training_env.env_method('update_lambdas', current_lambdas)
            
            if self.verbose > 0 and self.n_calls % (self.update_freq * 10) == 0:
                print(f"\n[AdamLambdaUpdate] Step {self.num_timesteps}:")
                for key in self.constraint_keys:
                    print(f"  - {key:<20s} | λ: {current_lambdas[key]:<8.4f} | Cost: {mean_costs[key]:.4f}")

            # Log to TensorBoard
            for key, value in current_lambdas.items():
                self.logger.record(f'lagrangian/lambda_{key}', value)
            for key, value in mean_costs.items():
                self.logger.record(f'lagrangian/cost_{key}', value)
        return True

class ClaudeAdamLambdaUpdateCallback(BaseCallback):
    """
    Enhanced Adam-based Lagrangian multiplier updater.
    
    Key improvements:
    - Supports constraint thresholds (e.g., drop_rate <= 1.0%)
    - Automatic lambda clipping to prevent explosion
    - Convergence detection
    - Gradient clipping for stability
    - Adaptive learning rate with warmup and decay
    - Comprehensive logging
    
    Math:
        For each constraint: cost = max(0, metric - threshold)
        Lambda update: λ ← λ + α * cost  (gradient ascent)
        With Adam: Uses momentum + adaptive LR for stable convergence
    """
    
    def __init__(self, 
                 constraint_keys: Optional[List[str]] = None,
                 constraint_thresholds: Optional[Dict[str, float]] = None,
                 initial_lambda_value: float = 1.0,
                 lambda_lr: float = 0.01, 
                 max_lambda: float = 100.0,
                 min_lambda: float = 0.0,
                 update_freq: int = 1000,
                 gradient_clip: float = 10.0,
                 use_warmup: bool = True,
                 warmup_steps: int = 10000,
                 lr_decay: bool = True,
                 decay_rate: float = 0.9995,
                 verbose: int = 1):
        """
        Args:
            constraint_keys: List of constraint names (e.g., ['drop_rate', 'latency'])
            constraint_thresholds: Target thresholds for each constraint
                                  E.g., {'drop_rate': 1.0, 'latency': 50.0}
                                  cost = max(0, metric - threshold)
            initial_lambda_value: Starting λ value for all constraints
            lambda_lr: Adam learning rate
            max_lambda: Maximum λ value (prevent explosion)
            min_lambda: Minimum λ value (usually 0)
            update_freq: Update λ every N steps
            gradient_clip: Clip gradients to [-clip, +clip] for stability
            use_warmup: Gradually increase LR from 0 to lambda_lr
            warmup_steps: Number of steps for warmup
            lr_decay: Decay learning rate over time
            decay_rate: Decay factor per update
            verbose: Logging verbosity (0=silent, 1=periodic, 2=every update)
        """
        super().__init__(verbose)
        
        # Constraint configuration
        if constraint_keys is None:
            self.constraint_keys = ['drop_rate', 'latency', 'cpu_usage', 'prb_usage']
        else:
            self.constraint_keys = constraint_keys
        
        # Default thresholds if not provided
        if constraint_thresholds is None:
            self.constraint_thresholds = {
                'drop_rate': 1.0,    # 1% max
                'latency': 50.0,     # 50ms max
                'cpu_usage': 80.0,   # 80% max
                'prb_usage': 80.0    # 80% max
            }
        else:
            self.constraint_thresholds = constraint_thresholds
        
        # Lambda configuration
        self.initial_lambda_value = initial_lambda_value
        self.base_lambda_lr = lambda_lr
        self.current_lambda_lr = lambda_lr
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda
        self.update_freq = update_freq
        self.gradient_clip = gradient_clip
        
        # Learning rate schedule
        self.use_warmup = use_warmup
        self.warmup_steps = warmup_steps
        self.lr_decay = lr_decay
        self.decay_rate = decay_rate
        
        # Initialize lambdas as PyTorch parameters (log space for stability)
        # log_lambda = log(lambda) ensures lambda > 0 after exp()
        initial_log_lambdas = {
            key: torch.tensor(
                np.log(max(initial_lambda_value, 1e-6)),  # Avoid log(0)
                dtype=torch.float32,
                requires_grad=True
            )
            for key in self.constraint_keys
        }
        self.log_lambdas = torch.nn.ParameterDict(initial_log_lambdas)
        
        # Initialize Adam optimizer
        self.optimizer = torch.optim.Adam(
            self.log_lambdas.parameters(), 
            lr=self.current_lambda_lr,
            betas=(0.9, 0.999),  # Standard Adam betas
            eps=1e-8
        )
        
        # Tracking for convergence detection
        self.cost_history = {key: deque(maxlen=100) for key in self.constraint_keys}
        self.lambda_history = {key: deque(maxlen=50) for key in self.constraint_keys}
        self.update_count = 0
        self.converged_constraints = set()
        
        # Statistics
        self.total_updates = 0
        self.gradient_norms = deque(maxlen=100)
        
        print(f"\n{'='*70}")
        print("AdamLambdaUpdateCallback Initialized")
        print(f"{'='*70}")
        print(f"  Constraint Keys: {self.constraint_keys}")
        print(f"  Thresholds: {self.constraint_thresholds}")
        print(f"  Initial λ: {initial_lambda_value:.4f}")
        print(f"  Learning Rate: {lambda_lr:.6f}")
        print(f"  Max λ: {max_lambda:.2f}")
        print(f"  Update Frequency: {update_freq} steps")
        print(f"  Gradient Clipping: {gradient_clip:.2f}")
        print(f"  Warmup: {'Enabled' if use_warmup else 'Disabled'}")
        print(f"  LR Decay: {'Enabled' if lr_decay else 'Disabled'}")
        print(f"{'='*70}\n")
    
    def _on_step(self) -> bool:
        """Called at every environment step."""
        
        # Only update every N steps
        if self.n_calls % self.update_freq != 0:
            return True
        
        self.update_count += 1
        
        # ====================================================================
        # STEP 1: Collect constraint costs from all parallel environments
        # ====================================================================
        all_costs = {key: [] for key in self.constraint_keys}
        
        for info in self.locals.get("infos", []):
            # Get metrics from info dict
            if 'lagrangian_costs' in info:
                # Pre-computed costs
                for key, cost in info['lagrangian_costs'].items():
                    if key in all_costs:
                        all_costs[key].append(cost)
            else:
                # Compute costs from metrics
                for key in self.constraint_keys:
                    if key in info:
                        metric_value = info[key]
                        threshold = self.constraint_thresholds.get(key, 0.0)
                        # Cost = violation amount (0 if satisfied)
                        cost = max(0.0, metric_value - threshold)
                        all_costs[key].append(cost)
        
        # Average costs across environments
        mean_costs = {
            key: np.mean(values) if values else 0.0 
            for key, values in all_costs.items()
        }
        
        # Store in history
        for key, cost in mean_costs.items():
            self.cost_history[key].append(cost)
        
        # ====================================================================
        # STEP 2: Update learning rate (warmup + decay)
        # ====================================================================
        self._update_learning_rate()
        
        # ====================================================================
        # STEP 3: Compute gradients and update lambdas with Adam
        # ====================================================================
        self.optimizer.zero_grad()
        
        # Dual problem: maximize sum(lambda * cost)
        # Equivalently: minimize -sum(lambda * cost)
        loss = torch.tensor(0.0, dtype=torch.float32)
        
        for key in self.constraint_keys:
            # Get actual lambda (exponentiate from log space)
            lambda_val = torch.exp(self.log_lambdas[key])
            
            # Cost (converted to tensor)
            cost = torch.tensor(mean_costs[key], dtype=torch.float32)
            
            # Accumulate loss: -lambda * cost
            # Gradient: d(-lambda * cost)/d(log_lambda) = -cost * lambda
            # This gives gradient ascent on lambda
            loss = loss - lambda_val * cost
        
        # Backpropagate
        loss.backward()
        
        # ====================================================================
        # STEP 4: Gradient clipping for stability
        # ====================================================================
        total_grad_norm = 0.0
        for param in self.log_lambdas.parameters():
            if param.grad is not None:
                # Clip individual gradients
                torch.nn.utils.clip_grad_value_(param, self.gradient_clip)
                total_grad_norm += param.grad.norm().item()
        
        self.gradient_norms.append(total_grad_norm)
        
        # ====================================================================
        # STEP 5: Adam optimizer step
        # ====================================================================
        self.optimizer.step()
        
        # ====================================================================
        # STEP 6: Clamp lambdas to valid range
        # ====================================================================
        with torch.no_grad():
            for key in self.constraint_keys:
                # Get actual lambda
                lambda_val = torch.exp(self.log_lambdas[key]).item()
                
                # Clamp to [min_lambda, max_lambda]
                lambda_val = max(self.min_lambda, min(lambda_val, self.max_lambda))
                
                # Update log_lambda
                self.log_lambdas[key].copy_(torch.log(torch.tensor(lambda_val + 1e-8)))
        
        # ====================================================================
        # STEP 7: Extract current lambdas and update environment
        # ====================================================================
        current_lambdas = self._get_current_lambdas()
        
        # Store in history
        for key, lambda_val in current_lambdas.items():
            self.lambda_history[key].append(lambda_val)
        
        # Update lambdas in wrapped environments
        try:
            self.training_env.env_method('update_lambdas', current_lambdas)
        except AttributeError:
            # If environment doesn't have update_lambdas method, skip
            pass
        
        # ====================================================================
        # STEP 8: Check convergence
        # ====================================================================
        self._check_convergence()
        
        # ====================================================================
        # STEP 9: Logging
        # ====================================================================
        self.total_updates += 1
        
        if self.verbose > 0:
            # Log every 10 updates for verbose=1, every update for verbose=2
            should_log = (self.verbose == 2) or (self.total_updates % 10 == 0)
            
            if should_log:
                self._log_update(current_lambdas, mean_costs, total_grad_norm)
        
        # TensorBoard logging
        self._log_to_tensorboard(current_lambdas, mean_costs, total_grad_norm)
        
        return True
    
    def _update_learning_rate(self):
        """Update learning rate with warmup and decay."""
        current_step = self.num_timesteps
        
        # Warmup phase
        if self.use_warmup and current_step < self.warmup_steps:
            warmup_progress = current_step / self.warmup_steps
            self.current_lambda_lr = self.base_lambda_lr * warmup_progress
        else:
            # Decay phase
            if self.lr_decay:
                decay_steps = max(0, current_step - self.warmup_steps) // self.update_freq
                self.current_lambda_lr = self.base_lambda_lr * (self.decay_rate ** decay_steps)
            else:
                self.current_lambda_lr = self.base_lambda_lr
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lambda_lr
    
    def _get_current_lambdas(self) -> Dict[str, float]:
        """Get current lambda values (in original space, not log space)."""
        with torch.no_grad():
            return {
                key: torch.exp(self.log_lambdas[key]).item()
                for key in self.constraint_keys
            }
    
    def _check_convergence(self):
        """Check if lambdas have converged for each constraint."""
        for key in self.constraint_keys:
            # Skip if already converged
            if key in self.converged_constraints:
                continue
            
            # Need enough history
            if len(self.lambda_history[key]) < 20:
                continue
            
            # Check lambda stability
            recent_lambdas = list(self.lambda_history[key])[-20:]
            lambda_std = np.std(recent_lambdas)
            lambda_mean = np.mean(recent_lambdas)
            
            # Check cost near zero
            if len(self.cost_history[key]) >= 10:
                recent_costs = list(self.cost_history[key])[-10:]
                avg_cost = np.mean(recent_costs)
            else:
                avg_cost = float('inf')
            
            # Convergence criteria:
            # 1. Lambda stable (low variance relative to mean)
            # 2. Cost near zero (constraint satisfied)
            relative_std = lambda_std / (lambda_mean + 1e-8)
            
            if relative_std < 0.05 and avg_cost < 0.01:
                self.converged_constraints.add(key)
                print(f"\n✅ Lambda for '{key}' has converged!")
                print(f"   Final λ: {lambda_mean:.4f}")
                print(f"   Avg cost: {avg_cost:.6f}")
    
    def _log_update(self, current_lambdas: Dict[str, float], 
                   mean_costs: Dict[str, float], grad_norm: float):
        """Print update information to console."""
        print(f"\n{'='*70}")
        print(f"[AdamLambdaUpdate] Step {self.num_timesteps} (Update #{self.total_updates})")
        print(f"{'='*70}")
        print(f"  Learning Rate: {self.current_lambda_lr:.6f}")
        print(f"  Gradient Norm: {grad_norm:.4f}")
        print(f"  Converged Constraints: {len(self.converged_constraints)}/{len(self.constraint_keys)}")
        print(f"\n  {'Constraint':<20} | {'Lambda':<10} | {'Cost':<10} | {'Status':<10}")
        print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        
        for key in self.constraint_keys:
            lambda_val = current_lambdas[key]
            cost = mean_costs[key]
            
            # Determine status
            if key in self.converged_constraints:
                status = "✅ Converged"
            elif cost < 0.01:
                status = "✓ Satisfied"
            elif cost < 0.1:
                status = "→ Near"
            else:
                status = "⚠ Violated"
            
            print(f"  {key:<20} | {lambda_val:<10.4f} | {cost:<10.4f} | {status}")
        
        # Additional diagnostics
        if self.gradient_norms:
            avg_grad = np.mean(list(self.gradient_norms)[-10:])
            print(f"\n  Recent Avg Gradient Norm: {avg_grad:.4f}")
        
        print(f"{'='*70}\n")
    
    def _log_to_tensorboard(self, current_lambdas: Dict[str, float],
                           mean_costs: Dict[str, float], grad_norm: float):
        """Log metrics to TensorBoard."""
        # Lambda values
        for key, value in current_lambdas.items():
            self.logger.record(f'lagrangian/lambda_{key}', value)
        
        # Costs
        for key, value in mean_costs.items():
            self.logger.record(f'lagrangian/cost_{key}', value)
        
        # Training metrics
        self.logger.record('lagrangian/learning_rate', self.current_lambda_lr)
        self.logger.record('lagrangian/gradient_norm', grad_norm)
        self.logger.record('lagrangian/converged_count', len(self.converged_constraints))
        self.logger.record('lagrangian/total_updates', self.total_updates)
        
        # Average lambda and cost
        avg_lambda = np.mean(list(current_lambdas.values()))
        avg_cost = np.mean(list(mean_costs.values()))
        self.logger.record('lagrangian/avg_lambda', avg_lambda)
        self.logger.record('lagrangian/avg_cost', avg_cost)
    
    def get_diagnostics(self) -> Dict[str, any]:
        """Get diagnostic information about the optimizer state."""
        current_lambdas = self._get_current_lambdas()
        
        return {
            'current_lambdas': current_lambdas,
            'converged_constraints': list(self.converged_constraints),
            'learning_rate': self.current_lambda_lr,
            'total_updates': self.total_updates,
            'gradient_norm_mean': np.mean(list(self.gradient_norms)) if self.gradient_norms else 0.0,
            'gradient_norm_std': np.std(list(self.gradient_norms)) if self.gradient_norms else 0.0,
        }
    
class MetricsLoggerCallback(BaseCallback):
    """
    A callback to log custom environment metrics at the end of each episode.
    This is the correct way to get detailed performance data into your logs.
    """
    def __init__(self, verbose: int = 0, logger = None):
        super().__init__(verbose)
        # Use deques to store metrics from recent episodes for averaging
        self.recent_kpi_violations = deque(maxlen=100)
        self.recent_compliance = deque(maxlen=100)

    def _on_step(self) -> bool:
        # Check for episode ends in any of the parallel environments
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                # Get the info dict for the environment that just finished
                info = self.locals["infos"][i]
                
                # --- Record custom metrics using self.logger.record() ---
                # This is the SB3-idiomatic way to log.
                self.logger.record('custom/kpi_violations', info.get('kpi_violations', 0))
                
                is_compliant = info.get('kpi_violations', 1) == 0
                self.logger.record('custom/is_compliant', float(is_compliant))
                
                # Store for multi-episode averaging
                self.recent_kpi_violations.append(info.get('kpi_violations', 0))
                self.recent_compliance.append(float(is_compliant))

        # At the end of a rollout, log the averaged metrics
        if self.n_calls % 5000 == 0:
            if self.recent_compliance:
                mean_compliance_rate = np.mean(self.recent_compliance) * 100
                mean_violations = np.mean(self.recent_kpi_violations)
                self.logger.record('custom/mean_compliance_rate', mean_compliance_rate)
                self.logger.record('custom/mean_violations', mean_violations)
                if self.verbose > 0:
                    self.logger.info(f"\n[MetricsLogger] Last {len(self.recent_compliance)} episodes: "
                          f"Compliance Rate={mean_compliance_rate:.1f}%, Avg Violations={mean_violations:.2f}")

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
                 best_model_save_path: str = 'best_models/', verbose: int = 1, logger = None):
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

class FileLoggingCallback(BaseCallback):
    """
    A custom callback that correctly intercepts Stable Baselines3's logger data
    at the end of each rollout and writes it to a structured log file.

    :param log_path: Path to the log file to be written.
    """
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        
        # --- 1. Set up a dedicated Python logger for file output ---
        self.file_logger = logging.getLogger("FileTrainingLogger")
        self.file_logger.setLevel(logging.INFO)
        
        # Prevent double-logging to the console
        self.file_logger.propagate = False
        
        # Ensure we're not adding handlers repeatedly if this object is recreated
        if not self.file_logger.handlers:
            # 'w' mode overwrites the file at the start of each new run
            file_handler = logging.FileHandler(self.log_path, mode='w')
            # The formatter only includes the raw message, creating a clean log file
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            self.file_logger.addHandler(file_handler)
        
        if self.verbose > 0:
            print(f"\n[FileLoggingCallback] Initialized. Training data will be logged to: {self.log_path}")

    def _on_rollout_end(self) -> bool:
        """
        This method is called by SB3 at the end of each data collection rollout.
        It's the perfect moment to capture the aggregated training metrics.
        """
        # --- 2. CRITICAL FIX: Get data from the correct source ---
        # `self.logger` is the official SB3 logger object. `get_latest_values()` returns
        # the dictionary of all metrics recorded since the last log.
        latest_values = self.logger.get_latest_values()

        if not latest_values:
            return True # Do nothing if no new values were logged

        # --- 3. Format the data into a human-readable string ---
        log_string = self._format_log_string(latest_values)
        
        # --- 4. Write the formatted string to the file via our logger ---
        self.file_logger.info(log_string)
        
        return True
    
    def _format_log_string(self, values: dict) -> str:
        """Formats the dictionary of values into the desired table-like string."""
        log_groups = defaultdict(dict)
        for key, value in values.items():
            if '/' in key:
                group, metric = key.split('/', 1)
                log_groups[group][metric] = value
        
        output = f"-------------------[ Timestep {self.num_timesteps} ]-------------------\n"
        
        # Define the order for consistent, readable logs
        group_order = ['lagrangian', 'rollout', 'time', 'train']
        
        for group in group_order:
            if group in log_groups:
                output += f"| {group}/\n"
                for metric, value in sorted(log_groups[group].items()):
                    # Format numbers for clean alignment
                    if isinstance(value, (float, np.floating)):
                        output += f"|    {metric:<20s} | {value:<10.3g}\n"
                    else:
                        output += f"|    {metric:<20s} | {value:<10}\n"

        output += "--------------------------------------------------------\n"
        return output

    def _on_training_end(self) -> None:
        """Close the file handler cleanly when training ends."""
        for handler in self.file_logger.handlers:
            handler.close()
            self.file_logger.removeHandler(handler)
