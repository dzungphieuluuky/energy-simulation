import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
from fiveg_env import FiveGEnv

import numpy as np

class StateNormalizer:
    """Handles state normalization with running statistics"""
    
    def __init__(self, state_dim, epsilon=1e-8, n_cells=10):
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.n_cells = n_cells

        # Simulation features normalization bounds (first 17 features)
        self.simulation_bounds = {
            'totalCells': [1, 50],               # number of cells
            'totalUEs': [1, 500],                # number of UEs
            'simTime': [600, 3600],              # simulation time
            'timeStep': [1, 10],                 # time step
            'timeProgress': [0, 1],              # progress ratio
            'carrierFrequency': [700e6, 6e9],    # frequency Hz
            'isd': [100, 2000],                  # inter-site distance
            'minTxPower': [0, 46],               # dBm
            'maxTxPower': [0, 46],              # dBm
            'basePower': [100, 100000],            # watts
            'idlePower': [50, 50000],              # watts
            'dropCallThreshold': [1, 10],        # percentage
            'latencyThreshold': [10, 100],       # ms
            'cpuThreshold': [70, 95],            # percentage
            'prbThreshold': [70, 95],            # percentage
            'trafficLambda': [0.1, 10],          # traffic rate
            'peakHourMultiplier': [1, 5]         # multiplier
        }
        
        # Network features normalization bounds (next 14 features)
        self.network_bounds = {
            'totalEnergy': [0, 10000],           # kWh
            'activeCells': [0, 50],              # number of cells
            'avgDropRate': [0, 20],              # percentage
            'avgLatency': [0, 200],              # ms
            'totalTraffic': [0, 5000],           # traffic units
            'connectedUEs': [0, 500],            # number of UEs
            'connectionRate': [0, 100],         # percentage
            'cpuViolations': [0, 10000],            # number of violations
            'prbViolations': [0, 10000],            # number of violations
            'maxCpuUsage': [0, 100],             # percentage
            'maxPrbUsage': [0, 100],             # percentage
            'kpiViolations': [0, 10000],          # number of violations
            'totalTxPower': [0, 1000],           # total power
            'avgPowerRatio': [0, 1]              # ratio
        }
        
        # Cell features normalization bounds (12 features per cell)
        self.cell_bounds = {
            'cpuUsage': [0, 100],                # percentage
            'prbUsage': [0, 100],                # percentage
            'currentLoad': [0, 1000],            # load units
            'maxCapacity': [0, 1000],            # capacity units
            'numConnectedUEs': [0, 50],          # number of UEs
            'txPower': [0, 46],                  # dBm
            'energyConsumption': [0, 5000],      # watts
            'avgRSRP': [-140, -70],              # dBm
            'avgRSRQ': [-20, 0],                 # dB
            'avgSINR': [-10, 30],                # dB
            'totalTrafficDemand': [0, 500],      # traffic units
            'loadRatio': [0, 1]                  # ratio
        }
    
    def normalize(self, state_vector):
        """
        Normalize state vector to [0, 1] range
        
        State structure:
        [sim_1, ..., sim_17,              # Index 0-16 (17 features)
         net_1, ..., net_14,              # Index 17-30 (14 features)
         c1_f1, c2_f1, ..., cn_f1,       # cpuUsage for all cells
         c1_f2, c2_f2, ..., cn_f2,       # prbUsage for all cells
         ...                              # etc for all 12 cell features
         c1_f12, c2_f12, ..., cn_f12]    # loadRatio for all cells
        """
        normalized = np.zeros_like(state_vector)
        
        # Normalize simulation features (indices 0-16)
        simulation_keys = list(self.simulation_bounds.keys())
        for i, key in enumerate(simulation_keys):
            if i < len(state_vector):
                min_val, max_val = self.simulation_bounds[key]
                normalized[i] = self._normalize_value(state_vector[i], min_val, max_val)
        
        # Normalize network features (indices 17-30)
        network_keys = list(self.network_bounds.keys())
        for i, key in enumerate(network_keys):
            global_idx = 17 + i
            if global_idx < len(state_vector):
                min_val, max_val = self.network_bounds[key]
                normalized[global_idx] = self._normalize_value(state_vector[global_idx], min_val, max_val)
        
        # Normalize cell features (indices 31 onwards)
        cell_keys = list(self.cell_bounds.keys())
        start_idx = 31  # After simulation (17) and network (14) features
        
        for feat_idx, key in enumerate(cell_keys):
            min_val, max_val = self.cell_bounds[key]
            
            # Normalize all cells for this feature
            for cell_idx in range(self.n_cells):
                global_idx = start_idx + feat_idx * self.n_cells + cell_idx
                if global_idx < len(state_vector):
                    normalized[global_idx] = self._normalize_value(
                        state_vector[global_idx], min_val, max_val)
        
        return normalized
    
    def _normalize_value(self, value, min_val, max_val):
        """Normalize single value to [0, 1] range"""
        if max_val == min_val:
            return 0.5  # Default middle value
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    def update_stats(self, state_vector):
        pass

class StateNormalizerWrapper(gym.ObservationWrapper):
    """
    Applies the competition's mandatory StateNormalizer to each observation.
    This ensures the agent is trained on the exact state representation
    it will encounter during inference.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Access max_cells from the unwrapped base environment for robustness
        n_cells = self.env.unwrapped.max_cells 
        self.normalizer = StateNormalizer(state_dim=self.observation_space.shape[0], n_cells=n_cells)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return self.normalizer.normalize(obs)

class LagrangianRewardWrapper(gym.Wrapper):
    """
    Wraps the 5G environment to implement a Lagrangian reward structure.

    This wrapper separates the primary objective (reward) from the constraints (costs).
    - The "primal reward" is what the agent tries to maximize (e.g., energy efficiency).
    - The "cost" is a non-negative value for each constraint violation.
    - The final reward given to the agent is: `Reward = PrimalReward - dot(Lambdas, Costs)`.
    """
    def __init__(self, env: FiveGEnv, constraint_keys: List[str] = None):
        super().__init__(env)
        self.env = env
        if constraint_keys is None:
            self.constraint_keys = ['drop_rate', 'latency', 'cpu_violations', 'prb_violations']

        # Initialize Lagrange multipliers (lambdas) for each constraint
        self.lambdas = {key: 1.0 for key in self.constraint_keys}
        
    def _compute_primal_reward(self, metrics: Dict[str, Any]) -> float:
        """The agent's primary objective: maximize energy efficiency."""
        p = self.env.sim_params
        total_energy = metrics.get('total_energy', 0)
        max_power_consumption = 10**((p.maxTxPower - 30) / 10)
        max_possible_energy = self.env.n_cells * (p.idlePower + max_power_consumption)
        energy_efficiency = 1.0 - (total_energy / max(1, max_possible_energy))
        return max(0.0, energy_efficiency)

    def _compute_costs(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the non-negative cost for each constraint violation."""
        p = self.env.sim_params
        costs = {key: 0.0 for key in self.constraint_keys}

        # --- Drop Rate Constraint ---
        threshold = p.dropCallThreshold
        value = metrics.get('avg_drop_rate', 0)
        if value > threshold:
            costs['drop_rate'] = (value - threshold) / threshold # Normalized severity

        # --- Latency Constraint ---
        threshold = p.latencyThreshold
        value = metrics.get('avg_latency', 0)
        if value > threshold:
            costs['latency'] = (value - threshold) / threshold

        # --- CPU Violations Constraint ---
        value = metrics.get('cpu_violations', 0)
        if value > 0:
            costs['cpu_violations'] = value / self.env.n_cells # Normalize by number of cells

        # --- PRB Violations Constraint ---
        value = metrics.get('prb_violations', 0)
        if value > 0:
            costs['prb_violations'] = value / self.env.n_cells
            
        return costs

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        primal_reward = self._compute_primal_reward(info)
        costs = self._compute_costs(info)
        
        # Calculate the Lagrangian penalty by summing the product of each lambda and its cost
        lagrangian_penalty = sum(self.lambdas[key] * costs[key] for key in self.constraint_keys)
        
        # The final reward the agent sees is the trade-off determined by the lambdas
        reward = primal_reward - lagrangian_penalty
        
        # Store costs in the info dict for the callback to use
        info['lagrangian_costs'] = costs
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # *** CRITICAL FIX ***
        # Must pass **kwargs to the underlying environment to handle seeds and options.
        return self.env.reset(**kwargs)
    
    def update_lambdas(self, new_lambdas: Dict[str, float]):
        """Allows the callback to update the lambda values from outside the wrapper."""
        for key, value in new_lambdas.items():
            if key in self.lambdas:
                self.lambdas[key] = value

class LagrangianCostWrapper(gym.Wrapper):
    """
    Wrapper that computes Lagrangian costs and adds them to info dict.
    This allows AdamLambdaUpdateCallback to access costs without needing
    to know internal environment structure.
    """
    
    def __init__(self, env, constraint_thresholds: Dict[str, float]):
        """
        Args:
            env: Base environment
            constraint_thresholds: Thresholds for each constraint
        """
        super().__init__(env)
        self.constraint_thresholds = constraint_thresholds
        self.current_lambdas = {key: 1.0 for key in constraint_thresholds.keys()}
    
    def step(self, action):
        """Execute step and compute Lagrangian costs."""
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Add a primal reward for the main objective (energy saving)
        primal_reward = self._compute_energy_reward(info)
        
        # Compute violation costs
        lagrangian_costs = self._compute_costs(info)
        
        # Add to info for the callback
        info['lagrangian_costs'] = lagrangian_costs
        
        lagrangian_penalty = sum(
            self.current_lambdas.get(key, 0.0) * cost
            for key, cost in lagrangian_costs.items()
        )
        
        info['lagrangian_penalty'] = lagrangian_penalty
        info['primal_reward'] = primal_reward
        
        reward = primal_reward - lagrangian_penalty
        
        return obs, reward, terminated, truncated, info

    def _compute_energy_reward(self, metrics: Dict[str, Any]) -> float:
        """Computes the primary objective: energy efficiency reward."""
        p = self.env.unwrapped.sim_params
        total_energy = metrics.get('total_energy', 0)
        max_tx_power_watts = 10**((p.maxTxPower - 30) / 10)
        max_possible_energy = self.env.unwrapped.n_cells * (p.basePower + max_tx_power_watts)
        
        if max_possible_energy > 0:
            energy_saved_ratio = 1.0 - (total_energy / max_possible_energy)
            return max(0.0, energy_saved_ratio)
        return 0.0

    def _compute_costs(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the non-negative cost for each constraint violation."""
        costs = {}
        for key, threshold in self.constraint_thresholds.items():
            if key in metrics:
                metric_value = metrics[key]
                cost = max(0.0, metric_value - threshold)
                costs[key] = cost
        return costs
    
    def update_lambdas(self, new_lambdas: Dict[str, float]):
        """Update lambda values (called by callback)."""
        self.current_lambdas.update(new_lambdas)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        return self.env.reset(seed=seed, options=options)

class StrictConstraintWrapper(gym.Wrapper):
    """
    Zero reward if ANY constraint is violated.
    Energy efficiency reward ONLY when all constraints satisfied.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.consecutive_compliant_steps = 0
        
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        metrics = self.env.compute_metrics()
        
        # Check ALL constraints
        violations = self._check_all_constraints(metrics)
        has_violations = any(violations.values())
        
        if has_violations:
            # ZERO reward during violations
            reward = 0.0
            self.consecutive_compliant_steps = 0
        else:
            # Energy reward ONLY when compliant
            self.consecutive_compliant_steps += 1
            reward = self._compute_energy_reward(metrics)
            
            # Bonus for sustained compliance
            if self.consecutive_compliant_steps > 20:
                reward += np.log1p(self.consecutive_compliant_steps - 20) * 0.2
        
        # Enhanced logging
        info['strict_reward'] = {
            'has_violations': has_violations,
            'energy_reward': reward if not has_violations else 0.0,
            'consecutive_compliant_steps': self.consecutive_compliant_steps,
            'violations': violations
        }
        
        return obs, reward, terminated, truncated, info
    
    def _check_all_constraints(self, metrics):
        """Check all QoS and resource constraints."""
        p = self.env.sim_params
        
        violations = {
            'drop_rate': metrics['avg_drop_rate'] > p.dropCallThreshold,
            'latency': metrics['avg_latency'] > p.latencyThreshold,
            'cpu': metrics['cpu_violations'] > 0,
            'prb': metrics['prb_violations'] > 0,
            'connectivity': metrics.get('connection_rate', 1.0) < 0.95
        }
        
        return violations
    
    def _compute_energy_reward(self, metrics):
        """Compute energy efficiency reward (0-1 scale)."""
        total_energy = metrics['total_energy']
        
        # Calculate maximum possible energy
        p = self.env.sim_params
        max_power = 10**((p.max_tx_power - 30)/10)
        max_energy = self.env.n_cells * (p.base_power + max_power)
        
        if max_energy > 0:
            efficiency = 1.0 - (total_energy / max_energy)
            return max(0.0, efficiency)
        return 0.0

"""
Enhanced Hindsight Experience Replay (HER) for PPO
Designed specifically for zero-reward-on-violation training strategy

Key improvements:
1. Multi-scale progress tracking (short/medium/long term)
2. Constraint-specific curriculum learning
3. Milestone-based reward shaping
4. Adaptive reward scaling based on training phase
5. Better handling of partial constraint satisfaction
"""
class SimplifiedHERForPPO(gym.Wrapper):
    """
    Enhanced HER wrapper for PPO with zero-reward-on-violation strategy.
    
    Philosophy:
    - Original reward: 0 for any violation, positive only when all constraints met
    - HER reward: Provide intermediate feedback for PROGRESS toward constraints
    - This helps agent learn constraint boundaries faster
    
    Key features:
    - Multi-timescale progress tracking (short/medium/long)
    - Adaptive reward scaling based on training stage
    - Milestone bonuses for achieving constraint satisfaction
    - Curriculum learning: focus on hardest constraints first
    """
    
    def __init__(self, env, enable_curriculum: bool = True, 
                 enable_milestones: bool = True,
                 progress_weight: float = 0.3,
                 milestone_weight: float = 0.5):
        """
        Args:
            env: Base environment
            enable_curriculum: Enable adaptive curriculum learning
            enable_milestones: Enable milestone bonuses
            progress_weight: Weight for progress rewards (0-1)
            milestone_weight: Weight for milestone rewards (0-1)
        """
        super().__init__(env)
        
        # Progress tracking at multiple timescales
        self.constraint_progress = {
            'drop_rate': {
                'short': deque(maxlen=10),   # Last 10 steps
                'medium': deque(maxlen=50),  # Last 50 steps
                'long': deque(maxlen=200)    # Last 200 steps
            },
            'latency': {
                'short': deque(maxlen=10),
                'medium': deque(maxlen=50),
                'long': deque(maxlen=200)
            },
            'resources': {
                'short': deque(maxlen=10),
                'medium': deque(maxlen=50),
                'long': deque(maxlen=200)
            }
        }
        
        # Training statistics
        self.total_steps = 0
        self.episode_steps = 0
        self.total_episodes = 0
        self.consecutive_satisfactions = {
            'drop_rate': 0,
            'latency': 0,
            'resources': 0,
            'all': 0
        }
        
        # Constraint difficulty tracking (for curriculum)
        self.constraint_difficulty = {
            'drop_rate': 1.0,
            'latency': 1.0,
            'resources': 1.0
        }
        
        # Milestone tracking
        self.milestones_reached = {
            'first_drop_rate_satisfy': False,
            'first_latency_satisfy': False,
            'first_resource_satisfy': False,
            'first_all_satisfy': False,
            'sustained_10_steps': False,
            'sustained_50_steps': False,
            'sustained_100_steps': False
        }
        
        # Configuration
        self.enable_curriculum = enable_curriculum
        self.enable_milestones = enable_milestones
        self.progress_weight = progress_weight
        self.milestone_weight = milestone_weight
        
        # Adaptive scaling based on training stage
        self.training_stage = 'early'  # early, middle, late
        self.stage_thresholds = {
            'early_to_middle': 50000,  # Switch after 50k steps
            'middle_to_late': 200000   # Switch after 200k steps
        }
        
        print(f"\n{'='*70}")
        print("SimplifiedHERForPPO Initialized")
        print(f"{'='*70}")
        print(f"  Curriculum Learning: {'Enabled' if enable_curriculum else 'Disabled'}")
        print(f"  Milestone Rewards: {'Enabled' if enable_milestones else 'Disabled'}")
        print(f"  Progress Weight: {progress_weight}")
        print(f"  Milestone Weight: {milestone_weight}")
        print(f"{'='*70}\n")
    
    def reset(self, **kwargs):
        """Reset environment and tracking."""
        obs, info = self.env.reset(**kwargs)
        
        self.episode_steps = 0
        self.total_episodes += 1
        
        # Update training stage
        self._update_training_stage()
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute step with HER-enhanced rewards.
        
        Reward structure:
        1. Original reward (0 if violated, positive if satisfied + energy efficient)
        2. Progress reward (for moving toward constraint satisfaction)
        3. Milestone reward (for achieving new goals)
        4. Curriculum bonus (for mastering difficult constraints)
        """
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        self.total_steps += 1
        self.episode_steps += 1
        
        # Get current metrics
        metrics = self.env.compute_metrics()
        
        # ================================================================
        # STEP 1: Compute constraint satisfaction progress
        # ================================================================
        drop_progress = self._compute_drop_rate_progress(metrics)
        latency_progress = self._compute_latency_progress(metrics)
        resource_progress = self._compute_resource_progress(metrics)
        
        # Store in all timescales
        for timescale in ['short', 'medium', 'long']:
            self.constraint_progress['drop_rate'][timescale].append(drop_progress)
            self.constraint_progress['latency'][timescale].append(latency_progress)
            self.constraint_progress['resources'][timescale].append(resource_progress)
        
        # Update consecutive satisfaction counters
        self._update_satisfaction_counters(metrics)
        
        # ================================================================
        # STEP 2: Compute HER progress rewards
        # ================================================================
        progress_reward = self._compute_progress_reward(
            drop_progress, latency_progress, resource_progress
        )
        
        # ================================================================
        # STEP 3: Compute milestone rewards
        # ================================================================
        milestone_reward = 0.0
        if self.enable_milestones:
            milestone_reward = self._compute_milestone_reward(metrics)
        
        # ================================================================
        # STEP 4: Compute curriculum bonus
        # ================================================================
        curriculum_bonus = 0.0
        if self.enable_curriculum:
            curriculum_bonus = self._compute_curriculum_bonus(
                drop_progress, latency_progress, resource_progress
            )
        
        # ================================================================
        # STEP 5: Combine rewards with adaptive weighting
        # ================================================================
        # Scale HER components based on training stage
        stage_multipliers = {
            'early': 1.0,    # Full HER assistance early
            'middle': 0.7,   # Reduce HER as agent improves
            'late': 0.4      # Minimal HER when converged
        }
        
        her_multiplier = stage_multipliers[self.training_stage]
        
        # Combine all reward components
        her_reward = (
            progress_reward * self.progress_weight * her_multiplier +
            milestone_reward * self.milestone_weight +
            curriculum_bonus * 0.2 * her_multiplier
        )
        
        total_reward = original_reward + her_reward
        
        # ================================================================
        # STEP 6: Update constraint difficulty (curriculum learning)
        # ================================================================
        if self.enable_curriculum and self.episode_steps % 10 == 0:
            self._update_constraint_difficulty()
        
        # ================================================================
        # STEP 7: Add HER info to info dict
        # ================================================================
        info['her_metrics'] = {
            'progress_reward': float(progress_reward),
            'milestone_reward': float(milestone_reward),
            'curriculum_bonus': float(curriculum_bonus),
            'total_her_reward': float(her_reward),
            'training_stage': self.training_stage,
            'constraint_difficulty': self.constraint_difficulty.copy(),
            'drop_progress': float(drop_progress),
            'latency_progress': float(latency_progress),
            'resource_progress': float(resource_progress),
        }
        
        # Log milestone achievements
        if terminated or truncated:
            self._log_episode_summary(info)
        
        return obs, total_reward, terminated, truncated, info
    
    def _compute_drop_rate_progress(self, metrics: Dict) -> float:
        """
        Compute progress toward satisfying drop rate constraint.
        Returns value between 0 (far from goal) and 1 (goal achieved).
        """
        p = self.env.sim_params
        
        if metrics['avg_drop_rate'] <= p.dropCallThreshold:
            return 1.0  # Perfect satisfaction
        
        # Compute normalized distance from threshold
        # Use exponential decay for smoother progress signal
        violation_ratio = (metrics['avg_drop_rate'] - p.dropCallThreshold) / p.dropCallThreshold
        
        # Exponential decay: progress = exp(-k * violation)
        # k=2 gives good sensitivity
        progress = np.exp(-2.0 * violation_ratio)
        
        return float(max(0.0, min(1.0, progress)))
    
    def _compute_latency_progress(self, metrics: Dict) -> float:
        """Compute progress toward satisfying latency constraint."""
        p = self.env.sim_params
        
        if metrics['avg_latency'] <= p.latencyThreshold:
            return 1.0
        
        violation_ratio = (metrics['avg_latency'] - p.latencyThreshold) / p.latencyThreshold
        progress = np.exp(-2.0 * violation_ratio)
        
        return float(max(0.0, min(1.0, progress)))
    
    def _compute_resource_progress(self, metrics: Dict) -> float:
        """
        Compute progress toward satisfying resource constraints.
        Considers both CPU and PRB violations.
        """
        total_violations = metrics['cpu_violations'] + metrics['prb_violations']
        
        if total_violations == 0:
            return 1.0
        
        # Normalize by maximum possible violations
        max_possible = 2 * self.env.n_cells
        violation_ratio = total_violations / max_possible
        
        # Exponential decay
        progress = np.exp(-3.0 * violation_ratio)  # k=3 for sharper response
        
        return float(max(0.0, min(1.0, progress)))
    
    def _compute_progress_reward(self, drop_progress: float, 
                                 latency_progress: float, 
                                 resource_progress: float) -> float:
        """
        Compute reward for progress across all constraints.
        
        Key insight: Reward IMPROVEMENT, not just absolute progress.
        This encourages agent to keep improving.
        """
        progress_reward = 0.0
        
        # Check each constraint for improvement
        improvements = []
        
        # Drop rate improvement
        if len(self.constraint_progress['drop_rate']['medium']) > 1:
            prev_avg = np.mean(list(self.constraint_progress['drop_rate']['medium'])[:-1])
            if drop_progress > prev_avg + 0.01:  # Significant improvement
                improvement = drop_progress - prev_avg
                improvements.append(('drop_rate', improvement))
                progress_reward += improvement * 0.5
        
        # Latency improvement
        if len(self.constraint_progress['latency']['medium']) > 1:
            prev_avg = np.mean(list(self.constraint_progress['latency']['medium'])[:-1])
            if latency_progress > prev_avg + 0.01:
                improvement = latency_progress - prev_avg
                improvements.append(('latency', improvement))
                progress_reward += improvement * 0.5
        
        # Resource improvement
        if len(self.constraint_progress['resources']['medium']) > 1:
            prev_avg = np.mean(list(self.constraint_progress['resources']['medium'])[:-1])
            if resource_progress > prev_avg + 0.01:
                improvement = resource_progress - prev_avg
                improvements.append(('resources', improvement))
                progress_reward += improvement * 0.5
        
        # Bonus for satisfying individual constraints
        satisfaction_bonus = 0.0
        if drop_progress >= 0.99:
            satisfaction_bonus += 0.3
        if latency_progress >= 0.99:
            satisfaction_bonus += 0.3
        if resource_progress >= 0.99:
            satisfaction_bonus += 0.3
        
        # Bonus for satisfying ALL constraints simultaneously
        if drop_progress >= 0.99 and latency_progress >= 0.99 and resource_progress >= 0.99:
            satisfaction_bonus += 0.5  # Extra bonus for full satisfaction
        
        progress_reward += satisfaction_bonus
        
        return progress_reward
    
    def _compute_milestone_reward(self, metrics: Dict) -> float:
        """
        Compute one-time milestone rewards for achieving goals.
        This helps mark important learning moments.
        """
        milestone_reward = 0.0
        p = self.env.sim_params
        
        # First-time achievements (one-time bonuses)
        if not self.milestones_reached['first_drop_rate_satisfy']:
            if metrics['avg_drop_rate'] <= p.dropCallThreshold:
                milestone_reward += 2.0
                self.milestones_reached['first_drop_rate_satisfy'] = True
                print(f"\n🎯 MILESTONE: First drop rate satisfaction at step {self.total_steps}!")
        
        if not self.milestones_reached['first_latency_satisfy']:
            if metrics['avg_latency'] <= p.latencyThreshold:
                milestone_reward += 2.0
                self.milestones_reached['first_latency_satisfy'] = True
                print(f"\n🎯 MILESTONE: First latency satisfaction at step {self.total_steps}!")
        
        if not self.milestones_reached['first_resource_satisfy']:
            if metrics['cpu_violations'] == 0 and metrics['prb_violations'] == 0:
                milestone_reward += 2.0
                self.milestones_reached['first_resource_satisfy'] = True
                print(f"\n🎯 MILESTONE: First resource satisfaction at step {self.total_steps}!")
        
        if not self.milestones_reached['first_all_satisfy']:
            all_satisfied = (
                metrics['avg_drop_rate'] <= p.dropCallThreshold and
                metrics['avg_latency'] <= p.latencyThreshold and
                metrics['cpu_violations'] == 0 and
                metrics['prb_violations'] == 0
            )
            if all_satisfied:
                milestone_reward += 5.0
                self.milestones_reached['first_all_satisfy'] = True
                print(f"\n🎉 MAJOR MILESTONE: First FULL constraint satisfaction at step {self.total_steps}!")
        
        # Sustained satisfaction milestones
        if not self.milestones_reached['sustained_10_steps']:
            if self.consecutive_satisfactions['all'] >= 10:
                milestone_reward += 3.0
                self.milestones_reached['sustained_10_steps'] = True
                print(f"\n⭐ MILESTONE: 10 consecutive satisfied steps at {self.total_steps}!")
        
        if not self.milestones_reached['sustained_50_steps']:
            if self.consecutive_satisfactions['all'] >= 50:
                milestone_reward += 5.0
                self.milestones_reached['sustained_50_steps'] = True
                print(f"\n🌟 MILESTONE: 50 consecutive satisfied steps at {self.total_steps}!")
        
        if not self.milestones_reached['sustained_100_steps']:
            if self.consecutive_satisfactions['all'] >= 100:
                milestone_reward += 10.0
                self.milestones_reached['sustained_100_steps'] = True
                print(f"\n✨ MAJOR MILESTONE: 100 consecutive satisfied steps at {self.total_steps}!")
        
        return milestone_reward
    
    def _compute_curriculum_bonus(self, drop_progress: float, 
                                  latency_progress: float, 
                                  resource_progress: float) -> float:
        """
        Adaptive curriculum learning: reward progress on HARDEST constraints more.
        This focuses agent's attention on bottlenecks.
        """
        # Weight by difficulty (harder constraints get more reward)
        weighted_reward = (
            drop_progress * self.constraint_difficulty['drop_rate'] +
            latency_progress * self.constraint_difficulty['latency'] +
            resource_progress * self.constraint_difficulty['resources']
        ) / 3.0
        
        return weighted_reward * 0.5  # Scale appropriately
    
    def _update_satisfaction_counters(self, metrics: Dict):
        """Update consecutive satisfaction counters."""
        p = self.env.sim_params
        
        # Individual constraints
        if metrics['avg_drop_rate'] <= p.dropCallThreshold:
            self.consecutive_satisfactions['drop_rate'] += 1
        else:
            self.consecutive_satisfactions['drop_rate'] = 0
        
        if metrics['avg_latency'] <= p.latencyThreshold:
            self.consecutive_satisfactions['latency'] += 1
        else:
            self.consecutive_satisfactions['latency'] = 0
        
        if metrics['cpu_violations'] == 0 and metrics['prb_violations'] == 0:
            self.consecutive_satisfactions['resources'] += 1
        else:
            self.consecutive_satisfactions['resources'] = 0
        
        # All constraints
        all_satisfied = (
            metrics['avg_drop_rate'] <= p.dropCallThreshold and
            metrics['avg_latency'] <= p.latencyThreshold and
            metrics['cpu_violations'] == 0 and
            metrics['prb_violations'] == 0
        )
        
        if all_satisfied:
            self.consecutive_satisfactions['all'] += 1
        else:
            self.consecutive_satisfactions['all'] = 0
    
    def _update_constraint_difficulty(self):
        """
        Update difficulty estimate for each constraint.
        Difficulty = 1 - average satisfaction rate
        Higher difficulty = harder to satisfy = more reward for progress
        """
        for constraint_name in ['drop_rate', 'latency', 'resources']:
            if len(self.constraint_progress[constraint_name]['long']) > 10:
                # Average progress over long window
                avg_progress = np.mean(list(self.constraint_progress[constraint_name]['long']))
                
                # Difficulty is inverse of progress
                difficulty = 1.0 - avg_progress
                
                # Smooth update (EMA)
                alpha = 0.1
                self.constraint_difficulty[constraint_name] = (
                    (1 - alpha) * self.constraint_difficulty[constraint_name] +
                    alpha * difficulty
                )
                
                # Clamp to reasonable range
                self.constraint_difficulty[constraint_name] = max(0.5, min(2.0, 
                    self.constraint_difficulty[constraint_name]))
    
    def _update_training_stage(self):
        """Update training stage based on total steps."""
        if self.total_steps < self.stage_thresholds['early_to_middle']:
            self.training_stage = 'early'
        elif self.total_steps < self.stage_thresholds['middle_to_late']:
            if self.training_stage == 'early':
                self.training_stage = 'middle'
                print(f"\n{'='*70}")
                print(f"Training Stage: EARLY → MIDDLE (step {self.total_steps})")
                print(f"  Reducing HER assistance to 70%")
                print(f"{'='*70}\n")
        else:
            if self.training_stage != 'late':
                self.training_stage = 'late'
                print(f"\n{'='*70}")
                print(f"Training Stage: MIDDLE → LATE (step {self.total_steps})")
                print(f"  Reducing HER assistance to 40%")
                print(f"{'='*70}\n")
    
    def _log_episode_summary(self, info: Dict):
        """Log episode summary with HER metrics."""
        if self.total_episodes % 10 == 0:  # Log every 10 episodes
            print(f"\n{'='*70}")
            print(f"HER Episode {self.total_episodes} Summary (Total steps: {self.total_steps})")
            print(f"{'='*70}")
            print(f"  Training Stage: {self.training_stage.upper()}")
            print(f"  Consecutive Satisfactions:")
            print(f"    Drop Rate: {self.consecutive_satisfactions['drop_rate']}")
            print(f"    Latency: {self.consecutive_satisfactions['latency']}")
            print(f"    Resources: {self.consecutive_satisfactions['resources']}")
            print(f"    ALL: {self.consecutive_satisfactions['all']}")
            
            print(f"  Constraint Difficulty (curriculum):")
            for name, diff in self.constraint_difficulty.items():
                print(f"    {name}: {diff:.2f}")
            
            print(f"  Milestones Reached: {sum(self.milestones_reached.values())}/{len(self.milestones_reached)}")
            print(f"{'='*70}\n")

class GatedRewardWrapper(gym.Wrapper):
    """
    Provides a simple and effective reward signal for constraint satisfaction.
    - Returns a positive reward for energy efficiency ONLY when all constraints are met.
    - Returns a fixed negative penalty (-1.0) if ANY constraint is violated.
    This gives the agent a very clear learning signal.
    """
    def __init__(self, env: FiveGEnv):
        super().__init__(env)
        # --- FINAL TUNING: Use .unwrapped for safety ---
        self.sim_params = self.env.unwrapped.sim_params

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        metrics = info 
        
        is_compliant = self._check_constraints(metrics)
        
        if not is_compliant:
            # A strong, consistent penalty for violating constraints.
            reward = -1.0
        else:
            # A positive reward for energy saving, only available in compliant states.
            total_energy = metrics.get('total_energy', 0)
            n_cells = self.env.unwrapped.n_cells
            
            # Use a realistic maximum energy consumption as the baseline for reward calculation.
            max_power_watts = 10**((self.sim_params.maxTxPower - 30) / 10)
            max_energy = n_cells * (self.sim_params.basePower + max_power_watts)
            
            efficiency = 1.0 - (total_energy / max(1, max_energy))
            reward = max(0.0, efficiency)

        return obs, reward, terminated, truncated, info

    def _check_constraints(self, metrics: Dict[str, Any]) -> bool:
        """Checks if all critical performance and resource constraints are satisfied."""
        if metrics.get('avg_drop_rate', 100) > self.sim_params.dropCallThreshold: return False
        if metrics.get('avg_latency', 1000) > self.sim_params.latencyThreshold: return False
        if metrics.get('cpu_violations', 1) > 0: return False
        if metrics.get('prb_violations', 1) > 0: return False
        if metrics.get('connection_rate', 0) < 90.0: return False # Ensure most UEs are connected
        return True

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        # This is the correct, compliant signature
        return self.env.reset(seed=seed, options=options)
