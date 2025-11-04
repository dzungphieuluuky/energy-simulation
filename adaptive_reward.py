# adaptive_reward.py
import numpy as np
from typing import Dict, Any, List
from enum import Enum

class ConstraintPriority(Enum):
    CRITICAL = 3    # Service disruption (drop rate, connectivity)
    HIGH = 2        # Performance degradation (latency)
    MEDIUM = 1      # Resource saturation (CPU, PRB)

class AdaptiveRewardComputer:
    """
    Enhanced reward computer with adaptive constraint enforcement
    and multi-stage training support.
    """
    
    def __init__(self, sim_params, n_cells: int, training_stage: str = "stable"):
        self.sim_params = sim_params
        self.n_cells = n_cells
        self.training_stage = training_stage  # "early", "medium", "stable"
        
        # Adaptive parameters based on training stage
        self.stage_config = self._get_stage_config(training_stage)
        
        # Enhanced tracking
        self.qos_compliant_steps = 0
        self.consecutive_violations = 0
        self.total_steps = 0
        self.constraint_history = []
        self.violation_pattern = {}
        
        # Constraint criticality weights
        self.constraint_weights = {
            'drop_rate': ConstraintPriority.CRITICAL,
            'connectivity': ConstraintPriority.CRITICAL,
            'latency': ConstraintPriority.HIGH,
            'cpu': ConstraintPriority.MEDIUM,
            'prb': ConstraintPriority.MEDIUM
        }
        
        # Momentum tracking for sustained performance
        self.compliance_momentum = 0.0
        self.energy_efficiency_trend = 0.0
        
    def _get_stage_config(self, stage: str) -> Dict[str, Any]:
        """Get configuration parameters for current training stage."""
        configs = {
            "early": {
                "base_penalty": -5.0,
                "min_compliant_steps": 20,
                "energy_reward_scale": 1.0,
                "constraint_tolerance": 0.05,  # 5% tolerance on thresholds
                "enable_energy_early": True
            },
            "medium": {
                "base_penalty": -8.0,
                "min_compliant_steps": 35,
                "energy_reward_scale": 1.5,
                "constraint_tolerance": 0.02,
                "enable_energy_early": True
            },
            "stable": {
                "base_penalty": -10.0,
                "min_compliant_steps": 50,
                "energy_reward_scale": 2.0,
                "constraint_tolerance": 0.00,
                "enable_energy_early": False
            }
        }
        return configs.get(stage, configs["stable"])
    
    def compute_reward(self, metrics: Dict[str, Any], cells_info: list = None) -> float:
        """
        Enhanced reward computation with adaptive constraint handling.
        """
        self.total_steps += 1
        p = self.sim_params
        
        # Apply stage-specific tolerance to constraints
        adjusted_metrics = self._adjust_constraint_thresholds(metrics)
        
        # Identify violations with criticality awareness
        violations = self._identify_violations_with_priority(adjusted_metrics, p)
        
        if violations['has_violations']:
            reward = self._compute_adaptive_penalty(violations, metrics)
            self._update_violation_pattern(violations)  # This method now exists
        else:
            reward = self._compute_enhanced_positive_reward(metrics, cells_info)
        
        # Update momentum and trends
        self._update_performance_trends(violations, metrics)
        
        return reward
    
    def _update_violation_pattern(self, violations: Dict[str, Any]):
        """Update violation pattern tracking for repeated violations."""
        # Track current violation pattern
        current_pattern = []
        for vtype in ['drop_rate', 'latency', 'cpu', 'prb', 'connectivity']:
            if vtype in violations and violations[vtype]['violated']:
                current_pattern.append(vtype)
        
        pattern_key = tuple(sorted(current_pattern))
        
        # Update pattern count
        self.violation_pattern[pattern_key] = self.violation_pattern.get(pattern_key, 0) + 1
        
        # Clean up old patterns to prevent memory growth
        if len(self.violation_pattern) > 100:
            # Remove least frequent patterns
            pattern_items = list(self.violation_pattern.items())
            pattern_items.sort(key=lambda x: x[1])
            for key, _ in pattern_items[:20]:  # Remove 20 least frequent
                del self.violation_pattern[key]
    
    def _adjust_constraint_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tolerance to constraints based on training stage."""
        tolerance = self.stage_config["constraint_tolerance"]
        if tolerance == 0:
            return metrics
            
        adjusted = metrics.copy()
        p = self.sim_params
        
        # Relax thresholds during early training
        adjusted['avg_drop_rate'] = metrics['avg_drop_rate'] * (1 - tolerance)
        adjusted['avg_latency'] = metrics['avg_latency'] * (1 - tolerance)
        adjusted['max_cpu_usage'] = metrics['max_cpu_usage'] * (1 - tolerance)
        adjusted['max_prb_usage'] = metrics['max_prb_usage'] * (1 - tolerance)
        adjusted['connection_rate'] = metrics['connection_rate'] + (1 - metrics['connection_rate']) * tolerance
        
        return adjusted
    
    def _identify_violations_with_priority(self, metrics: Dict[str, Any], p) -> Dict[str, Any]:
        """Identify violations with criticality information."""
        violations = {
            'has_violations': False,
            'critical_violations': 0,
            'high_violations': 0,
            'medium_violations': 0
        }
        
        # Check each constraint with priority
        constraints = [
            ('drop_rate', metrics['avg_drop_rate'] > p.dropCallThreshold,
             self._compute_severity(metrics['avg_drop_rate'], p.dropCallThreshold)),
            ('latency', metrics['avg_latency'] > p.latencyThreshold,
             self._compute_severity(metrics['avg_latency'], p.latencyThreshold)),
            ('cpu', metrics['cpu_violations'] > 0,
             self._compute_resource_severity(metrics['max_cpu_usage'], p.cpuThreshold, metrics['cpu_violations'])),
            ('prb', metrics['prb_violations'] > 0,
             self._compute_resource_severity(metrics['max_prb_usage'], p.prbThreshold, metrics['prb_violations'])),
            ('connectivity', metrics['connection_rate'] < 0.95,
             self._compute_severity(0.95, metrics['connection_rate']))  # Inverted for connectivity
        ]
        
        for name, violated, severity in constraints:
            if violated:
                violations['has_violations'] = True
                priority = self.constraint_weights[name]
                violations[f'{name}'] = {
                    'violated': True,
                    'severity': severity,
                    'priority': priority
                }
                
                # Count by priority
                if priority == ConstraintPriority.CRITICAL:
                    violations['critical_violations'] += 1
                elif priority == ConstraintPriority.HIGH:
                    violations['high_violations'] += 1
                else:
                    violations['medium_violations'] += 1
        
        return violations
    
    def _compute_adaptive_penalty(self, violations: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """Compute penalty with adaptive scaling based on violation patterns."""
        # Base penalty from config
        penalty = self.stage_config["base_penalty"]
        
        # Priority-based penalty scaling
        critical_multiplier = 1.0 + violations['critical_violations'] * 2.0
        high_multiplier = 1.0 + violations['high_violations'] * 1.5
        medium_multiplier = 1.0 + violations['medium_violations'] * 1.0
        
        penalty *= critical_multiplier * high_multiplier * medium_multiplier
        
        # Pattern-based penalty: additional penalty for repeated violation types
        pattern_penalty = self._compute_pattern_penalty(violations)
        penalty += pattern_penalty
        
        # Consecutive violation penalty (softer than original)
        if self.consecutive_violations > 3:
            consecutive_factor = 1.0 + (min(self.consecutive_violations - 3, 20) * 0.1)
            penalty *= consecutive_factor
        
        # Update tracking
        self.qos_compliant_steps = 0
        self.consecutive_violations += 1
        
        # Ensure reasonable penalty bounds
        return max(penalty, -200.0)
    
    def _compute_pattern_penalty(self, violations: Dict[str, Any]) -> float:
        """Additional penalty for repeated violation patterns."""
        pattern_penalty = 0.0
        
        # Track current violation pattern
        current_pattern = []
        for vtype in ['drop_rate', 'latency', 'cpu', 'prb', 'connectivity']:
            if vtype in violations and violations[vtype]['violated']:
                current_pattern.append(vtype)
        
        pattern_key = tuple(sorted(current_pattern))
        
        # Check if this pattern has occurred recently
        if pattern_key in self.violation_pattern:
            recurrence_count = self.violation_pattern[pattern_key]
            pattern_penalty = -2.0 * min(recurrence_count, 5)  # Cap at 5x
        
        return pattern_penalty
    
    def _compute_enhanced_positive_reward(self, metrics: Dict[str, Any], cells_info: list = None) -> float:
        """Compute positive reward with multi-objective optimization."""
        self.qos_compliant_steps += 1
        self.consecutive_violations = 0
        
        # Base compliance reward with momentum
        base_reward = 1.0 + (self.compliance_momentum * 0.5)
        
        # Progressive compliance bonus (softer than original)
        compliance_bonus = 0.0
        if self.qos_compliant_steps > 10:
            compliance_bonus = np.log1p(self.qos_compliant_steps - 10) * 0.3
        
        # Multi-objective optimization rewards
        energy_reward = self._compute_adaptive_energy_reward(metrics, cells_info)
        quality_reward = self._compute_connection_quality_reward(metrics)
        fairness_reward = self._compute_fairness_reward(cells_info)
        stability_reward = self._compute_stability_reward(metrics)
        
        total_reward = (base_reward + compliance_bonus + energy_reward + 
                       quality_reward + fairness_reward + stability_reward)
        
        # Log detailed reward breakdown at milestones
        if self.qos_compliant_steps in [25, 50, 100, 200]:
            self._log_reward_breakdown(compliance_bonus, energy_reward, 
                                     quality_reward, fairness_reward, stability_reward)
        
        return total_reward
    
    def _compute_adaptive_energy_reward(self, metrics: Dict[str, Any], cells_info: list) -> float:
        """Compute energy reward with early training support."""
        # Early training: small energy hints even before full compliance
        if (self.stage_config["enable_energy_early"] and 
            self.qos_compliant_steps < self.stage_config["min_compliant_steps"]):
            energy_hint = self._compute_energy_efficiency(metrics) * 0.1
            return energy_hint
        
        # Standard energy reward after minimum compliance
        if self.qos_compliant_steps >= self.stage_config["min_compliant_steps"]:
            energy_efficiency = self._compute_energy_efficiency(metrics)
            power_optimization = 1.0 - metrics.get('avg_power_ratio', 0.5)
            
            # Scale with training progress
            progress_factor = min(1.0, (self.qos_compliant_steps - 
                                      self.stage_config["min_compliant_steps"]) / 50.0)
            
            energy_reward = (energy_efficiency * 0.7 + power_optimization * 0.3) * progress_factor
            return energy_reward * self.stage_config["energy_reward_scale"]
        
        return 0.0
    
    def _compute_fairness_reward(self, cells_info: list) -> float:
        """Reward fair resource distribution across cells."""
        if cells_info is None or len(cells_info) == 0:
            return 0.0
        
        # Calculate fairness of load distribution
        loads = [cell.current_load for cell in cells_info]
        if len(loads) < 2:
            return 0.0
        
        # Use coefficient of variation to measure fairness (lower is better)
        load_std = np.std(loads)
        load_mean = np.mean(loads)
        if load_mean == 0:
            return 0.0
            
        fairness = 1.0 / (1.0 + (load_std / load_mean))
        return fairness * 0.5
    
    def _compute_stability_reward(self, metrics: Dict[str, Any]) -> float:
        """Reward stable performance without oscillations."""
        # Track metric stability over recent steps
        if len(self.constraint_history) < 5:
            return 0.0
        
        recent_metrics = self.constraint_history[-5:]
        stability_score = 0.0
        
        # Check stability of key metrics
        for metric in ['avg_drop_rate', 'avg_latency', 'connection_rate']:
            if metric in recent_metrics[0]:
                values = [step[metric] for step in recent_metrics if metric in step]
                if len(values) >= 3:
                    # Lower variance = higher stability
                    variance = np.var(values)
                    stability = 1.0 / (1.0 + variance * 10)
                    stability_score += stability * 0.2
        
        return min(stability_score, 1.0)
    
    def _compute_connection_quality_reward(self, metrics: Dict[str, Any]) -> float:
        """Reward maintaining high connection quality."""
        p = self.sim_params
        
        # Connection rate bonus (above 95%)
        connection_bonus = 0.0
        if metrics['connection_rate'] >= 0.95:
            # Bonus for every 1% above 95%
            connection_bonus = (metrics['connection_rate'] - 0.95) * 10.0
        
        # Drop rate quality bonus (how far below threshold)
        drop_quality = 0.0
        if metrics['avg_drop_rate'] < p.dropCallThreshold:
            # More reward for being far below threshold
            margin = (p.dropCallThreshold - metrics['avg_drop_rate']) / p.dropCallThreshold
            drop_quality = margin * 0.3
        
        # Latency quality bonus (how far below threshold)
        latency_quality = 0.0
        if metrics['avg_latency'] < p.latencyThreshold:
            margin = (p.latencyThreshold - metrics['avg_latency']) / p.latencyThreshold
            latency_quality = margin * 0.3
        
        total_quality_reward = connection_bonus + drop_quality + latency_quality
        
        return total_quality_reward
    
    def _update_performance_trends(self, violations: Dict[str, Any], metrics: Dict[str, Any]):
        """Update momentum and trend tracking."""
        # Update compliance momentum (EMA)
        compliance = 0.0 if violations['has_violations'] else 1.0
        self.compliance_momentum = (0.95 * self.compliance_momentum + 0.05 * compliance)
        
        # Update energy efficiency trend
        energy_eff = self._compute_energy_efficiency(metrics)
        self.energy_efficiency_trend = (0.9 * self.energy_efficiency_trend + 0.1 * energy_eff)
        
        # Maintain constraint history
        self.constraint_history.append({
            'drop_rate': metrics['avg_drop_rate'],
            'latency': metrics['avg_latency'],
            'connection_rate': metrics['connection_rate']
        })
        # Keep only recent history
        if len(self.constraint_history) > 20:
            self.constraint_history.pop(0)
    
    def _compute_energy_efficiency(self, metrics: Dict[str, Any]) -> float:
        """Compute normalized energy efficiency metric."""
        total_energy = metrics['total_energy']
        max_power_consumption = 10**((self.sim_params.maxTxPower - 30)/10)
        max_possible_energy = self.n_cells * (self.sim_params.basePower + max_power_consumption)

        energy_efficiency = 1.0 - (total_energy / max(1, max_possible_energy))
        return max(0.0, energy_efficiency)
    
    def _compute_severity(self, value: float, threshold: float) -> float:
        """Compute normalized severity of violation."""
        return max(0.0, (value - threshold) / threshold) if value > threshold else 0.0
    
    def _compute_resource_severity(self, max_usage: float, threshold: float, violation_count: int) -> float:
        """Compute severity for resource violations."""
        usage_severity = self._compute_severity(max_usage, threshold)
        count_severity = violation_count / self.n_cells
        return (usage_severity * 0.7 + count_severity * 0.3)
    
    def _log_reward_breakdown(self, compliance_bonus: float, energy_reward: float,
                            quality_reward: float, fairness_reward: float, stability_reward: float):
        """Log detailed reward breakdown."""
        print(f"\n📊 Reward Breakdown at {self.qos_compliant_steps} steps:")
        print(f"   Compliance Bonus: {compliance_bonus:.2f}")
        print(f"   Energy Reward: {energy_reward:.2f}")
        print(f"   Quality Reward: {quality_reward:.2f}")
        print(f"   Fairness Reward: {fairness_reward:.2f}")
        print(f"   Stability Reward: {stability_reward:.2f}")
        print(f"   Total Momentum: {self.compliance_momentum:.3f}")
    
    def advance_training_stage(self):
        """Advance to next training stage for curriculum learning."""
        stages = ["early", "medium", "stable"]
        current_index = stages.index(self.training_stage)
        if current_index < len(stages) - 1:
            self.training_stage = stages[current_index + 1]
            self.stage_config = self._get_stage_config(self.training_stage)
            print(f"\n🎯 Advanced to {self.training_stage} training stage")
    
    def reset(self):
        """Reset episode-level tracking."""
        self.qos_compliant_steps = 0
        self.consecutive_violations = 0
        self.violation_pattern = {}
        # Don't reset momentum completely - it tracks long-term performance
        self.compliance_momentum *= 0.8
        self.energy_efficiency_trend *= 0.8
        self.constraint_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'qos_compliant_steps': self.qos_compliant_steps,
            'consecutive_violations': self.consecutive_violations,
            'total_steps': self.total_steps,
            'compliance_rate': (self.qos_compliant_steps / max(1, self.total_steps)) * 100,
            'training_stage': self.training_stage,
            'compliance_momentum': self.compliance_momentum,
            'energy_efficiency_trend': self.energy_efficiency_trend,
            'violation_patterns_count': len(self.violation_pattern)
        }