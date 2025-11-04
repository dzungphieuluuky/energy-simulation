"""
Improved reward computation that enforces constraint satisfaction at all cost.

Key improvements:
1. Exponentially increasing penalties for constraint violations
2. No positive rewards until ALL constraints are satisfied
3. Grace period removed - immediate enforcement
4. Graduated penalties based on violation severity
5. Energy rewards only after sustained constraint satisfaction
"""

import numpy as np
from typing import Dict, Any


class ImprovedRewardComputer:
    """
    Reward computer that enforces constraints strictly.
    
    Design Philosophy:
    - Constraint satisfaction is PRIMARY objective
    - Energy optimization is SECONDARY objective
    - Agent gets NO positive reward if ANY constraint is violated
    - Penalties scale with both severity and duration of violations
    """
    
    def __init__(self, sim_params, n_cells: int):
        self.sim_params = sim_params
        self.n_cells = n_cells
        
        # Tracking state across steps
        self.qos_compliant_steps = 0
        self.consecutive_violations = 0
        self.total_steps = 0
        
        # Penalty scaling factors
        self.base_violation_penalty = -10.0  # Much larger base penalty
        self.violation_multiplier = 1.5  # Exponential growth factor
        
        # Thresholds for reward unlocking
        self.min_compliant_steps_for_energy_reward = 50  # Must maintain compliance longer
        self.energy_reward_scale = 2.0  # Scale of energy reward when unlocked
        
    def compute_reward(self, metrics: Dict[str, Any], cells_info: list = None) -> float:
        """
        Compute reward with strict constraint enforcement.
        
        Returns:
            float: Reward value (negative if constraints violated, positive only if optimizing)
        """
        self.total_steps += 1
        p = self.sim_params
        
        # =====================================================================
        # STEP 1: Check ALL constraints
        # =====================================================================
        violations = self._identify_all_violations(metrics, p)
        
        if violations['has_violations']:
            # Reset compliance counter
            self.qos_compliant_steps = 0
            self.consecutive_violations += 1
            
            # Compute penalty
            penalty = self._compute_violation_penalty(violations)
            
            # Log violation details every 10 consecutive violations
            if self.consecutive_violations % 10 == 0:
                print(f"\n⚠️  WARNING: {self.consecutive_violations} consecutive violations!")
                for vtype, vinfo in violations.items():
                    if vtype != 'has_violations' and vinfo['violated']:
                        print(f"   - {vtype}: {vinfo['value']:.2f} > {vinfo['threshold']:.2f}")
            
            return penalty
        
        # =====================================================================
        # STEP 2: All constraints satisfied - compute positive reward
        # =====================================================================
        self.qos_compliant_steps += 1
        self.consecutive_violations = 0
        
        # Base reward for maintaining compliance
        compliance_reward = 1.0
        
        # Bonus for sustained compliance
        if self.qos_compliant_steps > 20:
            compliance_bonus = np.log10(self.qos_compliant_steps - 19) * 0.5
            compliance_reward += compliance_bonus
        
        # Energy optimization reward (only after sustained compliance)
        energy_reward = 0.0
        if self.qos_compliant_steps >= self.min_compliant_steps_for_energy_reward:
            energy_reward = self._compute_energy_reward(metrics, cells_info)
        
        # Connection quality bonus
        connection_reward = self._compute_connection_quality_reward(metrics)
        
        total_reward = compliance_reward + energy_reward + connection_reward
        
        # Log milestone achievements
        if self.qos_compliant_steps in [20, 50, 100, 200, 500]:
            print(f"\n✅ MILESTONE: {self.qos_compliant_steps} consecutive compliant steps!")
            print(f"   Reward breakdown: Compliance={compliance_reward:.2f}, "
                  f"Energy={energy_reward:.2f}, Connection={connection_reward:.2f}")
        
        return total_reward
    
    def _identify_all_violations(self, metrics: Dict[str, Any], p) -> Dict[str, Any]:
        """
        Identify all constraint violations with detailed information.
        
        Returns:
            Dict containing violation status and severity for each constraint
        """
        violations = {
            'has_violations': False,
            'drop_rate': {
                'violated': False,
                'value': metrics['avg_drop_rate'],
                'threshold': p.dropCallThreshold,
                'severity': 0.0
            },
            'latency': {
                'violated': False,
                'value': metrics['avg_latency'],
                'threshold': p.latencyThreshold,
                'severity': 0.0
            },
            'cpu': {
                'violated': False,
                'value': metrics['max_cpu_usage'],
                'threshold': p.cpuThreshold,
                'severity': 0.0,
                'count': metrics['cpu_violations']
            },
            'prb': {
                'violated': False,
                'value': metrics['max_prb_usage'],
                'threshold': p.prbThreshold,
                'severity': 0.0,
                'count': metrics['prb_violations']
            },
            'connectivity': {
                'violated': False,
                'value': metrics['connection_rate'],
                'threshold': 0.95,  # At least 95% UEs connected
                'severity': 0.0
            }
        }
        
        # Check drop rate
        if metrics['avg_drop_rate'] > p.dropCallThreshold:
            violations['drop_rate']['violated'] = True
            violations['drop_rate']['severity'] = (
                (metrics['avg_drop_rate'] - p.dropCallThreshold) / p.dropCallThreshold
            )
            violations['has_violations'] = True
        
        # Check latency
        if metrics['avg_latency'] > p.latencyThreshold:
            violations['latency']['violated'] = True
            violations['latency']['severity'] = (
                (metrics['avg_latency'] - p.latencyThreshold) / p.latencyThreshold
            )
            violations['has_violations'] = True
        
        # Check CPU usage (any cell over threshold is violation)
        if metrics['cpu_violations'] > 0:
            violations['cpu']['violated'] = True
            # Severity based on how far over threshold and how many cells
            cpu_overage = (metrics['max_cpu_usage'] - p.cpuThreshold) / p.cpuThreshold
            cell_fraction = metrics['cpu_violations'] / self.n_cells
            violations['cpu']['severity'] = cpu_overage * (1 + cell_fraction)
            violations['has_violations'] = True
        
        # Check PRB usage (any cell over threshold is violation)
        if metrics['prb_violations'] > 0:
            violations['prb']['violated'] = True
            prb_overage = (metrics['max_prb_usage'] - p.prbThreshold) / p.prbThreshold
            cell_fraction = metrics['prb_violations'] / self.n_cells
            violations['prb']['severity'] = prb_overage * (1 + cell_fraction)
            violations['has_violations'] = True
        
        # Check connectivity (strict requirement)
        if metrics['connection_rate'] < 0.95:
            violations['connectivity']['violated'] = True
            violations['connectivity']['severity'] = (
                (0.95 - metrics['connection_rate']) / 0.95
            )
            violations['has_violations'] = True
        
        return violations
    
    def _compute_violation_penalty(self, violations: Dict[str, Any]) -> float:
        """
        Compute penalty for constraint violations.
        
        Penalty structure:
        - Base penalty for ANY violation
        - Additional penalty per violation type
        - Severity multiplier based on how much threshold is exceeded
        - Exponential scaling for consecutive violations
        """
        penalty = self.base_violation_penalty
        
        # Accumulate penalties from each violation type
        for vtype, vinfo in violations.items():
            if vtype == 'has_violations':
                continue
            
            if vinfo['violated']:
                # Base penalty for this violation type
                type_penalty = -5.0
                
                # Scale by severity (how far over threshold)
                severity_multiplier = 1.0 + (2.0 * vinfo['severity'])
                
                # Additional penalty for resource violations affecting multiple cells
                if vtype in ['cpu', 'prb'] and 'count' in vinfo:
                    count_multiplier = 1.0 + (vinfo['count'] / self.n_cells)
                    severity_multiplier *= count_multiplier
                
                penalty += type_penalty * severity_multiplier
        
        # Exponential penalty for consecutive violations (forces agent to fix issues)
        if self.consecutive_violations > 5:
            consecutive_multiplier = self.violation_multiplier ** min(
                (self.consecutive_violations - 5) / 10.0, 3.0  # Cap at 3x exponential
            )
            penalty *= consecutive_multiplier
        
        # Ensure penalty doesn't become infinite
        penalty = max(penalty, -1000.0)
        
        return penalty
    
    def _compute_energy_reward(self, metrics: Dict[str, Any], 
                               cells_info: list = None) -> float:
        """
        Compute energy efficiency reward (only when constraints satisfied).
        
        Strategy:
        - Reward reducing total energy consumption
        - Reward balanced power distribution across cells
        - Penalize unnecessary high power usage
        """
        p = self.sim_params
        
        # Calculate energy efficiency
        total_energy = metrics['total_energy']
        
        # Maximum possible energy (all cells at max power + base consumption)
        max_power_consumption = 10**((p.maxTxPower - 30)/10)  # dBm to watts
        max_possible_energy = self.n_cells * (p.basePower + max_power_consumption)

        # Normalized energy efficiency (0 = worst, 1 = best)
        energy_efficiency = 1.0 - (total_energy / max(1, max_possible_energy))
        energy_efficiency = max(0.0, energy_efficiency)  # Clamp to [0, 1]
        
        # Compute average power ratio (how close to minimum power)
        avg_power_ratio = metrics['avg_power_ratio']
        
        # Reward low power operation
        power_reduction_bonus = (1.0 - avg_power_ratio) * 0.5
        
        # Unlock factor: gradually increase energy reward importance
        unlock_progress = (self.qos_compliant_steps - self.min_compliant_steps_for_energy_reward)
        unlock_factor = min(1.0, unlock_progress / 100.0)
        
        # Combined energy reward
        energy_reward = (energy_efficiency + power_reduction_bonus) * unlock_factor * self.energy_reward_scale
        
        return energy_reward
    
    def _compute_connection_quality_reward(self, metrics: Dict[str, Any]) -> float:
        """
        Reward maintaining high connection quality.
        
        Encourages:
        - High connection rate
        - Low drop rate (below threshold but still reward improvement)
        - Low latency (below threshold but still reward improvement)
        """
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
    
    def reset(self):
        """Reset episode-level tracking variables."""
        self.qos_compliant_steps = 0
        self.consecutive_violations = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'qos_compliant_steps': self.qos_compliant_steps,
            'consecutive_violations': self.consecutive_violations,
            'total_steps': self.total_steps,
            'compliance_rate': (self.qos_compliant_steps / max(1, self.total_steps)) * 100
        }