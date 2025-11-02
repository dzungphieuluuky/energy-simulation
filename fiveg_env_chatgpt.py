# fiveg_env.py
"""
Merged FiveG environment:
- Implements FiveGEnv class (API/variable names matching your uploaded file)
- Contains run_simulation_step(...) as the internal simulation driver (replaces sim.run_simulation_step)
- Uses Cell and UE lightweight classes for attribute access (cell.txPower, ue.rsrp, ...)
- Reuses path-loss, measurement, CPU/PRB/energy math from the translated MATLAB PDF
"""
import os
import math
import json
import numpy as np
import gymnasium as gym
import numba
from numba import jit, float64, int32, boolean, types
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional

from fiveg_objects import Cell, UE
from numba_utils import NumbaUE, NumbaCell
from simulation_logic import *
from scenario_creator import *

# neighbor dtype fallback used in observation init (original code imported neighbor_dtype)
neighbor_dtype = np.int32
# ------------------------------
# FiveGEnv class (API + reward + diagnostics merged from uploaded file)
# ------------------------------
class FiveGEnv(gym.Env):
    """A Gymnasium environment with improved stability and performance."""

    def __init__(self, env_config: Dict[str, Any], max_cells: int = 57) -> None:
        super().__init__()
        self.config: Dict[str, Any] = dict(env_config)
        self._set_default_config()

        self.time_step_duration: float = float(self.config['timeStep'])
        self.max_time_steps: int = int(self.config['simTime'])
        self.max_neighbors: int = 8
        self.max_cells: int = int(max_cells)
        self.state_dim_per_cell: int = 25

        # Define spaces
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.max_cells,), dtype=np.float32)
        state_dim: int = 17 + 14 + (self.max_cells * self.state_dim_per_cell)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # Initialize state variables
        self.cells: List[Cell] = []
        self.ues: List[UE] = []
        self.n_cells: int = 0
        self.n_ues: int = 0
        self.current_step: int = 0
        self.total_energy_kwh: float = 0.0
        self.total_episodes: int = 0
        self.previous_powers: np.ndarray = np.zeros(self.max_cells, dtype=np.float32)
        self.neighbor_measurements: Optional[np.ndarray] = None
        self._warm_up_numba()

    def _warm_up_numba(self):
        """Warm up Numba functions to avoid compilation delay during training."""
        print("Warming up Numba functions...")

        # Create dummy data
        dummy_ue = NumbaUE()
        dummy_cell = NumbaCell()

        dummy_ues = [dummy_ue]
        dummy_cells = [dummy_cell]

        # Call each Numba function with dummy data
        numba_update_signal_measurements(dummy_ues, dummy_cells, -115.0, 0.0, 42)
        numba_update_ue_mobility(dummy_ues, 1.0, 0.0, 42, 1000.0)
        numba_update_cell_resource_usage(dummy_ues, dummy_cells)
        numba_generate_traffic(dummy_ues, 30.0, 1, 42, 0)

        print("Numba functions warmed up!")

    def _set_default_config(self) -> None:
        """Set default configuration values with type safety."""
        defaults: Dict[str, Any] = {
            'timeStep': 1.0, 'simTime': 600, 'carrierFrequency': 3.5e9,
            'minTxPower': 30.0, 'maxTxPower': 46.0, 'basePower': 1000.0,
            'idlePower': 250.0, 'dropCallThreshold': 1.0, 'latencyThreshold': 50.0,
            'cpuThreshold': 80.0, 'prbThreshold': 80.0, 'trafficLambda': 30.0,
            'peakHourMultiplier': 1.0, 'numSites': 7, 'numUEs': 210, 'isd': 200.0,
            'deploymentScenario': 'indoor_hotspot', 'seed': 42
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)

    # ------------------------------
    # Scenario setup
    # ------------------------------
    def _setup_scenario(self):
        sites = create_hex_layout(self.config['numSites'], self.config.get('isd', 200.0), int(self.config.get('seed', 42)))
        self.cells = configure_cells_from_sites(self.config, sites)
        ues = initialize_ues_from_config(self.config, sites, int(self.config.get('seed', 42)))
        self.ues = ues
        self.n_cells = len(self.cells)
        self.n_ues = len(self.ues)
        if self.n_cells > self.max_cells:
            raise ValueError(f"Scenario has {self.n_cells} cells, but env configured for max {self.max_cells}.")

    # ------------------------------
    # Observation builder (mirrors createRLState/mapping in your uploaded file)
    # ------------------------------
    def _get_obs(self):
        sim_features = [
            float(self.n_cells), float(self.n_ues), float(self.config['simTime']), float(self.config['timeStep']),
            float(self.current_step / max(1.0, self.config['simTime'])), float(self.config['carrierFrequency']),
            float(self.config.get('isd', 500.0)), float(self.config['minTxPower']), float(self.config['maxTxPower']),
            float(self.config['basePower']), float(self.config['idlePower']), float(self.config['dropCallThreshold']),
            float(self.config['latencyThreshold']), float(self.config['cpuThreshold']), float(self.config['prbThreshold']),
            float(self.config['trafficLambda']), float(self.config.get('peakHourMultiplier', 1.0))
        ]

        metrics = self.compute_metrics()
        network_features = [
            float(self.total_energy_kwh), float(metrics.get("activeCells", 0)), float(metrics.get("avgDropRate", 0)),
            float(metrics.get("avgLatency", 0)), float(metrics.get("totalTraffic", 0)), float(metrics.get("connectedUEs", 0)),
            float(metrics.get("connectionRate", 0)), float(metrics.get("cpuViolations", 0)), float(metrics.get("prbViolations", 0)),
            float(metrics.get("maxCpuUsage", 0)), float(metrics.get("maxPrbUsage", 0)),
            float(metrics.get("kpiViolations", 0)), float(metrics.get("totalTxPower", 0)), float(metrics.get("avgPowerRatio", 0))
        ]

        # cell features array sized to max_cells
        cell_features = np.zeros(self.max_cells * self.state_dim_per_cell, dtype=np.float32)
        for i, cell in enumerate(self.cells):
            stats = self._get_ue_stats_for_cell(cell.id)
            cell_feats_list = [
                float(cell.txPower), float(cell.energyConsumption), float(cell.cpuUsage), float(cell.prbUsage),
                float(cell.maxCapacity), float(cell.currentLoad), float(cell.currentLoad / (cell.maxCapacity or 1)),
                float(cell.ttt), float(cell.a3Offset), float(len(cell.connectedUEs)),
                float(stats['active_sessions']), float(stats['total_traffic']),
                float(stats['avg_rsrp']), float(stats['min_rsrp']), float(stats['max_rsrp']), float(stats['std_rsrp']),
                float(stats['avg_rsrq']), float(stats['min_rsrq']), float(stats['max_rsrq']), float(stats['std_rsrq']),
                float(stats['avg_sinr']), float(stats['min_sinr']), float(stats['max_sinr']), float(stats['std_sinr']),
                float(getattr(cell, 'power_ratio', 1.0))
            ]
            start_idx = i * self.state_dim_per_cell
            cell_features[start_idx: start_idx + self.state_dim_per_cell] = np.array(cell_feats_list, dtype=np.float32)

        obs = np.concatenate([np.array(sim_features, dtype=np.float32), np.array(network_features, dtype=np.float32), cell_features]).astype(np.float32)
        return obs

    def _get_ue_stats_for_cell(self, cell_id: int) -> Dict[str, float]:
        """Get UE statistics for a cell with improved numerical stability."""
        ue_metrics = [
            (ue.rsrp, ue.rsrq, ue.sinr, ue.trafficDemand, ue.sessionActive)
            for ue in self.ues if ue.servingCell == cell_id
        ]

        if not ue_metrics:
            return self._get_default_ue_stats()

        rsrps, rsrqs, sinrs, traffic, sessions = zip(*ue_metrics)
        return self._compute_safe_stats(rsrps, rsrqs, sinrs, traffic, sessions)

    def _get_default_ue_stats(self) -> Dict[str, float]:
        """Return default UE statistics for empty cells."""
        return {
            'active_sessions': 0.0, 'total_traffic': 0.0,
            'avg_rsrp': -140.0, 'min_rsrp': -140.0, 'max_rsrp': -140.0, 'std_rsrp': 0.0,
            'avg_rsrq': -20.0, 'min_rsrq': -20.0, 'max_rsrq': -20.0, 'std_rsrq': 0.0,
            'avg_sinr': -20.0, 'min_sinr': -20.0, 'max_sinr': -20.0, 'std_sinr': 0.0
        }

    def _compute_safe_stats(self, rsrps: Tuple[float, ...], rsrqs: Tuple[float, ...],
                           sinrs: Tuple[float, ...], traffic: Tuple[float, ...],
                           sessions: Tuple[bool, ...]) -> Dict[str, float]:
        """Compute statistics with NaN handling and numerical stability."""
        def safe_stats(data: Tuple[float, ...], default_val: float) -> Tuple[float, float, float, float]:
            arr: np.ndarray = np.array(data, dtype=np.float32)
            valid_data: np.ndarray = arr[np.isfinite(arr)]
            if valid_data.size == 0:
                return default_val, default_val, default_val, 0.0
            return float(np.mean(valid_data)), float(np.min(valid_data)), float(np.max(valid_data)), float(np.std(valid_data))

        avg_rsrp, min_rsrp, max_rsrp, std_rsrp = safe_stats(rsrps, -140.0)
        avg_rsrq, min_rsrq, max_rsrq, std_rsrq = safe_stats(rsrqs, -20.0)
        avg_sinr, min_sinr, max_sinr, std_sinr = safe_stats(sinrs, -20.0)

        return {
            'active_sessions': float(np.sum(sessions)),
            'total_traffic': float(np.sum(traffic)),
            'avg_rsrp': avg_rsrp, 'min_rsrp': min_rsrp, 'max_rsrp': max_rsrp, 'std_rsrp': std_rsrp,
            'avg_rsrq': avg_rsrq, 'min_rsrq': min_rsrq, 'max_rsrq': max_rsrq, 'std_rsrq': std_rsrq,
            'avg_sinr': avg_sinr, 'min_sinr': min_sinr, 'max_sinr': max_sinr, 'std_sinr': std_sinr
        }

    # ------------------------------
    # Metrics computation (merged)
    # ------------------------------
    def compute_metrics(self) -> Dict[str, Any]:
        if not self.cells:
            return {}
        total_tx_power = sum(c.txPower for c in self.cells)
        power_range = self.config['maxTxPower'] - self.config['minTxPower']
        avg_power_ratio = np.mean([(c.txPower - c.minTxPower) / power_range for c in self.cells]) if power_range > 0 else 0.0
        connected_ues = sum(1 for ue in self.ues if ue.servingCell is not None)
        drop_rates = [c.dropRate for c in self.cells if not np.isnan(c.dropRate)]
        avg_drop = float(np.mean(drop_rates)) if drop_rates else 0.0
        latencies = [c.avgLatency for c in self.cells if not np.isnan(c.avgLatency)]
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        metrics = {
            "totalEnergy": float(sum(c.energyConsumption for c in self.cells)),
            "activeCells": int(sum(1 for c in self.cells if len(c.connectedUEs) > 0)),
            "avgDropRate": avg_drop, "avgLatency": avg_latency,
            "totalTraffic": float(sum(c.currentLoad for c in self.cells)),
            "connectedUEs": int(connected_ues),
            "connectionRate": (connected_ues / max(self.n_ues, 1)) * 100.0,
            "cpuViolations": int(sum(1 for c in self.cells if c.cpuUsage > self.config['cpuThreshold'])),
            "prbViolations": int(sum(1 for c in self.cells if c.prbUsage > self.config['prbThreshold'])),
            "maxCpuUsage": float(max((c.cpuUsage for c in self.cells), default=0.0)),
            "maxPrbUsage": float(max((c.prbUsage for c in self.cells), default=0.0)),
            "totalTxPower": float(total_tx_power),
            "avgPowerRatio": float(avg_power_ratio)
        }
        metrics["kpiViolations"] = int(metrics["avgDropRate"] > self.config['dropCallThreshold']) + int(metrics["avgLatency"] > self.config['latencyThreshold']) + metrics["cpuViolations"] + metrics["prbViolations"]
        return metrics

    # ------------------------------
    # Reset / step methods (merged behavior)
    # ------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.config['seed'] = int(seed)
        np.random.seed(int(self.config.get('seed', 42)))
        self.current_step = 0
        self.total_energy_kwh = 0.0

        self.qos_compliant_steps = 0 
        self._prev_qos_violated = False

        # Setup scenario
        self._setup_scenario()
        active_powers = np.array([c.txPower for c in self.cells])
        self.previous_powers = np.pad(active_powers, (0, self.max_cells - len(active_powers)), 'constant')
        self.neighbor_measurements = np.full((self.n_ues, self.max_neighbors), -1, dtype=neighbor_dtype)

        # run one simulation step (initialization) using default powers (action=None)
        self.ues, self.cells, self.neighbor_measurements = optimized_run_simulation_step(
            self.ues, self.cells, self.neighbor_measurements, self.config, self.time_step_duration, -1, action=None
        )
        # accumulate initial energy (optional)
        instantaneous_power = sum(c.energyConsumption for c in self.cells)
        self.total_energy_kwh += (instantaneous_power / 1000.0) * (self.time_step_duration / 3600.0)

        return self._get_obs(), {}

# ------------------------
# Smooth Reward Computation with Better Balance
# ------------------------

    def compute_smooth_reward(self, metrics):
        """
        Constraint-prioritized reward function.
        
        Key principle: QoS violations (drop rate, latency) MUST be avoided.
        Energy efficiency is only rewarded when QoS is consistently maintained.
        """
        
        # =================================================================
        # 1. CRITICAL QOS METRICS - HIGHEST PRIORITY
        # =================================================================
        avg_drop = metrics.get("avgDropRate", 0.0)
        avg_latency = metrics.get("avgLatency", 0.0)
        
        # Handle invalid values
        avg_drop = max(0.0, avg_drop) if not np.isnan(avg_drop) else 0.0
        avg_latency = max(0.0, avg_latency) if not np.isnan(avg_latency) else 0.0
        
        # Hard thresholds
        drop_threshold = self.config['dropCallThreshold']  # 1%
        latency_threshold = self.config['latencyThreshold']  # 50ms
        
        # Check if constraints are violated
        drop_violated = avg_drop > drop_threshold
        latency_violated = avg_latency > latency_threshold
        
        # Compute violation severity (how far beyond threshold)
        if drop_violated:
            drop_severity = (avg_drop - drop_threshold) / max(drop_threshold, 1e-6)
        else:
            drop_severity = 0.0
        
        if latency_violated:
            latency_severity = (avg_latency - latency_threshold) / max(latency_threshold, 1e-6)
        else:
            latency_severity = 0.0
        
        # QoS scores (smooth functions)
        # When below threshold: score approaches 1.0 as we get better
        # When above threshold: score becomes negative based on severity
        if not drop_violated:
            # Reward being well below threshold
            # 0% drop → 1.0, threshold → 0.5
            drop_score = 0.5 + 0.5 * (1.0 - avg_drop / max(drop_threshold, 1e-6))
        else:
            # Heavy penalty for violations, exponentially growing
            drop_score = -2.0 * (np.exp(drop_severity) - 1.0)
        
        if not latency_violated:
            # 0ms → 1.0, threshold → 0.5
            latency_score = 0.5 + 0.5 * (1.0 - avg_latency / max(latency_threshold, 1e-6))
        else:
            # Heavy penalty for violations
            latency_score = -2.0 * (np.exp(latency_severity) - 1.0)
        
        if not drop_violated and not latency_violated:
            self.qos_compliant_steps += 1
        else:
            self.qos_compliant_steps = 0  # Reset on any violation
        
        # =================================================================
        # 2. CONNECTION RATE - CRITICAL FOR SERVICE
        # =================================================================
        connected_ues = metrics.get("connectedUEs", 0)
        connection_rate = connected_ues / max(self.n_ues, 1)
        
        # Very steep penalty for poor coverage
        # 100% → 1.0, 95% → 0.5, 90% → 0.0, <90% → negative
        if connection_rate >= 0.95:
            connection_score = 0.5 + 0.5 * ((connection_rate - 0.95) / 0.05)
        elif connection_rate >= 0.90:
            connection_score = 0.5 * ((connection_rate - 0.90) / 0.05)
        else:
            # Heavy penalty below 90%
            connection_score = -1.0 * (0.90 - connection_rate) / 0.90
        
        # =================================================================
        # 3. RESOURCE UTILIZATION - PREVENT OVERLOAD
        # =================================================================
        cpu_violations = metrics.get("cpuViolations", 0)
        prb_violations = metrics.get("prbViolations", 0)
        total_violations = cpu_violations + prb_violations
        
        if self.n_cells > 0:
            violation_rate = total_violations / self.n_cells
            # Exponential penalty for resource violations
            resource_score = np.exp(-3 * violation_rate)
        else:
            resource_score = 1.0
        
        # =================================================================
        # 4. ENERGY EFFICIENCY - ONLY REWARDED WHEN QOS IS GOOD
        # =================================================================
        if self.n_cells > 0:
            max_power_per_cell = self.config['basePower'] + 10 ** ((self.config['maxTxPower'] - 30) / 10.0)
            max_possible_power = self.n_cells * max_power_per_cell
            total_current_power = sum(c.energyConsumption for c in self.cells)
            
            energy_ratio = total_current_power / max(max_possible_power, 1e-6)
            energy_efficiency = 1.0 - energy_ratio
            
            # Smooth sigmoid for energy
            energy_score = 1.0 / (1.0 + np.exp(-10 * (energy_efficiency - 0.5)))
        else:
            energy_score = 0.5
            energy_efficiency = 0.0
        
        # CRITICAL: Energy is only rewarded if QoS is maintained
        # Require at least 5 consecutive compliant steps before energy matters
        MIN_COMPLIANT_STEPS = 5
        
        if self.qos_compliant_steps >= MIN_COMPLIANT_STEPS:
            # Gradually increase energy importance with more compliant steps
            # 5 steps → 0.2×, 10 steps → 0.5×, 20 steps → 0.9×, 30+ steps → 1.0×
            energy_multiplier = min(1.0, 0.2 + 0.8 * (self.qos_compliant_steps - MIN_COMPLIANT_STEPS) / 25.0)
            effective_energy_score = energy_score * energy_multiplier
        else:
            # No energy reward until QoS is stable
            effective_energy_score = 0.0
        
        # =================================================================
        # 5. SECONDARY METRICS (lower priority)
        # =================================================================
        
        # Load balancing
        loads = [c.currentLoad / max(c.maxCapacity, 1e-6) for c in self.cells if c.maxCapacity > 0]
        if len(loads) > 1:
            load_variance = np.var(loads)
            load_balance_score = np.exp(-10 * load_variance)
        else:
            load_balance_score = 1.0
        
        # Signal quality
        sinr_values = [ue.sinr for ue in self.ues if not np.isnan(ue.sinr) and ue.sinr > -100]
        if len(sinr_values) > 0:
            avg_sinr = np.mean(sinr_values)
            sinr_score = 1.0 / (1.0 + np.exp(-0.2 * (avg_sinr - 10)))
        else:
            sinr_score = 0.1 if self.n_ues > 0 else 0.5
        
        # Power stability
        active_previous_powers = self.previous_powers[:self.n_cells]
        if hasattr(self, 'previous_powers') and len(active_previous_powers) == len(self.cells):
            power_changes = [abs(c.txPower - active_previous_powers[i]) for i, c in enumerate(self.cells)]
            if power_changes:
                avg_change = np.mean(power_changes)
                max_change = self.config['maxTxPower'] - self.config['minTxPower']
                change_ratio = avg_change / max(max_change, 1e-6)
                stability_score = np.exp(-change_ratio)
            else:
                stability_score = 1.0
        else:
            stability_score = 1.0
        
        # Update previous powers
        current_active_powers = np.array([c.txPower for c in self.cells])
        self.previous_powers = np.pad(current_active_powers, (0, self.max_cells - len(current_active_powers)), 'constant')
        
        # =================================================================
        # 6. CONSTRAINT-PRIORITY WEIGHTING SYSTEM
        # =================================================================
        if not hasattr(self, 'total_episodes'):
            self.total_episodes = 0
        
        ep = self.total_episodes
        
        # Weights are HEAVILY skewed toward QoS constraints
        # Energy only becomes significant after QoS is learned
        
        if ep < 100:
            # Phase 1: Learn to maintain QoS at ANY energy cost
            weights = {
                'drop': 10.0,           # CRITICAL - highest priority
                'latency': 10.0,        # CRITICAL - highest priority
                'connection': 6.0,      # Very important
                'resource': 2.0,        # Important
                'energy': 0.0,          # ZERO - not considered yet
                'load_balance': 0.3,    # Nice to have
                'sinr': 0.2,            # Nice to have
                'stability': 0.1        # Nice to have
            }
        elif ep < 300:
            # Phase 2: Maintain QoS, start considering efficiency
            weights = {
                'drop': 8.0,            # Still critical
                'latency': 8.0,         # Still critical
                'connection': 5.0,      # Very important
                'resource': 2.0,        # Important
                'energy': 1.0,          # Small consideration (but multiplied by compliant steps)
                'load_balance': 0.5,    
                'sinr': 0.3,
                'stability': 0.2
            }
        elif ep < 600:
            # Phase 3: Balance QoS and efficiency
            weights = {
                'drop': 6.0,
                'latency': 6.0,
                'connection': 4.0,
                'resource': 2.0,
                'energy': 3.0,          # Moderate importance (when compliant)
                'load_balance': 0.7,
                'sinr': 0.4,
                'stability': 0.3
            }
        else:
            # Phase 4: Optimize efficiency while maintaining QoS
            weights = {
                'drop': 5.0,            # Still high
                'latency': 5.0,         # Still high
                'connection': 3.0,
                'resource': 2.0,
                'energy': 5.0,          # Equal to QoS (when compliant)
                'load_balance': 0.8,
                'sinr': 0.5,
                'stability': 0.4
            }
        
        # =================================================================
        # 7. COMPOSITE REWARD WITH CONSTRAINT PRIORITY
        # =================================================================
        
        # Base reward from all components
        base_reward = (
            weights['drop'] * drop_score +
            weights['latency'] * latency_score +
            weights['connection'] * connection_score +
            weights['resource'] * resource_score +
            weights['energy'] * effective_energy_score +  # Note: effective_energy_score
            weights['load_balance'] * load_balance_score +
            weights['sinr'] * sinr_score +
            weights['stability'] * stability_score
        )
        
        # =================================================================
        # 8. BONUSES - ONLY FOR SUSTAINED QOS COMPLIANCE
        # =================================================================
        
        # Streak bonus: reward for maintaining QoS over time
        if self.qos_compliant_steps >= 10:
            streak_multiplier = min(2.0, 1.0 + 0.1 * (self.qos_compliant_steps - 10) / 10.0)
            streak_bonus = 1.0 * streak_multiplier
        else:
            streak_bonus = 0.0
        
        # Excellence bonus: all metrics good simultaneously
        if (drop_score > 0.8 and latency_score > 0.8 and 
            connection_score > 0.8 and resource_score > 0.9 and
            self.qos_compliant_steps >= MIN_COMPLIANT_STEPS):
            excellence_bonus = 2.0
            
            # Extra bonus for efficiency during excellence
            if energy_score > 0.7:
                excellence_bonus += 1.0
        else:
            excellence_bonus = 0.0
        
        reward = base_reward + streak_bonus + excellence_bonus
        
        # =================================================================
        # 9. PENALTY MULTIPLIER FOR VIOLATIONS
        # =================================================================
        
        # If BOTH drop and latency are violated, apply severe penalty
        if drop_violated and latency_violated:
            violation_penalty = -5.0 * (drop_severity + latency_severity)
            reward += violation_penalty
        elif drop_violated or latency_violated:
            violation_penalty = -3.0 * max(drop_severity, latency_severity)
            reward += violation_penalty
        else:
            violation_penalty = 0.0
        
        # =================================================================
        # 10. NORMALIZE AND SOFT CLIP
        # =================================================================
        
        # Normalize (max theoretical: ~28 + bonuses ~4 = 32)
        reward = reward / 32.0
        
        # Soft clipping
        reward = np.tanh(reward)
        
        # =================================================================
        # 11. DETAILED LOGGING
        # =================================================================
        return {
            'reward': reward,
            'components': {
                'connection_score': connection_score,
                'drop_score': drop_score,
                'latency_score': latency_score,
                'resource_score': resource_score,
                'energy_score': energy_score,
                'effective_energy_score': effective_energy_score,
                'load_balance_score': load_balance_score,
                'sinr_score': sinr_score,
                'stability_score': stability_score
            },
            'weighted': {
                'w_connection': weights['connection'] * connection_score,
                'w_drop': weights['drop'] * drop_score,
                'w_latency': weights['latency'] * latency_score,
                'w_resource': weights['resource'] * resource_score,
                'w_energy': weights['energy'] * effective_energy_score,
                'w_load_balance': weights['load_balance'] * load_balance_score,
                'w_sinr': weights['sinr'] * sinr_score,
                'w_stability': weights['stability'] * stability_score
            },
            'bonuses': {
                'streak': streak_bonus,
                'excellence': excellence_bonus
            },
            'constraints': {
                'drop_violated': drop_violated,
                'latency_violated': latency_violated,
                'drop_severity': drop_severity,
                'latency_severity': latency_severity,
                'violation_penalty': violation_penalty if 'violation_penalty' in locals() else 0.0,
                'qos_compliant_steps': self.qos_compliant_steps,
                'energy_multiplier': energy_multiplier if self.qos_compliant_steps >= MIN_COMPLIANT_STEPS else 0.0
            },
            'metrics': {
                'connection_rate_pct': connection_rate * 100,
                'connected_ues': connected_ues,
                'total_ues': self.n_ues,
                'avg_drop_rate': avg_drop,
                'avg_latency': avg_latency,
                'total_violations': total_violations,
                'energy_ratio': 1.0 - energy_efficiency if self.n_cells > 0 else 0.0
            }
        }


    # =================================================================
    # MODIFIED STEP FUNCTION
    # =================================================================
    def step(self, action):
        active_action = action[:self.n_cells]
        
        current_powers = np.array([c.txPower for c in self.cells])
        if len(current_powers) != len(self.previous_powers): 
            self.previous_powers = current_powers
        
        self.ues, self.cells, self.neighbor_measurements = optimized_run_simulation_step(
            self.ues, self.cells, self.neighbor_measurements, self.config,
            self.time_step_duration, self.current_step, active_action
        )
        
        metrics = self.compute_metrics()
        
        # Use constraint-priority reward
        reward_info = self.compute_smooth_reward(metrics)
        reward = reward_info['reward']
        
        # Add to metrics for logging
        metrics['reward_components'] = reward_info['components']
        metrics['weighted_components'] = reward_info['weighted']
        metrics['reward_bonuses'] = reward_info['bonuses']
        metrics['constraints'] = reward_info['constraints']
        metrics.update(reward_info['metrics'])
        metrics['normalized_reward'] = reward
        metrics['raw_reward'] = reward * 32.0
        
        # Debug print for first step and when QoS status changes
        if self.current_step == 0 or (hasattr(self, '_prev_qos_violated') and 
            self._prev_qos_violated != (reward_info['constraints']['drop_violated'] or 
                                        reward_info['constraints']['latency_violated'])):
            
            qos_violated = reward_info['constraints']['drop_violated'] or reward_info['constraints']['latency_violated']
            self._prev_qos_violated = qos_violated
            
            # print(f"\n[DEBUG] Step {self.current_step}:")
            # print(f"  QoS Status: {'❌ VIOLATED' if qos_violated else '✅ COMPLIANT'}")
            # print(f"  Drop: {reward_info['metrics']['avg_drop_rate']:.2f}% "
            #     f"(threshold: {self.config['dropCallThreshold']:.2f}%)")
            # print(f"  Latency: {reward_info['metrics']['avg_latency']:.2f}ms "
            #     f"(threshold: {self.config['latencyThreshold']:.2f}ms)")
            # print(f"  Compliant streak: {reward_info['constraints']['qos_compliant_steps']} steps")
            # print(f"  Energy multiplier: {reward_info['constraints']['energy_multiplier']:.2f}")
            # print(f"  Connection: {reward_info['metrics']['connected_ues']}/{reward_info['metrics']['total_ues']} "
            #     f"({reward_info['metrics']['connection_rate_pct']:.1f}%)")
            # print(f"  Reward: {reward:.3f}")
        
        self.current_step += 1
        terminated = self.current_step >= self.max_time_steps
        
        if terminated:
            if not hasattr(self, 'total_episodes'):
                self.total_episodes = 0
            self.total_episodes += 1
            
            # Print episode summary
            print(f"\n[EPISODE {self.total_episodes} COMPLETE]")
            print(f"  Total QoS compliant steps: {reward_info['constraints']['qos_compliant_steps']}/{self.current_step}")
            print(f"  Compliance rate: {100 * reward_info['constraints']['qos_compliant_steps'] / self.current_step:.1f}%")
        
        return self._get_obs(), float(reward), bool(terminated), False, metrics


    # =================================================================
    # UTILITY: Monitor QoS compliance
    # =================================================================
    def monitor_qos_compliance(self, num_episodes=10):
        """
        Track QoS compliance rates during random exploration.
        Helps verify that maintaining QoS is achievable.
        """
        compliance_data = {
            'episode': [],
            'compliant_steps': [],
            'total_steps': [],
            'compliance_rate': [],
            'avg_reward': [],
            'max_streak': []
        }
        
        for ep in range(num_episodes):
            obs, _ = self.reset()
            done = False
            step_count = 0
            compliant_count = 0
            rewards = []
            max_streak = 0
            
            while not done:
                action = np.random.uniform(0.3, 0.8, self.action_space.shape)  # Moderate power
                obs, reward, done, _, info = self.step(action)
                
                step_count += 1
                rewards.append(reward)
                
                if not (info['constraints']['drop_violated'] or info['constraints']['latency_violated']):
                    compliant_count += 1
                
                max_streak = max(max_streak, info['constraints']['qos_compliant_steps'])
            
            compliance_rate = 100 * compliant_count / step_count
            
            compliance_data['episode'].append(ep + 1)
            compliance_data['compliant_steps'].append(compliant_count)
            compliance_data['total_steps'].append(step_count)
            compliance_data['compliance_rate'].append(compliance_rate)
            compliance_data['avg_reward'].append(np.mean(rewards))
            compliance_data['max_streak'].append(max_streak)
            
            print(f"Episode {ep+1}: {compliant_count}/{step_count} compliant ({compliance_rate:.1f}%), "
                f"Max streak: {max_streak}, Avg reward: {np.mean(rewards):.3f}")
        
        print(f"\nOverall: {np.mean(compliance_data['compliance_rate']):.1f}% compliance rate")
        print(f"Average max streak: {np.mean(compliance_data['max_streak']):.1f} steps")
        
        return compliance_data

    # =================================================================
    # UTILITY: Visualize reward components over time
    # =================================================================
    def plot_reward_components(self, num_episodes=5):
        """
        Visualize how different reward components evolve during episodes.
        Useful for debugging and understanding agent behavior.
        """
        import matplotlib.pyplot as plt

        all_data = {
            'connection': [], 'drop': [], 'latency': [], 'resource': [],
            'energy': [], 'load_balance': [], 'sinr': [], 'stability': []
        }
        rewards = []

        for ep in range(num_episodes):
            obs, _ = self.reset()
            done = False

            while not done:
                action = np.random.uniform(0, 1, self.action_space.shape)
                obs, reward, done, _, info = self.step(action)

                rewards.append(reward)
                for key in all_data.keys():
                    all_data[key].append(info['reward_components'][f'{key}_score'])

        # Create subplot for each component
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, (key, values) in enumerate(all_data.items()):
            axes[idx].plot(values, alpha=0.7)
            axes[idx].set_title(f'{key.replace("_", " ").title()} Score')
            axes[idx].set_ylabel('Score')
            axes[idx].set_xlabel('Step')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Good')
            axes[idx].axhline(y=0.5, color='y', linestyle='--', alpha=0.5, label='Fair')
            axes[idx].legend()

        # Plot total reward
        axes[8].plot(rewards, color='red', linewidth=2)
        axes[8].set_title('Total Reward')
        axes[8].set_ylabel('Reward')
        axes[8].set_xlabel('Step')
        axes[8].grid(True, alpha=0.3)
        axes[8].axhline(y=0, color='k', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig('reward_components_analysis.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'reward_components_analysis.png'")
        plt.show()

    # ------------------------------
    # Diagnostics & utilities (copied from uploaded file)
    # ------------------------------
    def diagnose_connection_issue(self):
        print("\n=== CONNECTION DIAGNOSTIC ===")
        print(f"Total UEs: {self.n_ues}")
        print(f"Total Cells: {self.n_cells}")
        print("\nCell Status:")
        for i, cell in enumerate(self.cells):
            print(f"  Cell {i}: TxPower={cell.txPower:.1f}dBm, Connected={len(cell.connectedUEs)}, Load={cell.currentLoad:.1f}/{cell.maxCapacity:.1f}")
        connected_count = sum(1 for ue in self.ues if ue.servingCell is not None)
        no_serving_cell = self.n_ues - connected_count
        print(f"\nUE Status:\n  Connected: {connected_count}\n  No serving cell: {no_serving_cell}")
        print(f"\nSample UE details (first 5):")
        for i, ue in enumerate(self.ues[:5]):
            print(f"  UE {i}: ServingCell={ue.servingCell}, RSRP={ue.rsrp:.1f}, SINR={ue.sinr:.1f}")
        if no_serving_cell == self.n_ues:
            print("\n⚠️  WARNING: NO UEs are connected to any cell!")
            print("   Possible issues: cell power too low, UEs too far, simulation step not running, or action scale issue")
        return {'total_ues': self.n_ues, 'connected_ues': connected_count, 'disconnected_ues': no_serving_cell, 'cells_with_ues': sum(1 for c in self.cells if len(c.connectedUEs) > 0)}

    def test_action_impact(self):
        print("\n=== ACTION IMPACT TEST ===")
        obs, _ = self.reset()
        print("\nTest 1: All cells at minimum power (action=0.0)")
        action_min = np.zeros(self.max_cells)
        obs, reward, done, _, info = self.step(action_min)
        print(f"  Reward: {reward:.3f}")
        print("\nTest 2: All cells at maximum power (action=1.0)")
        obs, _ = self.reset()
        action_max = np.ones(self.max_cells)
        obs, reward, done, _, info = self.step(action_max)
        print(f"  Reward: {reward:.3f}")
        print("\nTest 3: Mid power (action=0.5)")
        obs, _ = self.reset()
        action_mid = np.ones(self.max_cells) * 0.5
        obs, reward, done, _, info = self.step(action_mid)
        print(f"  Reward: {reward:.3f}")

    def analyze_reward_distribution(self, num_episodes: int = 10):
        all_rewards = []
        all_components = {'connection': [], 'drop': [], 'latency': [], 'resource': [], 'energy': [], 'load_balance': [], 'sinr': [], 'stability': []}
        for ep in range(num_episodes):
            obs, _ = self.reset()
            done = False
            ep_rewards = []
            while not done:
                action = np.random.uniform(0, 1, self.action_space.shape)
                obs, reward, done, truncated, info = self.step(action)
                ep_rewards.append(reward)
                if 'reward_components' in info:
                    for key in all_components.keys():
                        score_key = f"{key}_score"
                        all_components[key].append(info['reward_components'].get(score_key, 0))
                if done or truncated:
                    break
            all_rewards.extend(ep_rewards)
            print(f"Episode {ep+1}: Mean Reward = {np.mean(ep_rewards):.3f}, Min = {np.min(ep_rewards):.3f}, Max = {np.max(ep_rewards):.3f}")
        print(f"\nOverall Statistics:\nMean Reward: {np.mean(all_rewards):.3f} Std Reward: {np.std(all_rewards):.3f} Min: {np.min(all_rewards):.3f} Max: {np.max(all_rewards):.3f}")
        return all_rewards, all_components

# ------------------------------
# Quick usage example
# ------------------------------
if __name__ == "__main__":
    # small config
    cfg = {
        'timeStep': 1.0,
        'simTime': 10,
        'carrierFrequency': 3.5e9,
        'minTxPower': 30.0,
        'maxTxPower': 46.0,
        'basePower': 1000.0,
        'idlePower': 250.0,
        'dropCallThreshold': 1.0,
        'latencyThreshold': 50.0,
        'cpuThreshold': 80.0,
        'prbThreshold': 80.0,
        'numSites': 3,
        'numUEs': 30,
        'isd': 200.0,
        'deploymentScenario': 'indoor_hotspot',
        'seed': 42
    }
    env = FiveGEnv(cfg, max_cells=12)
    obs, _ = env.reset()
    print("obs.shape:", obs.shape)
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print("step reward:", reward)

    # Example of running a diagnostic utility
    # env.diagnose_connection_issue()

    # Example of visualizing reward components (requires matplotlib)
    # try:
    #     import matplotlib.pyplot as plt
    #     env.plot_reward_components(num_episodes=2)
    # except ImportError:
    #     print("\nMatplotlib not found. Skipping reward component plot.")
    #     print("Install it with: pip install matplotlib")