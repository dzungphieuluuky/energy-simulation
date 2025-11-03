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
        """Enhanced reset that properly initializes tracking variables"""
        result = super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.total_energy_kwh = 0.0
        self._setup_scenario()
        
        # Initialize tracking variables
        self.qos_compliant_steps = 0
        self.current_episode_reward = 0.0
        current_powers = np.array([c.txPower for c in self.cells])
        self.previous_powers = np.pad(current_powers, (0, self.max_cells - len(current_powers)), 'constant')
        
        if not hasattr(self, 'total_episodes'):
            self.total_episodes = 0
        
        self.neighbor_measurements = np.full(
            (self.n_ues, self.max_neighbors), -1, dtype=neighbor_dtype
        )
        
        return self._get_obs(), {}

# ------------------------
# Smooth Reward Computation with Better Balance
# ------------------------

    def compute_smooth_reward(self, metrics):
        """
        Hard-constraint reward function.
        
        CRITICAL DESIGN:
        1. QoS violations (drop rate > 1%, latency > 50ms) result in ONLY penalties
        2. Connection rate below 95% results in ONLY penalties
        3. Positive rewards are ONLY possible when ALL constraints are satisfied
        4. Energy efficiency and other metrics only matter when constraints are met
        """
        
        # =================================================================
        # 1. EVALUATE ALL CONSTRAINTS (HARD REQUIREMENTS)
        # =================================================================
        avg_drop = max(0.0, metrics.get("avgDropRate", 0.0) if not np.isnan(metrics.get("avgDropRate", 0.0)) else 0.0)
        avg_latency = max(0.0, metrics.get("avgLatency", 0.0) if not np.isnan(metrics.get("avgLatency", 0.0)) else 0.0)
        connected_ues = metrics.get("connectedUEs", 0)
        connection_rate = connected_ues / max(self.n_ues, 1)
        
        # Define thresholds
        drop_threshold = self.config['dropCallThreshold']  # 1%
        latency_threshold = self.config['latencyThreshold']  # 50ms
        connection_threshold = 0.95  # 95% must be connected
        
        # Check constraint violations
        drop_violated = avg_drop > drop_threshold
        latency_violated = avg_latency > latency_threshold
        connection_violated = connection_rate < connection_threshold
        
        # ANY violation means constraints are not satisfied
        constraints_satisfied = not (drop_violated or latency_violated or connection_violated)
        
        # Track consecutive compliant steps
        if not hasattr(self, 'qos_compliant_steps'):
            self.qos_compliant_steps = 0
        
        if constraints_satisfied:
            self.qos_compliant_steps += 1
        else:
            self.qos_compliant_steps = 0
        
        # =================================================================
        # 2. COMPUTE CONSTRAINT PENALTY (if violated)
        # =================================================================
        if not constraints_satisfied:
            # Compute individual penalties based on severity
            penalties = []
            
            if drop_violated:
                severity = (avg_drop - drop_threshold) / max(drop_threshold, 1e-6)
                drop_penalty = -5.0 * (1.0 + severity)  # Base -5, grows with severity
                penalties.append(drop_penalty)
            
            if latency_violated:
                severity = (avg_latency - latency_threshold) / max(latency_threshold, 1e-6)
                latency_penalty = -5.0 * (1.0 + severity)
                penalties.append(latency_penalty)
            
            if connection_violated:
                gap = connection_threshold - connection_rate
                connection_penalty = -10.0 * gap  # Very steep penalty
                penalties.append(connection_penalty)
            
            # Total penalty is sum of all violations
            total_penalty = sum(penalties)
            
            # Return ONLY penalty, no other rewards
            return {
                'reward': np.tanh(total_penalty / 10.0),  # Normalize to ~[-1, 0]
                'constraints_satisfied': False,
                'constraint_penalties': {
                    'drop': drop_penalty if drop_violated else 0.0,
                    'latency': latency_penalty if latency_violated else 0.0,
                    'connection': connection_penalty if connection_violated else 0.0,
                    'total': total_penalty
                },
                'components': {
                    'drop_violated': drop_violated,
                    'latency_violated': latency_violated,
                    'connection_violated': connection_violated
                },
                'metrics': {
                    'avg_drop_rate': avg_drop,
                    'avg_latency': avg_latency,
                    'connection_rate_pct': connection_rate * 100,
                    'connected_ues': connected_ues,
                    'total_ues': self.n_ues,
                    'qos_compliant_steps': self.qos_compliant_steps
                }
            }
        
        # =================================================================
        # 3. CONSTRAINTS ARE SATISFIED - COMPUTE POSITIVE REWARDS
        # =================================================================
        
        # 3.1 Base reward for maintaining constraints
        base_constraint_reward = 1.0  # Fixed reward for being compliant
        
        # 3.2 QoS Quality (how far below thresholds)
        # Better margin = higher reward
        drop_margin = (drop_threshold - avg_drop) / max(drop_threshold, 1e-6)
        drop_quality = np.clip(drop_margin, 0, 1)  # 0 at threshold, 1 at perfect
        
        latency_margin = (latency_threshold - avg_latency) / max(latency_threshold, 1e-6)
        latency_quality = np.clip(latency_margin, 0, 1)
        
        connection_margin = (connection_rate - connection_threshold) / (1.0 - connection_threshold + 1e-6)
        connection_quality = np.clip(connection_margin, 0, 1)
        
        qos_quality_reward = 0.5 * (drop_quality + latency_quality + connection_quality) / 3.0
        
        # 3.3 Energy Efficiency (main optimization objective)
        if self.n_cells > 0:
            max_power_per_cell = self.config['basePower'] + 10 ** ((self.config['maxTxPower'] - 30) / 10.0)
            max_possible_power = self.n_cells * max_power_per_cell
            total_current_power = sum(c.energyConsumption for c in self.cells)
            
            energy_ratio = total_current_power / max(max_possible_power, 1e-6)
            energy_efficiency = 1.0 - energy_ratio
            
            # Energy score only kicks in after sustained compliance
            MIN_COMPLIANT_STEPS = 10
            if self.qos_compliant_steps >= MIN_COMPLIANT_STEPS:
                # Gradually unlock: 10 steps→25%, 20→50%, 40+→100%
                unlock_ratio = min(1.0, (self.qos_compliant_steps - MIN_COMPLIANT_STEPS) / 30.0)
                energy_reward = 2.0 * energy_efficiency * unlock_ratio
            else:
                energy_reward = 0.0
        else:
            energy_reward = 0.0
            energy_efficiency = 0.0
        
        # 3.4 Resource Efficiency
        cpu_violations = metrics.get("cpuViolations", 0)
        prb_violations = metrics.get("prbViolations", 0)
        
        if self.n_cells > 0:
            violation_rate = (cpu_violations + prb_violations) / self.n_cells
            resource_reward = 0.3 * (1.0 - np.clip(violation_rate, 0, 1))
        else:
            resource_reward = 0.3
        
        # 3.5 Load Balancing
        loads = [c.currentLoad / max(c.maxCapacity, 1e-6) for c in self.cells if c.maxCapacity > 0]
        if len(loads) > 1:
            load_std = np.std(loads)
            load_balance_reward = 0.2 * np.exp(-10 * load_std)
        else:
            load_balance_reward = 0.2
        
        # 3.6 Signal Quality
        sinr_values = [ue.sinr for ue in self.ues if not np.isnan(ue.sinr) and ue.sinr > -100]
        if len(sinr_values) > 0:
            avg_sinr = np.mean(sinr_values)
            sinr_normalized = (avg_sinr + 10) / 40  # -10dB to 30dB → 0 to 1
            sinr_reward = 0.2 * np.clip(sinr_normalized, 0, 1)
        else:
            sinr_reward = 0.0
        
        # 3.7 Power Stability
        active_previous_powers = self.previous_powers[:self.n_cells]
        if hasattr(self, 'previous_powers') and len(active_previous_powers) == len(self.cells):
            power_changes = [abs(c.txPower - active_previous_powers[i]) for i, c in enumerate(self.cells)]
            if power_changes:
                avg_change = np.mean(power_changes)
                max_change = self.config['maxTxPower'] - self.config['minTxPower']
                change_ratio = avg_change / max(max_change, 1e-6)
                stability_reward = 0.1 * (1.0 - np.clip(change_ratio, 0, 1))
            else:
                stability_reward = 0.1
        else:
            stability_reward = 0.1
        
        # Update previous powers
        current_active_powers = np.array([c.txPower for c in self.cells])
        self.previous_powers = np.pad(current_active_powers, (0, self.max_cells - len(current_active_powers)), 'constant')
        
        # =================================================================
        # 4. COMBINE REWARDS (all components are positive)
        # =================================================================
        total_reward = (
            base_constraint_reward +      # 1.0
            qos_quality_reward +           # up to 0.5
            energy_reward +                # up to 2.0 (when unlocked)
            resource_reward +              # up to 0.3
            load_balance_reward +          # up to 0.2
            sinr_reward +                  # up to 0.2
            stability_reward               # up to 0.1
        )
        # Theoretical max: 1.0 + 0.5 + 2.0 + 0.3 + 0.2 + 0.2 + 0.1 = 4.3
        
        # =================================================================
        # 5. BONUS REWARDS FOR EXCELLENCE
        # =================================================================
        
        # Sustained compliance bonus
        if self.qos_compliant_steps >= 50:
            streak_bonus = 0.5 * min(1.0, self.qos_compliant_steps / 100.0)
        else:
            streak_bonus = 0.0
        
        # High efficiency bonus (only after 20+ compliant steps)
        if self.qos_compliant_steps >= 20 and energy_efficiency > 0.7:
            efficiency_bonus = 0.5 * (energy_efficiency - 0.7) / 0.3
        else:
            efficiency_bonus = 0.0
        
        # Perfect performance bonus
        if (drop_quality > 0.9 and latency_quality > 0.9 and 
            connection_quality > 0.9 and self.qos_compliant_steps >= 30):
            perfection_bonus = 1.0
        else:
            perfection_bonus = 0.0
        
        total_reward += streak_bonus + efficiency_bonus + perfection_bonus
        # Max with bonuses: 4.3 + 0.5 + 0.5 + 1.0 = 6.3
        
        # =================================================================
        # 6. NORMALIZE TO [0, 1] RANGE
        # =================================================================
        normalized_reward = total_reward / 6.5
        normalized_reward = np.clip(normalized_reward, 0, 1)
        
        # =================================================================
        # 7. RETURN DETAILED INFORMATION
        # =================================================================
        return {
            'reward': normalized_reward,
            'constraints_satisfied': True,
            'constraint_penalties': {
                'drop': 0.0,
                'latency': 0.0,
                'connection': 0.0,
                'total': 0.0
            },
            'components': {
                'base_constraint_reward': base_constraint_reward,
                'qos_quality_reward': qos_quality_reward,
                'energy_reward': energy_reward,
                'resource_reward': resource_reward,
                'load_balance_reward': load_balance_reward,
                'sinr_reward': sinr_reward,
                'stability_reward': stability_reward,
                'streak_bonus': streak_bonus,
                'efficiency_bonus': efficiency_bonus,
                'perfection_bonus': perfection_bonus
            },
            'quality_scores': {
                'drop_quality': drop_quality,
                'latency_quality': latency_quality,
                'connection_quality': connection_quality,
                'energy_efficiency': energy_efficiency
            },
            'metrics': {
                'avg_drop_rate': avg_drop,
                'avg_latency': avg_latency,
                'connection_rate_pct': connection_rate * 100,
                'connected_ues': connected_ues,
                'total_ues': self.n_ues,
                'qos_compliant_steps': self.qos_compliant_steps,
                'energy_unlock_ratio': min(1.0, max(0.0, (self.qos_compliant_steps - 10) / 30.0))
            }
        }


    # =================================================================
    # MODIFIED STEP FUNCTION
    # =================================================================
    def step(self, action):
        active_action = action[:self.n_cells]
        
        current_powers = np.array([c.txPower for c in self.cells])
        if not hasattr(self, 'previous_powers') or len(current_powers) != len(self.previous_powers): 
            self.previous_powers = np.pad(current_powers, (0, self.max_cells - len(current_powers)), 'constant')
        
        self.ues, self.cells, self.neighbor_measurements = optimized_run_simulation_step(
            self.ues, self.cells, self.neighbor_measurements, self.config,
            self.time_step_duration, self.current_step, active_action
        )
        
        metrics = self.compute_metrics()
        reward_info = self.compute_smooth_reward(metrics)
        reward = reward_info['reward']
        
        # Accumulate episode reward
        self.current_episode_reward += reward

        # Comprehensive logging
        metrics['reward_info'] = reward_info
        metrics['normalized_reward'] = reward
        metrics['constraints_satisfied'] = reward_info['constraints_satisfied']
        
        # Periodic detailed logging
        if self.current_step % 50 == 0 or self.current_step == 0:
            status = "✅ COMPLIANT" if reward_info['constraints_satisfied'] else "❌ VIOLATED"
            print(f"\n[Step {self.current_step}] {status}")
            print(f"  Drop: {reward_info['metrics']['avg_drop_rate']:.3f}% (threshold: {self.config['dropCallThreshold']:.1f}%)")
            print(f"  Latency: {reward_info['metrics']['avg_latency']:.2f}ms (threshold: {self.config['latencyThreshold']:.0f}ms)")
            print(f"  Connection: {reward_info['metrics']['connection_rate_pct']:.1f}% (threshold: 95.0%)")
            print(f"  Compliant streak: {reward_info['metrics']['qos_compliant_steps']}")
            
            if reward_info['constraints_satisfied']:
                print(f"  Energy unlocked: {100*reward_info['metrics']['energy_unlock_ratio']:.0f}%")
                print(f"  Reward components:")
                for k, v in reward_info['components'].items():
                    if v > 0:
                        print(f"    {k}: {v:.3f}")
            else:
                print(f"  Penalties:")
                for k, v in reward_info['constraint_penalties'].items():
                    if v < 0:
                        print(f"    {k}: {v:.3f}")
            print(f"  Total reward: {reward:.4f}")
        
        self.current_step += 1
        terminated = self.current_step >= self.max_time_steps
        
        if terminated:
            if not hasattr(self, 'total_episodes'):
                self.total_episodes = 0
            self.total_episodes += 1
            
            compliance_rate = 100 * reward_info['metrics']['qos_compliant_steps'] / self.current_step
            
            print(f"\n{'='*60}")
            print(f"[EPISODE {self.total_episodes} COMPLETE]")
            print(f"  Total steps: {self.current_step}")
            print(f"  Total episode reward: {self.current_episode_reward:.4f}")
            print(f"  QoS compliant steps: {reward_info['metrics']['qos_compliant_steps']}")
            print(f"  Compliance rate: {compliance_rate:.1f}%")
            print(f"  Constraints satisfied: {reward_info['constraints_satisfied']}")
            print(f"{'='*60}\n")
        
        info = reward_info
        if terminated:
            info["episode_rewards"] = self.current_episode_reward
        
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

    def analyze_constraint_feasibility(self, num_episodes=10, power_levels=[0.3, 0.5, 0.7, 1.0]):
        """
        Test if constraints are achievable at different power levels.
        This helps identify if your thresholds are realistic.
        """
        print("\n" + "="*70)
        print("CONSTRAINT FEASIBILITY ANALYSIS")
        print("="*70)
        
        results = {}
        
        for power in power_levels:
            print(f"\nTesting with constant power = {power:.1f} (action = {power})")
            print("-" * 70)
            
            compliant_steps = []
            avg_drops = []
            avg_latencies = []
            connection_rates = []
            
            for ep in range(num_episodes):
                obs, _ = self.reset()
                done = False
                ep_compliant = 0
                ep_steps = 0
                
                while not done:
                    action = np.ones(self.action_space.shape) * power
                    obs, reward, done, _, info = self.step(action)
                    
                    ep_steps += 1
                    if info['constraints_satisfied']:
                        ep_compliant += 1
                    
                    avg_drops.append(info['reward_info']['metrics']['avg_drop_rate'])
                    avg_latencies.append(info['reward_info']['metrics']['avg_latency'])
                    connection_rates.append(info['reward_info']['metrics']['connection_rate_pct'])
                
                compliant_steps.append(100 * ep_compliant / ep_steps)
            
            results[power] = {
                'compliance_rate': np.mean(compliant_steps),
                'avg_drop': np.mean(avg_drops),
                'avg_latency': np.mean(avg_latencies),
                'avg_connection': np.mean(connection_rates)
            }
            
            print(f"  Compliance rate: {results[power]['compliance_rate']:.1f}%")
            print(f"  Avg drop rate: {results[power]['avg_drop']:.3f}% (threshold: {self.config['dropCallThreshold']:.1f}%)")
            print(f"  Avg latency: {results[power]['avg_latency']:.1f}ms (threshold: {self.config['latencyThreshold']:.0f}ms)")
            print(f"  Avg connection: {results[power]['avg_connection']:.1f}% (threshold: 95.0%)")
            
            if results[power]['compliance_rate'] > 80:
                print(f"  ✅ Constraints are ACHIEVABLE at this power level")
            elif results[power]['compliance_rate'] > 50:
                print(f"  ⚠️  Constraints are SOMETIMES achievable")
            else:
                print(f"  ❌ Constraints are DIFFICULT to achieve")
        
        print("\n" + "="*70)
        print("RECOMMENDATION:")
        best_power = max(results.keys(), key=lambda k: results[k]['compliance_rate'])
        print(f"  Best power level tested: {best_power:.1f}")
        print(f"  Achieved {results[best_power]['compliance_rate']:.1f}% compliance")
        print("="*70 + "\n")
        
        return results


    def get_reward_statistics(self, num_episodes=20):
        """
        Collect reward statistics to understand the reward distribution.
        """
        all_rewards = []
        compliant_rewards = []
        violated_rewards = []
        compliance_rates = []
        
        for ep in range(num_episodes):
            obs, _ = self.reset()
            done = False
            ep_rewards = []
            ep_compliant = 0
            ep_steps = 0
            
            while not done:
                action = np.random.uniform(0.3, 0.8, self.action_space.shape)
                obs, reward, done, _, info = self.step(action)
                
                ep_steps += 1
                ep_rewards.append(reward)
                all_rewards.append(reward)
                
                if info['constraints_satisfied']:
                    ep_compliant += 1
                    compliant_rewards.append(reward)
                else:
                    violated_rewards.append(reward)
            
            compliance_rates.append(100 * ep_compliant / ep_steps)
        
        print("\n" + "="*70)
        print("REWARD STATISTICS")
        print("="*70)
        print(f"\nOverall ({len(all_rewards)} steps):")
        print(f"  Mean: {np.mean(all_rewards):.4f}")
        print(f"  Std:  {np.std(all_rewards):.4f}")
        print(f"  Min:  {np.min(all_rewards):.4f}")
        print(f"  Max:  {np.max(all_rewards):.4f}")
        
        if compliant_rewards:
            print(f"\nWhen constraints satisfied ({len(compliant_rewards)} steps):")
            print(f"  Mean: {np.mean(compliant_rewards):.4f}")
            print(f"  Std:  {np.std(compliant_rewards):.4f}")
            print(f"  Range: [{np.min(compliant_rewards):.4f}, {np.max(compliant_rewards):.4f}]")
        
        if violated_rewards:
            print(f"\nWhen constraints violated ({len(violated_rewards)} steps):")
            print(f"  Mean: {np.mean(violated_rewards):.4f}")
            print(f"  Std:  {np.std(violated_rewards):.4f}")
            print(f"  Range: [{np.min(violated_rewards):.4f}, {np.max(violated_rewards):.4f}]")
        
        print(f"\nAverage compliance rate: {np.mean(compliance_rates):.1f}%")
        print("="*70 + "\n")
        
        return {
            'all': all_rewards,
            'compliant': compliant_rewards,
            'violated': violated_rewards,
            'compliance_rates': compliance_rates
        }
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