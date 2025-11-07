# fiveg_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional

from fiveg_objects import Cell, UE, SimulationParams
from simulation_logic import (
    update_ue_mobility, update_signal_measurements, check_handover_events,
    handle_disconnected_ues, update_traffic_generation,
    update_ue_drop_events, update_cell_resource_usage
)
from scenario_creator import load_scenario_config, create_sites, configure_cells, initialize_ues

def run_simulation_step(
    ues: List[UE], cells: List[Cell], sim_params: SimulationParams,
    time_step: float, current_time: float, seed: int, action: Optional[np.ndarray] = None
) -> Tuple[List[UE], List[Cell]]:
    """Runs one step of the simulation logic, matching the MATLAB runLoop sequence."""
    if action is not None:
        for i, cell in enumerate(cells):
            if i < len(action):
                power_range = cell.max_tx_power - cell.min_tx_power
                cell.tx_power = cell.min_tx_power + action[i] * power_range

    ues = update_ue_mobility(ues, time_step, current_time, sim_params, seed)
    ues, cells = update_traffic_generation(ues, cells, sim_params)
    ues = update_signal_measurements(ues, cells, sim_params, current_time, seed)
    ues = handle_disconnected_ues(ues, cells, sim_params, time_step)
    _, ues = check_handover_events(ues, cells, current_time, sim_params, seed)
    ues, cells = update_ue_drop_events(ues, cells)
    cells = update_cell_resource_usage(cells, ues)
    return ues, cells

class FiveGEnv(gym.Env):
    """
    A Gymnasium environment for 5G network simulation, matching the MATLAB model.
    """
    def __init__(self, env_config: Dict[str, Any], max_cells: int = 57):
        super().__init__()
        scenario_name = env_config.get('deploymentScenario', 'dense_urban')
        self.sim_params = load_scenario_config(scenario_name)
        for key, value in env_config.items():
            if hasattr(self.sim_params, key): setattr(self.sim_params, key, value)

        self.max_cells = int(max_cells)
        self.state_dim_per_cell = 12 
        self.seed = int(env_config.get('seed', 42))

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.max_cells,), dtype=np.float32)
        state_dim = 17 + 14 + (self.max_cells * self.state_dim_per_cell)
        # The observation space is defined with raw, unnormalized bounds.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        self.cells: List[Cell] = []
        self.ues: List[UE] = []
        self.n_cells = 0
        self.n_ues = 0
        self.current_step = 0

    def _setup_scenario(self, seed: int):
        """Initializes the simulation scenario."""
        sites = create_sites(self.sim_params, seed)
        self.cells = configure_cells(sites, self.sim_params)
        self.ues = initialize_ues(self.sim_params, sites, seed)
        self.n_cells = len(self.cells)
        self.n_ues = len(self.ues)
        if self.n_cells > self.max_cells:
            raise ValueError(f"Scenario has {self.n_cells} cells, but env configured for max {self.max_cells}.")

    def _get_obs(self) -> np.ndarray:
        """
        Constructs the observation vector, precisely matching MATLAB's createRLState.
        """
        p = self.sim_params
        # 1. Simulation Features (17)
        sim_features = np.array([
            float(self.n_cells), float(self.n_ues), float(p.simTime), float(p.timeStep),
            float(self.current_step / max(1, p.total_steps)), float(p.carrierFrequency),
            float(p.isd), float(p.minTxPower), float(p.maxTxPower), float(p.basePower),
            float(p.idlePower), float(p.dropCallThreshold), float(p.latencyThreshold),
            float(p.cpuThreshold), float(p.prbThreshold), float(p.trafficLambda),
            float(p.peakHourMultiplier)
        ], dtype=np.float32)

        # 2. Network-wide Metrics (14)
        metrics = self.compute_metrics()
        network_features = np.array([
            metrics["total_energy"], metrics["active_cells"], metrics["avg_drop_rate"],
            metrics["avg_latency"], metrics["total_traffic"], metrics["connected_ues"],
            metrics["connection_rate"], metrics["cpu_violations"], metrics["prb_violations"],
            metrics["max_cpu_usage"], metrics["max_prb_usage"], metrics["kpi_violations"],
            metrics["total_tx_power"], metrics["avg_power_ratio"]
        ], dtype=np.float32)

        # --- CRITICAL FIX: Reconstruct cell features to match StateNormalizer's expected interleaved format ---
        # StateNormalizer expects: [c1_f1, c2_f1, ..., cn_f1, c1_f2, c2_f2, ..., cn_f2, ...]
        # Previous implementation was: [c1_f1, c1_f2, ..., c2_f1, c2_f2, ...] (blocked)
        
        # 3. Per-Cell Features (12 per cell)
        cell_features_matrix = np.zeros((self.max_cells, self.state_dim_per_cell), dtype=np.float32)
        ue_stats_by_cell = self._get_ue_stats_by_cell()

        for i, cell in enumerate(self.cells):
            stats = ue_stats_by_cell.get(cell.id, {'avg_rsrp': -140, 'avg_rsrq': -20, 'avg_sinr': -20, 'total_traffic': 0})
            load_ratio = cell.current_load / max(1, cell.max_capacity)
            
            # This order must match the `cell_bounds` dictionary in state_normalizer.py
            cell_features_matrix[i, :] = [
                cell.cpu_usage, cell.prb_usage, cell.current_load, cell.max_capacity,
                float(len(cell.connected_ues)), cell.tx_power, cell.energy_consumption,
                stats['avg_rsrp'], stats['avg_rsrq'], stats['avg_sinr'],
                stats['total_traffic'], load_ratio
            ]

        # Now, flatten the matrix in column-major order ('F' for Fortran-style)
        # This interleaves the features correctly.
        cell_features = cell_features_matrix.flatten(order='F')
        
        # The final observation vector
        obs = np.concatenate([sim_features, network_features, cell_features])
        obs[np.isnan(obs) | np.isinf(obs)] = 0 # Handle potential NaN/inf values
        return obs

    def _get_ue_stats_by_cell(self) -> Dict[int, Dict[str, float]]:
        """Aggregates UE stats per cell, needed for the observation vector."""
        stats = {}
        for cell in self.cells: stats[cell.id] = {'rsrps': [], 'rsrqs': [], 'sinrs': [], 'traffics': []}
        for ue in self.ues:
            if ue.serving_cell in stats:
                if np.isfinite(ue.rsrp): stats[ue.serving_cell]['rsrps'].append(ue.rsrp)
                if np.isfinite(ue.rsrq): stats[ue.serving_cell]['rsrqs'].append(ue.rsrq)
                if np.isfinite(ue.sinr): stats[ue.serving_cell]['sinrs'].append(ue.sinr)
                if ue.session_active: stats[ue.serving_cell]['traffics'].append(ue.traffic_demand)
        
        agg_stats = {}
        for cell_id, data in stats.items():
            agg_stats[cell_id] = {
                'avg_rsrp': np.mean(data['rsrps']) if data['rsrps'] else -140.0,
                'avg_rsrq': np.mean(data['rsrqs']) if data['rsrqs'] else -20.0,
                'avg_sinr': np.mean(data['sinrs']) if data['sinrs'] else -20.0,
                'total_traffic': np.sum(data['traffics'])
            }
        return agg_stats

    def compute_metrics(self) -> Dict[str, float]:
        """Computes network-wide metrics, matching MATLAB's computeEnergySavingMetrics."""
        if not self.cells: return {k: 0.0 for k in ["total_energy", "active_cells", "avg_drop_rate", "avg_latency", "total_traffic", "connected_ues", "connection_rate", "cpu_violations", "prb_violations", "max_cpu_usage", "max_prb_usage", "kpi_violations", "total_tx_power", "avg_power_ratio"]}

        connected_ues_count = sum(len(c.connected_ues) for c in self.cells)
        active_cells_count = sum(1 for c in self.cells if len(c.connected_ues) > 0)
        power_range = max(1e-6, self.sim_params.maxTxPower - self.sim_params.minTxPower)
        
        metrics = {
            "total_energy": sum(c.energy_consumption for c in self.cells),
            "active_cells": float(active_cells_count),
            "avg_drop_rate": np.mean([c.drop_rate for c in self.cells if len(c.connected_ues) > 0] or [0]),
            "avg_latency": np.mean([c.avg_latency for c in self.cells if len(c.connected_ues) > 0] or [0]),
            "total_traffic": sum(c.current_load for c in self.cells),
            "connected_ues": float(connected_ues_count),
            "connection_rate": connected_ues_count / max(1, self.n_ues),
            "cpu_violations": float(sum(1 for c in self.cells if c.cpu_usage > self.sim_params.cpuThreshold)),
            "prb_violations": float(sum(1 for c in self.cells if c.prb_usage > self.sim_params.prbThreshold)),
            "max_cpu_usage": max((c.cpu_usage for c in self.cells), default=0.0),
            "max_prb_usage": max((c.prb_usage for c in self.cells), default=0.0),
            "total_tx_power": sum(c.tx_power for c in self.cells),
            "avg_power_ratio": np.mean([(c.tx_power - c.min_tx_power) / power_range for c in self.cells])
        }
        metrics["kpi_violations"] = float(metrics["avg_drop_rate"] > self.sim_params.dropCallThreshold) + float(metrics["avg_latency"] > self.sim_params.latencyThreshold) + metrics["cpu_violations"] + metrics["prb_violations"]
        return metrics

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.seed = seed if seed is not None else self.seed
        np.random.seed(self.seed)
        self.current_step = 0
        self._setup_scenario(self.seed)
        
        self.ues, self.cells = run_simulation_step(self.ues, self.cells, self.sim_params, self.sim_params.timeStep, -1, self.seed)
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        active_action = action[:self.n_cells]
        self.ues, self.cells = run_simulation_step(self.ues, self.cells, self.sim_params, self.sim_params.timeStep, self.current_step, self.seed, active_action)
        
        metrics = self.compute_metrics()
        self.current_step += 1
        terminated = self.current_step >= self.sim_params.total_steps
        
        return self._get_obs(), 0.0, terminated, False, metrics
