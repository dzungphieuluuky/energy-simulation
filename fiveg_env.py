# fiveg_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simulation_logic as sim
import scenario_creator as sc
from numba_utils import neighbor_dtype

class FiveGEnv(gym.Env):
    """A Gymnasium environment fully aligned with the canonical MATLAB implementation."""
    def __init__(self, env_config, max_cells=57):
        super().__init__()
        
        self.config = env_config
        self.time_step_duration = self.config['timeStep']
        self.max_time_steps = self.config['simTime']
        self.max_neighbors = 8
        self.max_cells = max_cells
        
        self.state_dim_per_cell = 25 
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.max_cells,), dtype=np.float32)
        state_dim = 17 + 14 + (self.max_cells * self.state_dim_per_cell)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        self.cells, self.ues, self.n_cells, self.n_ues = [], [], 0, 0
        self.current_step, self.total_energy_kwh = 0, 0.0
        self.reward_weights = self.get_adaptive_weights(0.0)
        self.previous_powers = np.zeros(self.max_cells)

    def _setup_scenario(self):
        """
        Creates the simulation scenario with the corrected data flow,
        mirroring the MATLAB `simulate5GNetwork` function.
        """
        # 1. Create the site layout (list of dictionaries)
        sites = sc.create_layout(self.config)
        
        # 2. Configure cells using the site layout
        self.cells = sc.configure_cells(self.config, sites)
        
        # 3. Initialize UEs using the site layout
        self.ues = sc.initialize_ues(self.config, sites)
        
        self.n_cells, self.n_ues = len(self.cells), len(self.ues)
        if self.n_cells > self.max_cells:
            raise ValueError(f"Scenario has {self.n_cells} cells, but env configured for max {self.max_cells}.")


    def _get_obs(self):
        """Constructs the observation vector, perfectly mirroring createRLState.m."""
        sim_features = [
            self.n_cells, self.n_ues, self.config['simTime'], self.config['timeStep'],
            self.current_step / self.config['simTime'], self.config['carrierFrequency'],
            self.config.get('isd', 500), self.config['minTxPower'], self.config['maxTxPower'],
            self.config['basePower'], self.config['idlePower'], self.config['dropCallThreshold'],
            self.config['latencyThreshold'], self.config['cpuThreshold'], self.config['prbThreshold'],
            self.config['trafficLambda'], self.config.get('peakHourMultiplier', 1.0)
        ]
        
        metrics = self.compute_metrics()
        network_features = [
            self.total_energy_kwh, metrics.get("activeCells", 0), metrics.get("avgDropRate", 0),
            metrics.get("avgLatency", 0), metrics.get("totalTraffic", 0), metrics.get("connectedUEs", 0),
            metrics.get("connectionRate", 0), metrics.get("cpuViolations", 0), metrics.get("prbViolations", 0),
            metrics.get("maxCpuUsage", 0), metrics.get("maxPrbUsage", 0),
            metrics.get("kpiViolations", 0), metrics.get("totalTxPower", 0), metrics.get("avgPowerRatio", 0)
        ]
        
        cell_features = np.zeros(self.max_cells * self.state_dim_per_cell, dtype=np.float32)
        for i, cell in enumerate(self.cells):
            stats = self._get_ue_stats_for_cell(cell.id)
            cell_feats_list = [
                cell.txPower, cell.energyConsumption, cell.cpuUsage, cell.prbUsage,
                cell.maxCapacity, cell.currentLoad, cell.currentLoad / (cell.maxCapacity or 1),
                cell.ttt, cell.a3Offset, len(cell.connectedUEs),
                stats['active_sessions'], stats['total_traffic'],
                stats['avg_rsrp'], stats['min_rsrp'], stats['max_rsrp'], stats['std_rsrp'],
                stats['avg_rsrq'], stats['min_rsrq'], stats['max_rsrq'], stats['std_rsrq'],
                stats['avg_sinr'], stats['min_sinr'], stats['max_sinr'], stats['std_sinr'],
                getattr(cell, 'power_ratio', 1.0)
            ]
            start_idx = i * self.state_dim_per_cell
            cell_features[start_idx : start_idx + self.state_dim_per_cell] = cell_feats_list
            
        return np.concatenate([sim_features, network_features, cell_features]).astype(np.float32)

    def _get_ue_stats_for_cell(self, cell_id):
        """Computes detailed UE statistics, matching createRLState.m."""
        ue_metrics = [
            (ue.rsrp, ue.rsrq, ue.sinr, ue.trafficDemand, ue.sessionActive)
            for ue in self.ues if ue.servingCell == cell_id
        ]
        if not ue_metrics:
            nan_stats = {'avg': -140, 'min': -140, 'max': -140, 'std': 0}
            return {'active_sessions': 0, 'total_traffic': 0, 'avg_rsrp': -140, 'min_rsrp': -140, 'max_rsrp': -140, 'std_rsrp': 0,
                    'avg_rsrq': -20, 'min_rsrq': -20, 'max_rsrq': -20, 'std_rsrq': 0,
                    'avg_sinr': -20, 'min_sinr': -20, 'max_sinr': -20, 'std_sinr': 0}
        
        rsrps, rsrqs, sinrs, traffic, sessions = zip(*ue_metrics)
        
        def safe_stats(data, default_val):
            arr = np.array(data)[np.isfinite(data)]
            if arr.size == 0: return default_val, default_val, default_val, 0
            return np.mean(arr), np.min(arr), np.max(arr), np.std(arr)

        avg_rsrp, min_rsrp, max_rsrp, std_rsrp = safe_stats(rsrps, -140)
        avg_rsrq, min_rsrq, max_rsrq, std_rsrq = safe_stats(rsrqs, -20)
        avg_sinr, min_sinr, max_sinr, std_sinr = safe_stats(sinrs, -20)
        
        return {
            'active_sessions': np.sum(sessions), 'total_traffic': np.sum(traffic),
            'avg_rsrp': avg_rsrp, 'min_rsrp': min_rsrp, 'max_rsrp': max_rsrp, 'std_rsrp': std_rsrp,
            'avg_rsrq': avg_rsrq, 'min_rsrq': min_rsrq, 'max_rsrq': max_rsrq, 'std_rsrq': std_rsrq,
            'avg_sinr': avg_sinr, 'min_sinr': min_sinr, 'max_sinr': max_sinr, 'std_sinr': std_sinr,
        }

    def compute_metrics(self):
        """Computes network-wide metrics for logging."""
        if not self.cells: return {}
        
        total_tx_power = sum(c.txPower for c in self.cells)
        power_range = self.config['maxTxPower'] - self.config['minTxPower']
        avg_power_ratio = np.mean([(c.txPower - c.minTxPower) / power_range for c in self.cells]) if power_range > 0 else 0
        connected_ues = sum(1 for ue in self.ues if ue.servingCell is not None)

        drop_rates = [c.dropRate for c in self.cells if not np.isnan(c.dropRate)]
        avg_drop = np.mean(drop_rates) if drop_rates else 0.0
        
        latencies = [c.avgLatency for c in self.cells if not np.isnan(c.avgLatency)]
        avg_latency = np.mean(latencies) if latencies else 0.0
        
        metrics = {
            "totalEnergy": sum(c.energyConsumption for c in self.cells),
            "activeCells": sum(1 for c in self.cells if len(c.connectedUEs) > 0),
            "avgDropRate": avg_drop, "avgLatency": avg_latency,
            "totalTraffic": sum(c.currentLoad for c in self.cells),
            "connectedUEs": connected_ues,
            "connectionRate": (connected_ues / self.n_ues) * 100 if self.n_ues > 0 else 0,
            "cpuViolations": sum(1 for c in self.cells if c.cpuUsage > self.config['cpuThreshold']),
            "prbViolations": sum(1 for c in self.cells if c.prbUsage > self.config['prbThreshold']),
            "maxCpuUsage": max((c.cpuUsage for c in self.cells), default=0),
            "maxPrbUsage": max((c.prbUsage for c in self.cells), default=0),
            "totalTxPower": total_tx_power,
            "avgPowerRatio": avg_power_ratio
        }
        metrics["kpiViolations"] = int(avg_drop > self.config['dropCallThreshold']) + \
                                   int(avg_latency > self.config['latencyThreshold']) + \
                                   metrics["cpuViolations"] + metrics["prbViolations"]
        return metrics
    
    def set_reward_weights(self, training_progress: float):
        self.reward_weights = self.get_adaptive_weights(training_progress)

    def reset(self, *, seed=None, options=None):
        """Resets the environment and runs a dummy step to ensure initial UE connection."""
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        self.current_step, self.total_energy_kwh = 0, 0.0
        
        # 1. Set up the scenario (places cells and UEs)
        self._setup_scenario()
        self.previous_powers = np.array([c.txPower for c in self.cells])
        self.neighbor_measurements = np.full((self.n_ues, self.max_neighbors), -1, dtype=neighbor_dtype)
        
        self.ues, self.cells, self.neighbor_measurements = sim.run_simulation_step(
            self.ues, self.cells, self.neighbor_measurements, self.config,
            self.time_step_duration,
            -1, # Use a dummy time for the initialization step
            action=None # Pass None to use the default powers
        )
        
        # 3. Return the first valid observation
        return self._get_obs(), {}

    def step(self, action):
        active_action = action[:self.n_cells]
        self.ues, self.cells, self.neighbor_measurements = sim.run_simulation_step(
            self.ues, self.cells, self.neighbor_measurements, self.config,
            self.time_step_duration, self.current_step, active_action
        )
        
        # Now that the simulation has evolved to the next state (t+1),
        # get the observation for this new state.
        obs = self._get_obs()

        current_powers = np.array([c.txPower for c in self.cells])
        if len(current_powers) != len(self.previous_powers): 
            self.previous_powers = current_powers
        
        # === DIAGNOSTIC: Check action values ===
        if self.current_step == 0:
            print(f"[DEBUG] First step action range: [{active_action.min():.3f}, {active_action.max():.3f}]")
            print(f"[DEBUG] Number of cells: {self.n_cells}, Number of UEs: {self.n_ues}")
        
        self.ues, self.cells, self.neighbor_measurements = sim.run_simulation_step(
            self.ues, self.cells, self.neighbor_measurements, self.config,
            self.time_step_duration, self.current_step, active_action
        )
        
        # === DIAGNOSTIC: Check simulation results ===
        if self.current_step == 0:
            print(f"[DEBUG] After sim step:")
            print(f"  - Cells with connected UEs: {sum(1 for c in self.cells if len(c.connectedUEs) > 0)}/{self.n_cells}")
            print(f"  - Total connected UEs: {sum(len(c.connectedUEs) for c in self.cells)}/{self.n_ues}")
            print(f"  - Cell powers: {[f'{c.txPower:.1f}' for c in self.cells[:5]]}")
        
        metrics = self.compute_metrics()
        
        # ==================================================================
        # REWARD COMPONENTS WITH SAFETY CHECKS
        # ==================================================================
        
        # 1. ENERGY EFFICIENCY
        if self.n_cells > 0:
            max_power_per_cell = self.config['basePower'] + 10**((self.config['maxTxPower'] - 30) / 10)
            max_possible_power = self.n_cells * max_power_per_cell
            total_current_power = sum(c.energyConsumption for c in self.cells)
            energy_efficiency = 1.0 - (total_current_power / max(max_possible_power, 1e-6))
            energy_efficiency = np.clip(energy_efficiency, 0, 1)
        else:
            energy_efficiency = 0.0
        
        # 2. QOS METRICS
        avg_drop = metrics.get("avgDropRate", 0)
        avg_latency = metrics.get("avgLatency", 0)
        
        if np.isnan(avg_drop) or avg_drop < 0: avg_drop = 0.0
        if np.isnan(avg_latency) or avg_latency < 0: avg_latency = 0.0
        
        # Drop rate score
        drop_threshold = self.config['dropCallThreshold']
        if avg_drop <= drop_threshold:
            drop_score = 1.0 - (avg_drop / max(drop_threshold, 1e-6))
        else:
            excess = (avg_drop - drop_threshold) / max(drop_threshold, 1e-6)
            drop_score = -min(excess, 1.0)  # Cap at -1.0
        
        # Latency score
        latency_threshold = self.config['latencyThreshold']
        if avg_latency <= latency_threshold:
            latency_score = 1.0 - (avg_latency / max(latency_threshold, 1e-6))
        else:
            excess = (avg_latency - latency_threshold) / max(latency_threshold, 1e-6)
            latency_score = -min(excess, 1.0)  # Cap at -1.0
        
        # 3. CONNECTION RATE - MODIFIED TO BE LESS PUNISHING
        connected_ues = metrics.get("connectedUEs", 0)
        connection_rate = connected_ues / max(self.n_ues, 1)
        
        # === DIAGNOSTIC ===
        if self.current_step == 0:
            print(f"[DEBUG] Connection rate: {connection_rate:.2%} ({connected_ues}/{self.n_ues})")
        
        # NEW: More gradual penalty
        if connection_rate >= 0.98:
            connection_score = 1.0
        elif connection_rate >= 0.90:
            # Linear: 90% to 98% maps to 0.5 to 1.0
            connection_score = 0.5 + 0.5 * (connection_rate - 0.90) / 0.08
        elif connection_rate >= 0.70:
            # Linear: 70% to 90% maps to 0.0 to 0.5
            connection_score = 0.5 * (connection_rate - 0.70) / 0.20
        elif connection_rate >= 0.50:
            # Linear: 50% to 70% maps to -0.5 to 0.0
            connection_score = -0.5 * (0.70 - connection_rate) / 0.20
        else:
            # Below 50%: stronger penalty but capped
            connection_score = -0.5 - min(0.50 - connection_rate, 0.5)
        
        # 4. RESOURCE UTILIZATION
        cpu_violations = metrics.get("cpuViolations", 0)
        prb_violations = metrics.get("prbViolations", 0)
        total_violations = cpu_violations + prb_violations
        
        if self.n_cells > 0:
            violation_rate = total_violations / self.n_cells
            resource_score = 1.0 - np.clip(violation_rate, 0, 1)
        else:
            resource_score = 1.0
        
        # Load balance
        loads = [c.currentLoad / max(c.maxCapacity, 1e-6) for c in self.cells if c.maxCapacity > 0]
        if len(loads) > 1:
            load_std = np.std(loads)
            load_balance_score = np.clip(1.0 - (load_std / 0.3), 0, 1)
        else:
            load_balance_score = 1.0
        
        # 5. SIGNAL QUALITY
        sinr_values = [ue.sinr for ue in self.ues if not np.isnan(ue.sinr) and ue.sinr > -100]
        if len(sinr_values) > 0:
            avg_sinr = np.mean(sinr_values)
            sinr_score = np.clip((avg_sinr + 10) / 40, 0, 1)
        else:
            # If no SINR measurements, check if UEs exist but aren't connected
            sinr_score = 0.0 if self.n_ues > 0 else 0.5
        
        # 6. POWER STABILITY
        if hasattr(self, 'previous_powers') and len(self.previous_powers) == len(self.cells):
            power_changes = [abs(c.txPower - self.previous_powers[i]) for i, c in enumerate(self.cells)]
            if power_changes:
                avg_change = np.mean(power_changes)
                max_change = self.config['maxTxPower'] - self.config['minTxPower']
                stability_score = 1.0 - np.clip(avg_change / max(max_change, 1e-6), 0, 1)
            else:
                stability_score = 1.0
        else:
            stability_score = 1.0
        
        self.previous_powers = [c.txPower for c in self.cells]
        
        # ==================================================================
        # ADAPTIVE WEIGHTS - MORE BALANCED
        # ==================================================================
        
        if not hasattr(self, 'total_episodes'):
            self.total_episodes = 0
        
        # Reduce connection weight to prevent it from dominating
        if self.total_episodes < 100:
            weights = {
                'connection': 3.0,     # Reduced from 5.0
                'drop': 2.0,
                'latency': 1.5,
                'resource': 1.5,
                'energy': 1.0,         # Increased from 0.5
                'load_balance': 0.5,
                'sinr': 0.5,
                'stability': 0.2
            }
        elif self.total_episodes < 500:
            weights = {
                'connection': 2.5,
                'drop': 2.5,
                'latency': 2.0,
                'resource': 1.5,
                'energy': 2.0,
                'load_balance': 0.7,
                'sinr': 0.5,
                'stability': 0.3
            }
        else:
            weights = {
                'connection': 2.0,
                'drop': 2.5,
                'latency': 2.0,
                'resource': 1.5,
                'energy': 3.0,
                'load_balance': 0.8,
                'sinr': 0.5,
                'stability': 0.4
            }
        
        # Compute reward
        reward = (
            weights['connection'] * connection_score +
            weights['drop'] * drop_score +
            weights['latency'] * latency_score +
            weights['resource'] * resource_score +
            weights['energy'] * energy_efficiency +
            weights['load_balance'] * load_balance_score +
            weights['sinr'] * sinr_score +
            weights['stability'] * stability_score
        )
        
        # Small step penalty
        step_penalty = -0.01
        reward += step_penalty
        
        # ==================================================================
        # BONUS REWARDS - MORE ACHIEVABLE
        # ==================================================================
        
        # Good performance bonus (easier to achieve)
        if (connection_score >= 0.7 and drop_score >= 0.7 and 
            latency_score >= 0.7 and resource_score >= 0.8):
            reward += 1.0
        
        # Excellent performance bonus
        if (connection_score >= 0.95 and drop_score >= 0.9 and 
            latency_score >= 0.9 and resource_score >= 0.95):
            reward += 2.0
        
        # Energy efficiency bonus (while maintaining minimum QoS)
        if energy_efficiency >= 0.6 and connection_score >= 0.7:
            reward += 0.5
        
        # ==================================================================
        # NORMALIZATION
        # ==================================================================
        
        # Sum of weights ≈ 13, plus bonuses ≈ 3.5
        # Normalize to approximately [-1, +1]
        reward = reward / 16.0
        reward = np.clip(reward, -2.0, 2.0)
        
        # ==================================================================
        # DETAILED LOGGING
        # ==================================================================
        metrics['reward_components'] = {
            'connection_score': connection_score,
            'drop_score': drop_score,
            'latency_score': latency_score,
            'resource_score': resource_score,
            'energy_efficiency': energy_efficiency,
            'load_balance_score': load_balance_score,
            'sinr_score': sinr_score,
            'stability_score': stability_score,
            'raw_reward': reward * 16.0,
            'normalized_reward': reward,
            'connection_rate_pct': connection_rate * 100,
            'connected_ues': connected_ues,
            'total_ues': self.n_ues,
            'avg_drop_rate': avg_drop,
            'avg_latency': avg_latency,
            'total_violations': total_violations,
        }
        
        metrics['weighted_components'] = {
            'w_connection': weights['connection'] * connection_score,
            'w_drop': weights['drop'] * drop_score,
            'w_latency': weights['latency'] * latency_score,
            'w_resource': weights['resource'] * resource_score,
            'w_energy': weights['energy'] * energy_efficiency,
            'w_load_balance': weights['load_balance'] * load_balance_score,
            'w_sinr': weights['sinr'] * sinr_score,
            'w_stability': weights['stability'] * stability_score
        }
        
        # === DIAGNOSTIC: Print first step details ===
        if self.current_step == 0:
            print(f"[DEBUG] Reward breakdown:")
            for k, v in metrics['weighted_components'].items():
                print(f"  {k}: {v:.3f}")
            print(f"  Total normalized reward: {reward:.3f}")
        
        self.current_step += 1
        terminated = self.current_step >= self.max_time_steps
        
        if terminated:
            self.total_episodes += 1
        
        return self._get_obs(), reward, terminated, False, metrics


    # ==================================================================
    # DIAGNOSTIC HELPER: Check why UEs aren't connecting
    # ==================================================================

    def diagnose_connection_issue(self):
        """
        Call this after a step to understand why UEs aren't connecting.
        """
        print("\n=== CONNECTION DIAGNOSTIC ===")
        print(f"Total UEs: {self.n_ues}")
        print(f"Total Cells: {self.n_cells}")
        
        # Check cell status
        print("\nCell Status:")
        for i, cell in enumerate(self.cells):
            print(f"  Cell {i}: TxPower={cell.txPower:.1f}dBm, "
                f"Connected={len(cell.connectedUEs)}, "
                f"Load={cell.currentLoad:.1f}/{cell.maxCapacity:.1f}")
        
        # Check UE status
        connected_count = 0
        no_serving_cell = 0
        for ue in self.ues:
            if ue.servingCell is not None:
                connected_count += 1
            else:
                no_serving_cell += 1
        
        print(f"\nUE Status:")
        print(f"  Connected: {connected_count}")
        print(f"  No serving cell: {no_serving_cell}")
        
        # Sample a few UEs
        print(f"\nSample UE details (first 5):")
        for i, ue in enumerate(self.ues[:5]):
            print(f"  UE {i}: ServingCell={ue.servingCell}, "
                f"RSRP={ue.rsrp:.1f}dBm, SINR={ue.sinr:.1f}dB")
        
        # Check if there's a simulation issue
        if no_serving_cell == self.n_ues:
            print("\n⚠️  WARNING: NO UEs are connected to any cell!")
            print("   Possible issues:")
            print("   1. Cell power is too low (check if action is being applied)")
            print("   2. UEs are too far from cells")
            print("   3. Simulation step is not running properly")
            print("   4. Action normalization issue (check action scale)")
        
        return {
            'total_ues': self.n_ues,
            'connected_ues': connected_count,
            'disconnected_ues': no_serving_cell,
            'cells_with_ues': sum(1 for c in self.cells if len(c.connectedUEs) > 0)
        }


    # ==================================================================
    # HELPER: Action scale test
    # ==================================================================

    def test_action_impact(self):
        """
        Test if actions actually affect the simulation.
        """
        print("\n=== ACTION IMPACT TEST ===")
        
        obs, _ = self.reset()
        
        # Test 1: Minimum power
        print("\nTest 1: All cells at minimum power (action=0.0)")
        action_min = np.zeros(self.max_cells)
        obs, reward, done, _, info = self.step(action_min)
        print(f"  Connected UEs: {info['reward_components']['connected_ues']}/{info['reward_components']['total_ues']}")
        print(f"  Reward: {reward:.3f}")
        
        # Test 2: Maximum power
        print("\nTest 2: All cells at maximum power (action=1.0)")
        obs, _ = self.reset()
        action_max = np.ones(self.max_cells)
        obs, reward, done, _, info = self.step(action_max)
        print(f"  Connected UEs: {info['reward_components']['connected_ues']}/{info['reward_components']['total_ues']}")
        print(f"  Reward: {reward:.3f}")
        
        # Test 3: Mid power
        print("\nTest 3: All cells at mid power (action=0.5)")
        obs, _ = self.reset()
        action_mid = np.ones(self.max_cells) * 0.5
        obs, reward, done, _, info = self.step(action_mid)
        print(f"  Connected UEs: {info['reward_components']['connected_ues']}/{info['reward_components']['total_ues']}")
        print(f"  Reward: {reward:.3f}")



    # ==================================================================
    # ADDITIONAL METHOD: Adaptive Weight Scheduling
    # ==================================================================

    def get_adaptive_weights(self, training_progress):
        """
        Adjust reward weights based on training progress (0.0 to 1.0).
        Start with focus on coverage, gradually shift to energy efficiency.
        
        Args:
            training_progress: Float between 0.0 and 1.0 representing training progress
        
        Returns:
            Dictionary of reward component weights
        """
        training_progress = np.clip(training_progress, 0.0, 1.0)
        
        if training_progress < 0.3:
            # Early training: Focus on coverage and QoS
            return {
                'energy': 0.5,
                'drop': 4.0,
                'latency': 2.0,
                'connection': 5.0,
                'load_balance': 0.3,
                'violation': 3.0,
                'sinr': 0.2,
                'smoothness': 0.1
            }
        elif training_progress < 0.7:
            # Mid training: Balance all objectives
            return {
                'energy': 1.5,
                'drop': 3.0,
                'latency': 1.5,
                'connection': 4.0,
                'load_balance': 0.5,
                'violation': 2.0,
                'sinr': 0.3,
                'smoothness': 0.2
            }
        else:
            # Late training: Optimize energy while maintaining QoS
            return {
                'energy': 2.5,
                'drop': 2.5,
                'latency': 1.2,
                'connection': 3.5,
                'load_balance': 0.6,
                'violation': 1.5,
                'sinr': 0.4,
                'smoothness': 0.3
            }
        
    def analyze_reward_distribution(self, num_episodes=10):
        """
        Run random policy for a few episodes to understand reward scale.
        Call this before training to verify reward ranges.
        """
        import pandas as pd
        
        all_rewards = []
        all_components = {
            'connection': [], 'drop': [], 'latency': [], 
            'resource': [], 'energy': [], 'load_balance': [],
            'sinr': [], 'stability': []
        }
        
        for ep in range(num_episodes):
            obs, _ = self.reset()
            done = False
            ep_rewards = []
            
            while not done:
                # Random action
                action = np.random.uniform(0, 1, self.action_space.shape)
                obs, reward, done, truncated, info = self.step(action)
                ep_rewards.append(reward)
                
                # Collect components
                if 'reward_components' in info:
                    for key in all_components.keys():
                        score_key = f"{key}_score" if key != 'energy' else 'energy_efficiency'
                        all_components[key].append(info['reward_components'].get(score_key, 0))
            
            all_rewards.extend(ep_rewards)
            print(f"Episode {ep+1}: Mean Reward = {np.mean(ep_rewards):.3f}, "
                f"Min = {np.min(ep_rewards):.3f}, Max = {np.max(ep_rewards):.3f}")
        
        print(f"\nOverall Statistics:")
        print(f"Mean Reward: {np.mean(all_rewards):.3f}")
        print(f"Std Reward: {np.std(all_rewards):.3f}")
        print(f"Min Reward: {np.min(all_rewards):.3f}")
        print(f"Max Reward: {np.max(all_rewards):.3f}")
        
        print(f"\nComponent Statistics:")
        for key, values in all_components.items():
            if values:
                print(f"{key:15s}: mean={np.mean(values):6.3f}, std={np.std(values):6.3f}, "
                    f"min={np.min(values):6.3f}, max={np.max(values):6.3f}")
        
        return all_rewards, all_components