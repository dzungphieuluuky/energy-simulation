"""
Core simulation logic matching MATLAB functions.
Handles UE mobility, measurements, handovers, resource usage.
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from fiveg_objects import UE, Cell, SimulationParams, NeighborMeasurement, HandoverEvent
from numba_utils import (
    evaluate_handover_success_prob, calculate_drop_probability,
    calculate_cell_cpu_usage, calculate_cell_prb_usage,
    calculate_cell_energy_consumption, calculate_theoretical_drop_rate,
    calculate_cell_latency, process_measurements_batch,
    calculate_rsrp_batch
)

# --- Main Simulation Step Functions ---

def update_ue_mobility(ues: List[UE], time_step: float, current_time: float, sim_params: SimulationParams, seed: int) -> List[UE]:
    """Update all UE positions based on mobility patterns, matching updateUEMobility."""
    for ue in ues:
        seed_val = abs((ue.rng_seed + int(math.floor(current_time * 1000)))) % (2**32 - 1)
        np.random.seed(seed_val)
        
        ue.step_counter += 1
        
        # Select and apply mobility handler
        _get_mobility_handler(ue.mobility_pattern)(ue, time_step, current_time)
        
        # Enforce boundaries for the current deployment scenario
        _get_boundary_handler(sim_params.deploymentScenario)(ue, sim_params)
        
        ue.direction %= (2 * math.pi)
    return ues

def update_signal_measurements(ues: List[UE], cells: List[Cell], sim_params: SimulationParams, current_time: float, seed: int) -> List[UE]:
    """Update RSRP/RSRQ/SINR measurements for all UEs, matching updateSignalMeasurements."""
    if not ues or not cells: return ues

    ue_positions = np.array([[ue.x, ue.y] for ue in ues], dtype=np.float64)
    cell_positions = np.array([[c.x, c.y] for c in cells], dtype=np.float64)
    tx_powers = np.array([c.tx_power for c in cells], dtype=np.float64)
    min_tx_powers = np.array([c.min_tx_power for c in cells], dtype=np.float64)
    ue_ids = np.array([ue.id for ue in ues], dtype=np.int32)
    frequency = cells[0].frequency if cells else 3.5e9

    rsrp_matrix = calculate_rsrp_batch(ue_positions, cell_positions, tx_powers, frequency, ue_ids, current_time, seed, min_tx_powers)
    rsrq_matrix, sinr_matrix = process_measurements_batch(rsrp_matrix, tx_powers, min_tx_powers, sim_params.rsrpMeasurementThreshold)

    for i, ue in enumerate(ues):
        measurements = []
        for j, cell in enumerate(cells):
            rsrp, rsrq, sinr = rsrp_matrix[i, j], rsrq_matrix[i, j], sinr_matrix[i, j]
            if not np.isnan(rsrp) and not np.isnan(sinr):
                measurements.append(NeighborMeasurement(cell.id, rsrp, rsrq, sinr))
        
        ue.neighbor_measurements = measurements
        if not measurements:
            ue.serving_cell, ue.rsrp, ue.rsrq, ue.sinr = None, np.nan, np.nan, np.nan
        else:
            serving_meas = next((m for m in measurements if m.cell_id == ue.serving_cell), None)
            if serving_meas:
                ue.rsrp, ue.rsrq, ue.sinr = serving_meas.rsrp, serving_meas.rsrq, serving_meas.sinr
            else: # If UE was connected to a cell it can no longer hear
                ue.serving_cell, ue.rsrp, ue.rsrq, ue.sinr = None, np.nan, np.nan, np.nan
    return ues

def check_handover_events(ues: List[UE], cells: List[Cell], current_time: float, sim_params: SimulationParams, seed: int) -> Tuple[List[HandoverEvent], List[UE]]:
    """Check and execute handover events, matching checkHandoverEvents."""
    handover_events = []
    for ue in ues:
        serving_cell = next((c for c in cells if c.id == ue.serving_cell), None)
        if not serving_cell or np.isnan(ue.rsrp): continue
        
        # Emergency Handover due to poor signal
        if ue.rsrp < sim_params.rsrpServingThreshold:
            best_cell_info = _find_best_cell_for_connection(ue, sim_params.rsrpTargetThreshold)
            if best_cell_info:
                ue.serving_cell, ue.rsrp, ue.rsrq, ue.sinr = best_cell_info
            continue
        
        # A3 Handover logic
        best_neighbor = _find_best_neighbor(ue)
        if (best_neighbor and best_neighbor.rsrp > (ue.rsrp + serving_cell.a3_offset) and best_neighbor.rsrp >= sim_params.rsrpTargetThreshold):
            if ue.ho_timer == 0: ue.ho_timer = current_time
            if (current_time - ue.ho_timer) >= serving_cell.ttt:
                target_cell = next((c for c in cells if c.id == best_neighbor.cell_id), None)
                if not target_cell: continue
                
                ho_prob = evaluate_handover_success_prob(serving_cell.tx_power, serving_cell.min_tx_power, target_cell.tx_power, target_cell.min_tx_power, best_neighbor.rsrp, best_neighbor.sinr, target_cell.cpu_usage, target_cell.prb_usage)
                ho_success = np.random.random() < ho_prob
                
                event = HandoverEvent(ue.id, serving_cell.id, best_neighbor.cell_id, ue.rsrp, best_neighbor.rsrp, ue.rsrq, best_neighbor.rsrq, ue.sinr, best_neighbor.sinr, serving_cell.a3_offset, serving_cell.ttt, ho_success, current_time)
                handover_events.append(event); ue.handover_history.append(event)
                
                if ho_success:
                    ue.serving_cell, ue.rsrp, ue.rsrq, ue.sinr = best_neighbor.cell_id, best_neighbor.rsrp, best_neighbor.rsrq, best_neighbor.sinr
                ue.ho_timer = 0
        else:
            ue.ho_timer = 0
    return handover_events, ues

def handle_disconnected_ues(ues: List[UE], cells: List[Cell], sim_params: SimulationParams, time_step: float) -> List[UE]:
    """Handle disconnected UEs and timers, matching handleDisconnectedUEs."""
    disconnection_timeout, connection_timeout, hysteresis = 5.0, 2.0, 3.0
    for ue in ues:
        if ue.serving_cell is not None:
            if not np.isnan(ue.rsrp) and ue.rsrp < (sim_params.rsrpServingThreshold - hysteresis):
                if ue.disconnection_timer == 0: ue.disconnection_timer = disconnection_timeout
                ue.disconnection_timer -= time_step
                if ue.disconnection_timer <= 0:
                    ue.is_dropped, ue.serving_cell, ue.session_active = True, None, False
                    ue.rsrp, ue.rsrq, ue.sinr, ue.traffic_demand = np.nan, np.nan, np.nan, 0
                    ue.drop_count += 1; ue.disconnection_timer = 0
            else:
                ue.disconnection_timer = 0
        else: # UE is not connected
            best_cell_info = _find_best_cell_for_connection(ue, sim_params.rsrpServingThreshold + hysteresis)
            if best_cell_info:
                if ue.connection_timer == 0: ue.connection_timer = connection_timeout
                ue.connection_timer -= time_step
                if ue.connection_timer <= 0:
                    current_best = _find_best_cell_for_connection(ue, sim_params.rsrpServingThreshold + hysteresis)
                    if current_best and current_best[0] == best_cell_info[0]:
                        ue.serving_cell, ue.rsrp, ue.rsrq, ue.sinr, ue.is_dropped = *best_cell_info, False
                    ue.connection_timer = 0
            else:
                ue.connection_timer = 0
    return ues

def update_traffic_generation(ues: List[UE], cells: List[Cell], sim_params: SimulationParams) -> Tuple[List[UE], List[Cell]]:
    """Generate Poisson traffic for UEs, matching updateTrafficGeneration."""
    lambda_val = sim_params.trafficLambda * sim_params.peakHourMultiplier / max(1, len(ues))
    for cell in cells: cell.current_load, cell.connected_ues = 0.0, []
    
    for ue in ues:
        if ue.serving_cell is not None and not ue.is_dropped:
            demand = float(np.random.poisson(lambda_val))
            ue.traffic_demand = demand
            ue.session_active = demand > 0
            serving_cell = next((c for c in cells if c.id == ue.serving_cell), None)
            if serving_cell:
                serving_cell.current_load += demand
                serving_cell.connected_ues.append(ue.id)
        else:
            ue.traffic_demand, ue.session_active = 0.0, False
    return ues, cells

def update_ue_drop_events(ues: List[UE], cells: List[Cell]) -> Tuple[List[UE], List[Cell]]:
    """Handle UE drop events based on conditions, matching updateUEDropEvents."""
    for ue in ues:
        if ue.serving_cell is not None and ue.session_active and not ue.is_dropped:
            serving_cell = next((c for c in cells if c.id == ue.serving_cell), None)
            if serving_cell:
                drop_prob = calculate_drop_probability(serving_cell.tx_power, serving_cell.min_tx_power, serving_cell.max_tx_power, ue.sinr, ue.rsrp, serving_cell.cpu_usage, serving_cell.prb_usage, serving_cell.current_load, serving_cell.max_capacity)
                if np.random.random() < drop_prob:
                    serving_cell.actual_drop_count += 1; serving_cell.total_drop_events += 1
                    ue.is_dropped, ue.serving_cell, ue.session_active = True, None, False
                    ue.rsrp, ue.rsrq, ue.sinr, ue.traffic_demand = np.nan, np.nan, np.nan, 0
                    ue.drop_count += 1
    return ues, cells

def update_cell_resource_usage(cells: List[Cell], ues: List[UE]) -> List[Cell]:
    """Update CPU, PRB, energy, and performance metrics for all cells."""
    for cell in cells:
        connected_ues = [ue for ue in ues if ue.serving_cell == cell.id and not ue.is_dropped]
        num_ues = len(connected_ues)
        load_ratio = min(1.0, cell.current_load / max(1.0, cell.max_capacity))
        power_range = max(1e-6, cell.max_tx_power - cell.min_tx_power)
        power_ratio = (cell.tx_power - cell.min_tx_power) / power_range
        
        cell.cpu_usage = calculate_cell_cpu_usage(power_ratio, num_ues, load_ratio)
        cell.prb_usage = calculate_cell_prb_usage(num_ues, load_ratio)
        cell.energy_consumption = calculate_cell_energy_consumption(cell.base_energy_consumption, cell.idle_energy_consumption, cell.tx_power, num_ues, load_ratio)
        
        valid_sinrs = [ue.sinr for ue in connected_ues if np.isfinite(ue.sinr)]
        cell.avg_sinr = np.mean(valid_sinrs) if valid_sinrs else -20.0
        
        cell.total_connection_time += num_ues
        actual_drop_rate = (cell.actual_drop_count / cell.total_connection_time) * 100 if cell.total_connection_time > 0 else 0.0
        theoretical_drop_rate = calculate_theoretical_drop_rate(0.1, cell.tx_power, cell.min_tx_power, cell.max_tx_power, cell.cpu_usage, cell.prb_usage, cell.avg_sinr, len(valid_sinrs))
        
        cell.drop_rate = actual_drop_rate if cell.total_connection_time > 10 else theoretical_drop_rate
        cell.theoretical_drop_rate = theoretical_drop_rate
        
        if cell.total_connection_time > 1000: # Decay old stats
            cell.actual_drop_count = int(cell.actual_drop_count * 0.9)
            cell.total_connection_time = int(cell.total_connection_time * 0.9)
        
        cell.avg_latency = calculate_cell_latency(10.0, load_ratio, num_ues, cell.tx_power, cell.min_tx_power, cell.max_tx_power, power_ratio)
    return cells

# --- Helper functions ---
def _find_best_cell_for_connection(ue: UE, rsrp_threshold: float) -> Optional[Tuple[int, float, float, float]]:
    eligible = [m for m in ue.neighbor_measurements if m.rsrp >= rsrp_threshold]
    if not eligible: return None
    best = max(eligible, key=lambda m: m.rsrp)
    return (best.cell_id, best.rsrp, best.rsrq, best.sinr)

def _find_best_neighbor(ue: UE) -> Optional[NeighborMeasurement]:
    best_neighbor = None; max_rsrp = -np.inf
    for neighbor in ue.neighbor_measurements:
        if neighbor.cell_id != ue.serving_cell and neighbor.rsrp > max_rsrp:
            max_rsrp, best_neighbor = neighbor.rsrp, neighbor
    return best_neighbor

def _move_ue(ue: UE, time_step: float):
    distance = ue.velocity * time_step
    ue.x += distance * math.cos(ue.direction)
    ue.y += distance * math.sin(ue.direction)

# --- Mobility Handlers ---
def _handle_stationary(ue, ts, ct):
    if np.random.random() < 0.05:
        ue.x += (np.random.random() - 0.5) * 2
        ue.y += (np.random.random() - 0.5) * 2
def _handle_pedestrian(ue, ts, ct):
    if ue.pause_timer > 0: ue.pause_timer -= ts
    elif np.random.random() < 0.1: ue.pause_timer = 5 + np.random.random() * 10
    else:
        if np.random.random() < 0.3: ue.direction += (np.random.random() - 0.5) * math.pi
        _move_ue(ue, ts)
def _handle_walk(ue, ts, ct, change_prob, angle):
    if np.random.random() < change_prob: ue.direction += (np.random.random() - 0.5) * angle
    _move_ue(ue, ts)
def _handle_vehicle(ue, ts, ct, interval, angle):
    if ct - ue.last_direction_change > (interval + np.random.random() * 15):
        ue.direction += (np.random.random() - 0.5) * angle
        ue.last_direction_change = ct
    _move_ue(ue, ts)
def _handle_high_speed_train(ue, ts, ct):
    if not ue.in_train: return _handle_vehicle(ue, ts, ct, 40, math.pi/4)
    ue.x += ue.velocity * ts
    if np.random.random() < 0.05: ue.y = max(-2, min(2, ue.y + (np.random.random() - 0.5) * 0.5))
    if ue.x >= (ue.train_start_x + ue.track_length): ue.x = ue.train_start_x + ue.position_in_train
def _handle_highway_vehicle(ue, ts, ct):
    if not ue.on_highway: return _handle_vehicle(ue, ts, ct, 40, math.pi/4)
    _move_ue(ue, ts)
    if np.random.random() < 0.05 and ct - ue.last_direction_change > 5:
        target_lane = max(1, min(ue.num_lanes, ue.lane + np.random.choice([-1, 1])))
        if target_lane != ue.lane:
            ue.lane = target_lane
            y_base = (target_lane - 0.5) * ue.lane_width
            ue.y = y_base if ue.is_forward else -y_base
            ue.last_direction_change = ct
    if ue.x > ue.highway_length / 2: ue.x -= ue.highway_length
    elif ue.x < -ue.highway_length / 2: ue.x += ue.highway_length
def _get_mobility_handler(pattern):
    return {
        'stationary': _handle_stationary, 'pedestrian': _handle_pedestrian,
        'slow_walk': lambda u, ts, ct: _handle_walk(u, ts, ct, 0.3, math.pi/2),
        'normal_walk': lambda u, ts, ct: _handle_walk(u, ts, ct, 0.4, math.pi/2),
        'fast_walk': lambda u, ts, ct: _handle_walk(u, ts, ct, 0.2, math.pi/4),
        'indoor_pedestrian': _handle_pedestrian, 'outdoor_vehicle': lambda u, ts, ct: _handle_vehicle(u, ts, ct, 30, math.pi/6),
        'slow_vehicle': lambda u, ts, ct: _handle_vehicle(u, ts, ct, 20, math.pi/2),
        'fast_vehicle': lambda u, ts, ct: _handle_vehicle(u, ts, ct, 40, math.pi/4),
        'extreme_vehicle': lambda u, ts, ct: _handle_vehicle(u, ts, ct, 60, math.pi/8),
        'vehicle': lambda u, ts, ct: _handle_vehicle(u, ts, ct, 25, math.pi/3),
        'high_speed_train': _handle_high_speed_train, 'highway_vehicle': _handle_highway_vehicle,
    }.get(pattern, _handle_pedestrian)

# --- Boundary Handlers ---
def _enforce_indoor(ue, p):
    if ue.x <= 5 or ue.x >= 115: ue.direction = math.pi - ue.direction; ue.x = np.clip(ue.x, 6, 114)
    if ue.y <= 5 or ue.y >= 45: ue.direction = -ue.direction; ue.y = np.clip(ue.y, 6, 44)
def _enforce_circular(ue, radius, margin, angle_var):
    if math.sqrt(ue.x**2 + ue.y**2) > radius:
        angle = math.atan2(ue.y, ue.x)
        ue.x = (radius - margin) * math.cos(angle)
        ue.y = (radius - margin) * math.sin(angle)
        ue.direction = angle + math.pi + (np.random.random() - 0.5) * angle_var
def _enforce_hst(ue, p):
    if not ue.in_train: return _enforce_circular(ue, p.max_radius, 50, math.pi/4)
    if ue.x > (ue.train_start_x + p.trackLength): ue.x = ue.train_start_x
def _enforce_highway(ue, p):
    if not ue.on_highway: return _enforce_circular(ue, p.max_radius, 20, math.pi/3)
    if ue.x > p.highway_length / 2: ue.x -= p.highway_length
    elif ue.x < -p.highway_length / 2: ue.x += p.highway_length
def _passthrough(ue, p): pass
def _get_boundary_handler(scenario):
    return {
        'indoor_hotspot': _enforce_indoor, 'dense_urban': lambda u, p: _enforce_circular(u, p.max_radius, 10, math.pi/2),
        'urban_macro': lambda u, p: _enforce_circular(u, p.max_radius, 20, math.pi/3),
        'rural': lambda u, p: _enforce_circular(u, p.max_radius, 50, math.pi/4),
        'extreme_rural': lambda u, p: _enforce_circular(u, p.max_radius, 100, math.pi/6),
        'high_speed': _enforce_hst, 'highway': _enforce_highway,
    }.get(scenario, lambda u, p: _enforce_circular(u, p.max_radius, 10, math.pi/2))