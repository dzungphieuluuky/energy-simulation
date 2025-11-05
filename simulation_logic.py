"""
Core simulation logic matching MATLAB functions
Handles UE mobility, measurements, handovers, resource usage
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from fiveg_objects import UE, Cell, SimulationParams, NeighborMeasurement, HandoverEvent
from numba_utils import (
    calculate_path_loss, calculate_rsrq_sinr, 
    evaluate_handover_success_prob, calculate_drop_probability,
    calculate_cell_cpu_usage, calculate_cell_prb_usage,
    calculate_cell_energy_consumption, calculate_theoretical_drop_rate,
    calculate_cell_latency, process_measurements_batch,
    calculate_rsrp_batch
)


def update_ue_mobility(ues: List[UE], time_step: float, current_time: float, seed: int) -> List[UE]:
    """Update all UE positions based on mobility patterns"""
    for ue in ues:
        # Set mobility-specific RNG
        # FIX: Ensure the seed is always a valid non-negative 32-bit integer
        seed_val = (ue.rng_seed + int(math.floor(current_time * 1000))) % (2**32)
        np.random.seed(seed_val)
        
        if not hasattr(ue, 'step_counter'): ue.step_counter = 0
        ue.step_counter += 1
        
        # Update position based on mobility pattern
        if ue.mobility_pattern == 'stationary':
            _handle_stationary_mobility(ue, time_step)
        elif ue.mobility_pattern in ['pedestrian', 'indoor_pedestrian']:
            _handle_pedestrian_mobility(ue, time_step, current_time)
        elif 'walk' in ue.mobility_pattern:
            _handle_walk_mobility(ue, time_step, ue.mobility_pattern)
        elif 'vehicle' in ue.mobility_pattern or ue.mobility_pattern in ['slow_vehicle', 'fast_vehicle']:
            _handle_vehicle_mobility(ue, time_step, current_time, ue.mobility_pattern)
        elif ue.mobility_pattern == 'high_speed_train':
            _handle_high_speed_train_mobility(ue, time_step, current_time)
        elif ue.mobility_pattern == 'highway_vehicle':
            _handle_highway_vehicle_mobility(ue, time_step, current_time)
        
        # Enforce deployment scenario boundaries
        _enforce_scenario_bounds(ue)
        
        # Normalize direction
        ue.direction = ue.direction % (2 * math.pi)
    
    return ues


def _handle_stationary_mobility(ue: UE, time_step: float):
    """Stationary UE with occasional small movements"""
    if np.random.random() < 0.05:  # 5% chance
        ue.x += (np.random.random() - 0.5) * 2
        ue.y += (np.random.random() - 0.5) * 2


def _handle_pedestrian_mobility(ue: UE, time_step: float, current_time: float):
    """Pedestrian mobility with pauses and direction changes"""
    distance = ue.velocity * time_step
    
    if ue.pause_timer > 0:
        ue.pause_timer -= time_step
    elif np.random.random() < 0.1:  # 10% chance to pause
        ue.pause_timer = 5 + np.random.random() * 10
    elif np.random.random() < 0.3:  # 30% chance to change direction
        ue.direction += (np.random.random() - 0.5) * math.pi
        _move_ue(ue, distance)
    else:
        _move_ue(ue, distance)


def _handle_walk_mobility(ue: UE, time_step: float, pattern: str):
    """Different walking speeds with direction changes"""
    distance = ue.velocity * time_step
    
    if 'slow' in pattern and np.random.random() < 0.3:
        ue.direction += (np.random.random() - 0.5) * math.pi / 2
    elif 'normal' in pattern and np.random.random() < 0.4:
        ue.direction += (np.random.random() - 0.5) * math.pi / 2
    elif 'fast' in pattern and np.random.random() < 0.2:
        ue.direction += (np.random.random() - 0.5) * math.pi / 4
    
    _move_ue(ue, distance)


def _handle_vehicle_mobility(ue: UE, time_step: float, current_time: float, pattern: str):
    """Vehicle mobility with less frequent direction changes"""
    distance = ue.velocity * time_step
    
    if 'slow' in pattern:
        change_interval = 20 + np.random.random() * 30
        angle_change = math.pi / 2
    elif 'fast' in pattern or 'extreme' in pattern:
        change_interval = 40 + np.random.random() * 20
        angle_change = math.pi / 4
    else:
        change_interval = 25 + np.random.random() * 15
        angle_change = math.pi / 3
    
    if current_time - ue.last_direction_change > change_interval:
        ue.direction += (np.random.random() - 0.5) * angle_change
        ue.last_direction_change = current_time
    
    _move_ue(ue, distance)


def _handle_high_speed_train_mobility(ue: UE, time_step: float, current_time: float):
    """High-speed train mobility along track"""
    if not ue.in_train:
        _handle_vehicle_mobility(ue, time_step, current_time, 'fast_vehicle')
        return
    
    distance = ue.velocity * time_step
    ue.x += distance  # Move along x-axis
    
    if np.random.random() < 0.05:
        ue.y += (np.random.random() - 0.5) * 0.5
        ue.y = max(-2, min(2, ue.y))
    
    track_end_x = ue.train_start_x + ue.track_length
    if ue.x >= track_end_x:
        ue.x = ue.train_start_x + ue.position_in_train


def _handle_highway_vehicle_mobility(ue: UE, time_step: float, current_time: float):
    """Highway vehicle mobility with lane changes"""
    if not ue.on_highway:
        _handle_vehicle_mobility(ue, time_step, current_time, 'fast_vehicle')
        return
    
    distance = ue.velocity * time_step
    ue.x += distance * math.cos(ue.direction)
    
    if np.random.random() < 0.05 and current_time - ue.last_direction_change > 5:
        target_lane = ue.lane + np.random.choice([-1, 1])
        target_lane = max(1, min(ue.num_lanes, target_lane))
        
        if target_lane != ue.lane:
            ue.lane = target_lane
            y_base = (target_lane - 0.5) * ue.lane_width
            ue.y = y_base if ue.direction == 0 else -y_base
            ue.last_direction_change = current_time
    
    if ue.x > ue.highway_length / 2: ue.x -= ue.highway_length
    elif ue.x < -ue.highway_length / 2: ue.x += ue.highway_length


def _move_ue(ue: UE, distance: float):
    """Move UE in current direction"""
    ue.x += distance * math.cos(ue.direction)
    ue.y += distance * math.sin(ue.direction)


def _enforce_scenario_bounds(ue: UE):
    """Enforce deployment scenario boundaries"""
    scenario_bounds = {
        'indoor_hotspot': (lambda u: _enforce_indoor_bounds(u)),
        'dense_urban': (lambda u: _enforce_urban_bounds(u, max_radius=800)),
        'urban_macro': (lambda u: _enforce_urban_bounds(u, max_radius=800)),
        'rural': (lambda u: _enforce_rural_bounds(u, max_radius=2000)),
        'extreme_rural': (lambda u: _enforce_rural_bounds(u, max_radius=50000)),
    }
    
    bound_func = scenario_bounds.get(ue.deployment_scenario)
    if bound_func:
        bound_func(ue)


def _enforce_indoor_bounds(ue: UE):
    """Indoor building bounds (120m x 50m)"""
    if ue.x <= 5 or ue.x >= 115:
        ue.x = np.clip(ue.x, 6, 114)
        ue.direction = math.pi - ue.direction
    if ue.y <= 5 or ue.y >= 45:
        ue.y = np.clip(ue.y, 6, 44)
        ue.direction = -ue.direction

def _enforce_urban_bounds(ue: UE, max_radius: float):
    """Urban area circular bounds"""
    if math.sqrt(ue.x**2 + ue.y**2) > max_radius:
        angle = math.atan2(ue.y, ue.x)
        ue.x = (max_radius - 10) * math.cos(angle)
        ue.y = (max_radius - 10) * math.sin(angle)
        ue.direction = angle + math.pi + (np.random.random() - 0.5) * math.pi / 2

def _enforce_rural_bounds(ue: UE, max_radius: float):
    """Rural area circular bounds"""
    if math.sqrt(ue.x**2 + ue.y**2) > max_radius:
        angle = math.atan2(ue.y, ue.x)
        ue.x = (max_radius - 50) * math.cos(angle)
        ue.y = (max_radius - 50) * math.sin(angle)
        ue.direction = angle + math.pi + (np.random.random() - 0.5) * math.pi / 4


def update_signal_measurements(ues: List[UE], cells: List[Cell], 
                               rsrp_measurement_threshold: float, 
                               current_time: float, seed: int) -> List[UE]:
    """Update RSRP/RSRQ/SINR measurements for all UEs"""
    if not ues or not cells:
        return ues

    # --- Step 1: Data Extraction (same as before) ---
    # Convert Python object attributes into fast NumPy arrays
    ue_positions = np.array([[ue.x, ue.y] for ue in ues], dtype=np.float64)
    cell_positions = np.array([[c.x, c.y] for c in cells], dtype=np.float64)
    tx_powers = np.array([c.tx_power for c in cells], dtype=np.float64)
    min_tx_powers = np.array([c.min_tx_power for c in cells], dtype=np.float64)
    ue_ids = np.array([ue.id for ue in ues], dtype=np.int32)
    frequency = cells[0].frequency

    # --- Step 2: Numba Kernels ---
    # Call the first fast Numba kernel to get all RSRP values
    rsrp_matrix = calculate_rsrp_batch(
        ue_positions, cell_positions, tx_powers, frequency,
        ue_ids, current_time, seed, min_tx_powers
    )

    # Call the SECOND fast Numba kernel to process the RSRPs into RSRQs and SINRs
    rsrq_matrix, sinr_matrix = process_measurements_batch(
        rsrp_matrix, tx_powers, min_tx_powers, rsrp_measurement_threshold
    )

    # --- Step 3: Data Re-integration ---
    # Now, we do ONE final, fast Python loop to update the objects from the result matrices.
    # This is much faster than doing calculations inside the loop.
    for i, ue in enumerate(ues):
        measurements = []
        for j, cell in enumerate(cells):
            rsrp = rsrp_matrix[i, j]
            
            # Check for NaN, which indicates the measurement was below the threshold in Numba
            if not np.isnan(rsrp) and not np.isnan(sinr_matrix[i, j]):
                measurements.append(
                    NeighborMeasurement(
                        cell_id=cell.id, 
                        rsrp=rsrp, 
                        rsrq=rsrq_matrix[i, j], 
                        sinr=sinr_matrix[i, j]
                    )
                )
        
        # Update UE's state (this part is the same as before)
        if not measurements:
            ue.serving_cell = None
            ue.rsrp = ue.rsrq = ue.sinr = np.nan
            ue.neighbor_measurements = []
        else:
            serving_meas = next((m for m in measurements if m.cell_id == ue.serving_cell), None)
            if serving_meas:
                ue.rsrp, ue.rsrq, ue.sinr = serving_meas.rsrp, serving_meas.rsrq, serving_meas.sinr
            else:
                ue.serving_cell = None
                ue.rsrp = ue.rsrq = ue.sinr = np.nan
            ue.neighbor_measurements = measurements
            
    return ues


def check_handover_events(ues: List[UE], cells: List[Cell], current_time: float, 
                         sim_params: SimulationParams, seed: int) -> Tuple[List[HandoverEvent], List[UE]]:
    """Check and execute handover events"""
    handover_events = []
    
    for ue in ues:
        serving_cell = next((c for c in cells if c.id == ue.serving_cell), None)
        if not serving_cell:
            continue
        
        # Emergency Handover
        if not np.isnan(ue.rsrp) and ue.rsrp < sim_params.rsrpServingThreshold:
            best_cell = _find_best_cell_for_connection(ue, sim_params.rsrpTargetThreshold, cells)
            if best_cell:
                ue.serving_cell = best_cell['cell_id']
                ue.rsrp = best_cell['rsrp']
                ue.rsrq = best_cell['rsrq']
                ue.sinr = best_cell['sinr']
                ue.is_dropped = False  # FIX: Clear dropped flag on reconnection
            else:
                ue.serving_cell = None
                ue.is_dropped = True  # FIX: Mark as dropped
            continue
        
        # A3 Handover (find best neighbor)
        best_neighbor = None
        max_rsrp = -np.inf
        for neighbor in ue.neighbor_measurements:
            if neighbor.cell_id != ue.serving_cell and neighbor.rsrp > max_rsrp:
                max_rsrp = neighbor.rsrp
                best_neighbor = neighbor
        
        # Check A3 condition
        if (best_neighbor and 
            best_neighbor.rsrp > (ue.rsrp + serving_cell.a3_offset) and 
            best_neighbor.rsrp >= sim_params.rsrpTargetThreshold):
            
            # Start timer
            if ue.ho_timer == 0:
                ue.ho_timer = current_time
            
            # Check if TTT expired
            if (current_time - ue.ho_timer) >= serving_cell.ttt:
                target_cell = next((c for c in cells if c.id == best_neighbor.cell_id), None)
                if not target_cell:
                    continue
                
                # Evaluate handover success
                ho_success = _evaluate_handover_success(
                    ue, best_neighbor, serving_cell, target_cell, current_time, seed
                )
                
                # Create event
                ho_event = HandoverEvent(
                    ue_id=ue.id, cell_source=serving_cell.id, cell_target=best_neighbor.cell_id,
                    rsrp_source=ue.rsrp, rsrp_target=best_neighbor.rsrp,
                    rsrq_source=ue.rsrq, rsrq_target=best_neighbor.rsrq,
                    sinr_source=ue.sinr, sinr_target=best_neighbor.sinr,
                    a3_offset=serving_cell.a3_offset, ttt=serving_cell.ttt,
                    ho_success=ho_success, timestamp=current_time
                )
                
                handover_events.append(ho_event)
                ue.handover_history.append(ho_event)
                
                # Execute handover if successful
                if ho_success:
                    ue.serving_cell = best_neighbor.cell_id
                    ue.rsrp = best_neighbor.rsrp
                    ue.rsrq = best_neighbor.rsrq
                    ue.sinr = best_neighbor.sinr
                    # FIX: UE successfully handed over, not dropped
                    ue.is_dropped = False
                else:
                    # FIX: Failed handover might lead to drop
                    # (MATLAB doesn't explicitly handle this, but it's implicit)
                    pass
                
                ue.ho_timer = 0
        else:
            # A3 condition not met - reset timer
            ue.ho_timer = 0
            
    return handover_events, ues


def _find_best_cell_for_connection(ue: UE, rsrp_threshold: float, cells: List[Cell]) -> Optional[dict]:
    """Find best cell for emergency connection"""
    eligible = [m for m in ue.neighbor_measurements if m.rsrp >= rsrp_threshold]
    if not eligible: return None
    
    best = max(eligible, key=lambda m: m.rsrp)
    return {'cell_id': best.cell_id, 'rsrp': best.rsrp, 'rsrq': best.rsrq, 'sinr': best.sinr}


def _evaluate_handover_success(ue: UE, neighbor: NeighborMeasurement, 
                               serving_cell: Cell, target_cell: Cell,
                               current_time: float, seed: int) -> bool:
    """Evaluate if handover succeeds"""
    ho_seed = (seed + ue.id + neighbor.cell_id + int(math.floor(current_time))) % (2**32)
    np.random.seed(ho_seed)
    
    prob = evaluate_handover_success_prob(
        serving_cell.tx_power, serving_cell.min_tx_power, target_cell.tx_power, target_cell.min_tx_power,
        neighbor.rsrp, neighbor.sinr, target_cell.cpu_usage, target_cell.prb_usage
    )
    return np.random.random() < prob


def handle_disconnected_ues(ues: List[UE], cells: List[Cell], sim_params: SimulationParams,
                           time_step: float, current_time: float) -> List[UE]:
    """Handle disconnected UEs and connection/disconnection timers - MATLAB-exact logic"""
    disconnection_timeout, connection_timeout, hysteresis = 5.0, 2.0, 3.0
    
    for ue in ues:
        if ue.serving_cell is not None:
            # Check signal quality
            if not np.isnan(ue.rsrp) and ue.rsrp < (sim_params.rsrpServingThreshold - hysteresis):
                # FIX: MATLAB-style timer logic - check if 0 FIRST, then set
                if ue.disconnection_timer == 0:
                    ue.disconnection_timer = disconnection_timeout
                else:
                    ue.disconnection_timer -= time_step
                    if ue.disconnection_timer <= 0:
                        # Disconnect UE
                        ue.disconnection_timer = 0
                        ue.is_dropped = True  # FIX: Mark as dropped
                        ue.serving_cell = None
                        ue.rsrp = ue.rsrq = ue.sinr = np.nan
                        ue.session_active = False
                        ue.traffic_demand = 0
                        ue.drop_count += 1
            else:
                # Signal good - reset timer
                ue.disconnection_timer = 0
        
        else:  # UE not connected - try to connect
            best_cell = _find_best_cell_for_connection(
                ue, sim_params.rsrpServingThreshold + hysteresis, cells
            )
            
            if best_cell:
                # FIX: MATLAB-style timer logic
                if ue.connection_timer == 0:
                    ue.connection_timer = connection_timeout
                else:
                    ue.connection_timer -= time_step
                    if ue.connection_timer <= 0:
                        ue.connection_timer = 0
                        # Verify still best cell
                        current_best = _find_best_cell_for_connection(
                            ue, sim_params.rsrpServingThreshold + hysteresis, cells
                        )
                        if current_best and current_best['cell_id'] == best_cell['cell_id']:
                            # Connect UE
                            ue.serving_cell = best_cell['cell_id']
                            ue.rsrp = best_cell['rsrp']
                            ue.rsrq = best_cell['rsrq']
                            ue.sinr = best_cell['sinr']
                            ue.is_dropped = False  # FIX: Clear dropped flag
            else:
                # No suitable cell - reset timer
                ue.connection_timer = 0
    
    return ues


def update_traffic_generation(ues: List[UE], cells: List[Cell], 
                             current_time: float, sim_params: SimulationParams) -> Tuple[List[UE], List[Cell]]:
    """Generate Poisson traffic for UEs"""
    lambda_val = sim_params.trafficLambda * sim_params.peakHourMultiplier
    
    # Reset cell loads
    for cell in cells:
        cell.current_load = 0
        cell.connected_ues = []
    
    # Generate traffic for each UE
    for ue in ues:
        if ue.serving_cell is not None and not ue.is_dropped:  # FIX: Check is_dropped
            traffic_demand = float(np.random.poisson(lambda_val / len(ues)))
            ue.traffic_demand = traffic_demand
            ue.session_active = traffic_demand > 0
            
            serving_cell = next((c for c in cells if c.id == ue.serving_cell), None)
            if serving_cell:
                serving_cell.current_load += traffic_demand
                serving_cell.connected_ues.append(ue.id)
        else:
            # FIX: Ensure disconnected/dropped UEs have no traffic
            ue.traffic_demand = 0
            ue.session_active = False
    
    return ues, cells


def update_ue_drop_events(ues: List[UE], cells: List[Cell], current_time: float) -> Tuple[List[UE], List[Cell]]:
    """Handle UE drop events based on conditions"""
    for ue in ues:
        if ue.serving_cell is not None and ue.session_active:
            serving_cell = next((c for c in cells if c.id == ue.serving_cell), None)
            if serving_cell:
                drop_prob = calculate_drop_probability(
                    serving_cell.tx_power, serving_cell.min_tx_power, serving_cell.max_tx_power,
                    ue.sinr, ue.rsrp, serving_cell.cpu_usage, serving_cell.prb_usage,
                    serving_cell.current_load, serving_cell.max_capacity
                )
                if np.random.random() < drop_prob:
                    serving_cell.actual_drop_count = getattr(serving_cell, 'actual_drop_count', 0) + 1
                    serving_cell.total_drop_events = getattr(serving_cell, 'total_drop_events', 0) + 1
                    
                    # FIX: Set is_dropped flag
                    ue.is_dropped = True
                    ue.serving_cell = None
                    ue.rsrp = ue.rsrq = ue.sinr = np.nan
                    ue.session_active = False
                    ue.traffic_demand = 0
                    ue.drop_count += 1
            else:
                # FIX: Set is_dropped when cell not found
                ue.is_dropped = True
                ue.serving_cell = None
                ue.rsrp = ue.rsrq = ue.sinr = np.nan
                ue.session_active = False
                ue.traffic_demand = 0
                ue.drop_count += 1
    return ues, cells


def update_cell_resource_usage(cells: List[Cell], ues: List[UE]) -> List[Cell]:
    """Update CPU, PRB, energy, and performance metrics for all cells"""
    for cell in cells:
        connected_ues = [ue for ue in ues if ue.serving_cell == cell.id]
        num_ues = len(connected_ues)
        load_ratio = min(1.0, cell.current_load / max(1, cell.max_capacity))
        power_ratio = (cell.tx_power - cell.min_tx_power) / max(1e-6, cell.max_tx_power - cell.min_tx_power)
        
        cell.cpu_usage = calculate_cell_cpu_usage(power_ratio, num_ues, load_ratio)
        cell.prb_usage = calculate_cell_prb_usage(num_ues, load_ratio)
        cell.energy_consumption = calculate_cell_energy_consumption(
            cell.base_energy_consumption, cell.tx_power, num_ues, load_ratio
        )
        
        valid_sinrs = [ue.sinr for ue in connected_ues if not np.isnan(ue.sinr)]
        avg_sinr = np.mean(valid_sinrs) if valid_sinrs else 0.0
        cell.avg_sinr = avg_sinr
        
        cell.total_connection_time = getattr(cell, 'total_connection_time', 0) + num_ues
        cell.actual_drop_count = getattr(cell, 'actual_drop_count', 0)
        
        actual_drop_rate = (cell.actual_drop_count / cell.total_connection_time) * 100 if cell.total_connection_time > 0 else 0.0
        theoretical_drop_rate = calculate_theoretical_drop_rate(
            0.1, cell.tx_power, cell.min_tx_power, cell.max_tx_power,
            cell.cpu_usage, cell.prb_usage, avg_sinr, len(valid_sinrs)
        )
        
        cell.drop_rate = actual_drop_rate if cell.total_connection_time > 10 else theoretical_drop_rate
        cell.theoretical_drop_rate = theoretical_drop_rate
        
        if cell.total_connection_time > 1000:
            cell.actual_drop_count = int(cell.actual_drop_count * 0.9)
            cell.total_connection_time = int(cell.total_connection_time * 0.9)
        
        cell.avg_latency = calculate_cell_latency(
            10.0, load_ratio, num_ues, cell.tx_power, 
            cell.min_tx_power, cell.max_tx_power, power_ratio
        )
    return cells