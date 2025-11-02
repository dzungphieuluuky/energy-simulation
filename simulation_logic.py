import numpy as np
import math
from numba import jit
from typing import List, Dict, Any, Tuple, Optional

from fiveg_objects import UE, Cell
from numba_utils import *

# ------------------------------
# RNG helpers / path-loss
# ------------------------------
def rng_for(seed: int, offset: int = 0) -> np.random.RandomState:
    """Create a numpy RandomState with proper seed handling."""
    combined_seed: int = (int(seed) + int(offset)) & 0xFFFFFFFF
    return np.random.RandomState(combined_seed)

@jit(nopython=True)
def numba_calculate_path_loss(distance: float, frequency: float, rng_state: int) -> float:
    """Numba-optimized path loss calculation."""
    if distance < 10.0:
        distance = 10.0

    fc = frequency / 1e9

    # Calculate LOS probability
    if distance <= 18.0:
        pLOS = 1.0
    else:
        pLOS = 18.0 / distance + math.exp(-distance / 36.0) * (1.0 - 18.0 / distance)

    # Simple deterministic RNG for consistency
    # Using the rng_state to generate deterministic "random" numbers
    temp = (rng_state * 1103515245 + 12345) & 0x7fffffff
    los_random = (temp % 10000) / 10000.0

    if los_random < pLOS:
        path_loss = 32.4 + 21.0 * math.log10(distance) + 20.0 * math.log10(fc)
    else:
        path_loss = 35.3 * math.log10(distance) + 22.4 + 21.3 * math.log10(fc) - 0.3 * (1.5 - 1.5)

    # Generate shadowing using the same RNG approach
    temp = (temp * 1103515245 + 12345) & 0x7fffffff
    shadow = ((temp % 10000) / 10000.0 - 0.5) * 2.0 * 4.0  # ±4 dB shadowing

    return path_loss + shadow


# ------------------------------
# Measurements
# ------------------------------

@jit(nopython=True)
def numba_update_signal_measurements(ues: list[NumbaUE],
                                   cells: list[NumbaCell],
                                   rsrp_threshold: float, current_time: float, seed: int):
    """Numba-optimized signal measurement update."""
    n_ues = len(ues)
    n_cells = len(cells)

    for i in range(n_ues):
        ue = ues[i]
        best_rsrp = -200.0
        best_cell_id = -1
        best_rsrq = -40.0
        best_sinr = -40.0

        for j in range(n_cells):
            cell = cells[j]

            # Calculate distance
            dx = ue.x - cell.x
            dy = ue.y - cell.y
            distance = math.sqrt(dx*dx + dy*dy)

            # Create unique RNG state for this UE-cell pair
            rng_state = seed + ue.id * 1000 + cell.id + int(current_time * 1000)

            # Calculate path loss
            pl = numba_calculate_path_loss(distance, cell.frequency, rng_state)

            # Calculate measurements with noise
            rng_state2 = rng_state + 1000000
            temp = (rng_state2 * 1103515245 + 12345) & 0x7fffffff
            rsrp_noise = ((temp % 10000) / 10000.0 - 0.5) * 2.0 * 1.5

            temp = (temp * 1103515245 + 12345) & 0x7fffffff
            rsrq_noise = ((temp % 10000) / 10000.0 - 0.5) * 2.0 * 0.5

            temp = (temp * 1103515245 + 12345) & 0x7fffffff
            sinr_noise = ((temp % 10000) / 10000.0 - 0.5) * 2.0 * 1.0

            rsrp = cell.txPower - pl + rsrp_noise
            rsrq = rsrp - 10.0 + rsrq_noise
            sinr = rsrp - (-100.0) + sinr_noise

            # Update best cell
            if rsrp > best_rsrp:
                best_rsrp = rsrp
                best_cell_id = cell.id
                best_rsrq = rsrq
                best_sinr = sinr

        # Update UE measurements
        if best_rsrp >= rsrp_threshold:
            ue.servingCell = best_cell_id
            ue.rsrp = best_rsrp
            ue.rsrq = best_rsrq
            ue.sinr = best_sinr
        else:
            ue.servingCell = -1
            ue.rsrp = -200.0
            ue.rsrq = -40.0
            ue.sinr = -40.0

# ------------------------------
# Cell resource / energy update
# ------------------------------
@jit(nopython=True)
def numba_update_cell_resource_usage(ues: list[NumbaUE],
                                   cells: list[NumbaCell]):
    """Numba-optimized cell resource usage update."""
    n_cells = len(cells)

    # Reset cell loads and count connected UEs
    for i in range(n_cells):
        cells[i].currentLoad = 0.0

    # Calculate total traffic per cell
    for i in range(len(ues)):
        ue = ues[i]
        if ue.servingCell != -1:
            for j in range(n_cells):
                if cells[j].id == ue.servingCell:
                    cells[j].currentLoad += ue.trafficDemand
                    break

    # Update cell metrics
    for i in range(n_cells):
        cell = cells[i]

        # Count connected UEs (simplified - in practice you'd track this separately)
        connected_ues = 0
        for ue in ues:
            if ue.servingCell == cell.id:
                connected_ues += 1

        # Calculate load ratio
        load_ratio = min(1.0, cell.currentLoad / max(1.0, cell.maxCapacity))
        power_range = max(1e-6, cell.maxTxPower - cell.minTxPower)
        power_ratio = (cell.txPower - cell.minTxPower) / power_range

        # CPU usage
        base_cpu = 10.0 + power_ratio * 5.0
        per_ue_cpu = connected_ues * 2.5
        load_cpu = load_ratio * 50.0
        cell.cpuUsage = min(95.0, base_cpu + per_ue_cpu + load_cpu)

        # PRB usage
        base_prb = connected_ues * 3.0
        load_prb = load_ratio * 60.0
        cell.prbUsage = min(95.0, base_prb + load_prb)

        # Energy consumption
        tx_power_consumption = 10.0 ** ((cell.txPower - 30.0) / 10.0)
        per_ue_energy = connected_ues * 15.0
        load_energy = load_ratio * 200.0
        cell.energyConsumption = cell.baseEnergyConsumption + tx_power_consumption + per_ue_energy + load_energy

        # Simplified drop rate and latency (you can make this more sophisticated)
        cell.dropRate = max(0.1, 2.0 - power_ratio * 1.5 + load_ratio * 3.0)
        cell.avgLatency = 10.0 + load_ratio * 25.0 + connected_ues * 0.8

@jit(nopython=True)
def numba_update_ue_mobility(ues: list[NumbaUE],
                           time_step: float, current_time: float, seed: int,
                           max_radius: float):
    """Numba-optimized UE mobility update."""
    for i in range(len(ues)):
        ue = ues[i]
        distance = ue.velocity * time_step

        # Simple RNG for mobility decisions
        rng_state = seed + ue.id + int(current_time * 1000)
        temp = (rng_state * 1103515245 + 12345) & 0x7fffffff
        rand_val = (temp % 10000) / 10000.0

        # Handle different mobility patterns
        if ue.pauseTimer > 0:
            ue.pauseTimer -= time_step
        elif rand_val < 0.10:  # 10% chance to pause
            temp = (temp * 1103515245 + 12345) & 0x7fffffff
            ue.pauseTimer = 5.0 + ((temp % 10000) / 10000.0) * 10.0
        elif rand_val < 0.40:  # 30% chance to change direction
            temp = (temp * 1103515245 + 12345) & 0x7fffffff
            direction_change = ((temp % 10000) / 10000.0 - 0.5) * math.pi
            ue.direction = numba_clip_angle(ue.direction + direction_change)
            numba_move_ue(ue, distance)
        else:  # Continue in current direction
            numba_move_ue(ue, distance)

        # Enforce bounds if max_radius is specified
        if max_radius > 0:
            r = math.sqrt(ue.x*ue.x + ue.y*ue.y)
            if r > max_radius:
                # Redirect towards center
                angle_to_center = math.atan2(-ue.y, -ue.x)
                temp = (temp * 1103515245 + 12345) & 0x7fffffff
                ue.direction = angle_to_center + ((temp % 10000) / 10000.0 - 0.5) * math.pi / 8.0
                numba_move_ue(ue, 0.1)

@jit(nopython=True)
def numba_generate_traffic(ues: list[NumbaUE],
                         traffic_lambda: float, num_ues: int, seed: int, current_step: int):
    """Numba-optimized traffic generation."""
    # Simple Poisson-like traffic generation
    avg_traffic = max(1.0, traffic_lambda / max(1, num_ues))

    for i in range(len(ues)):
        # Simple deterministic "random" traffic
        rng_state = seed + ues[i].id + current_step
        temp = (rng_state * 1103515245 + 12345) & 0x7fffffff
        random_val = (temp % 10000) / 10000.0

        # Poisson approximation using exponential
        if random_val < 0.9:  # 90% chance of having traffic
            ues[i].trafficDemand = avg_traffic * (0.5 + random_val)
        else:
            ues[i].trafficDemand = 0.0


# ------------------------------
# The simulation step driver (replaces sim.run_simulation_step)
# ------------------------------
def optimized_run_simulation_step(ues: List[UE], cells: List[Cell], neighbor_measurements: np.ndarray,
                                config: Dict[str, Any], time_step_duration: float,
                                current_step: int, action: Optional[np.ndarray]) -> Tuple[List[UE], List[Cell], np.ndarray]:
    """
    Optimized simulation step using Numba for performance-critical parts.
    """
    # Convert to Numba objects (this is the main overhead, but worth it for large simulations)
    numba_ues = convert_ues_to_numba(ues)
    numba_cells = convert_cells_to_numba(cells)

    seed = config.get('seed', 0)
    max_radius = config.get('maxRadius', -1.0)  # -1 means no boundary

    # 1) Apply action (still in Python, but fast)
    if action is not None:
        for idx, cell in enumerate(cells):
            if idx < len(action):
                pr = float(np.clip(action[idx], 0.0, 1.0))
                cell.txPower = cell.minTxPower + pr * (cell.maxTxPower - cell.minTxPower)
                # Update the Numba object as well for subsequent calculations
                if idx < len(numba_cells):
                    numba_cells[idx].txPower = cell.txPower
    
    # 2) Mobility (Numba-optimized)
    numba_update_ue_mobility(numba_ues, time_step_duration,
                           current_step * time_step_duration, seed, max_radius)

    # 3) Traffic generation (Numba-optimized)
    traffic_lambda = config.get('trafficLambda', 30.0)
    numba_generate_traffic(numba_ues, traffic_lambda, len(ues), seed, current_step)

    # 4) Signal measurements (Numba-optimized)
    rsrp_threshold = config.get('rsrpMeasurementThreshold', -115.0)
    numba_update_signal_measurements(numba_ues, numba_cells, rsrp_threshold,
                                   current_step * time_step_duration, seed)

    # 5) Cell resource usage (Numba-optimized)
    numba_update_cell_resource_usage(numba_ues, numba_cells)

    # 6) Convert back to regular objects
    convert_numba_to_regular(numba_ues, numba_cells, ues, cells)
    
    # 7) Update connected UEs list on Python Cell objects (for metrics)
    for cell in cells:
        cell.connectedUEs.clear()
    
    ue_map = {ue.id: ue for ue in ues}
    for ue in ues:
        if ue.servingCell is not None:
            for cell in cells:
                if cell.id == ue.servingCell:
                    cell.connectedUEs.append(ue)
                    break

    # 8) Neighbor measurements (optional, can also be optimized with Numba if needed)
    if neighbor_measurements is not None:
        max_neighbors = neighbor_measurements.shape[1]
        neighbor_measurements[:] = -1

        # This part could also be optimized with Numba if it becomes a bottleneck
        for ui, ue in enumerate(ues):
            measures = []
            for cell in cells:
                dx = ue.x - cell.x
                dy = ue.y - cell.y
                d = math.hypot(dx, dy)
                # Use the optimized path loss function
                pl = numba_calculate_path_loss(d, cell.frequency, seed + ue.id + cell.id)
                est_rsrp = cell.txPower - pl
                measures.append((est_rsrp, cell.id))

            measures.sort(reverse=True)
            for ni in range(min(max_neighbors, len(measures))):
                neighbor_measurements[ui, ni] = int(measures[ni][1])

    return ues, cells, neighbor_measurements
