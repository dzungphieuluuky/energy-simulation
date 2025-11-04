"""
Numba-accelerated utility functions for 5G simulation
Optimizes computational bottlenecks using JIT compilation
"""

import numpy as np
from numba import jit, prange
import math


@jit(nopython=True, cache=True)
def calculate_path_loss(distance, frequency_hz, ue_id, current_time, seed):
    """
    Calculate path loss using 3GPP indoor hotspot model with shadow fading
    Matches MATLAB calculatePathLoss exactly
    """
    # Set seed for reproducibility
    pl_seed = seed + 5000 + ue_id + int(math.floor(current_time * 100))
    np.random.seed(pl_seed)
    
    # Minimum distance check
    if distance < 10:
        distance = 10.0
    
    fc = frequency_hz / 1e9  # Convert to GHz
    hBS = 25.0  # Base station height
    hUT = 1.5   # User terminal height
    
    # LOS probability calculation
    if distance <= 18:
        pLOS = 1.0
    else:
        pLOS = 18.0/distance + math.exp(-distance/36.0) * (1.0 - 18.0/distance)
    
    # Determine LOS/NLOS
    if np.random.random() < pLOS:
        # LOS path loss
        path_loss = 32.4 + 21.0*math.log10(distance) + 20.0*math.log10(fc)
    else:
        # NLOS path loss
        path_loss = 35.3*math.log10(distance) + 22.4 + 21.3*math.log10(fc) - 0.3*(hUT-1.5)
    
    # Shadow fading (log-normal)
    shadow_fading = np.random.randn() * 4.0
    path_loss = path_loss + shadow_fading
    
    return path_loss


@jit(nopython=True, cache=True)
def calculate_rsrp_batch(ue_positions, cell_positions, tx_powers, frequency, 
                         ue_ids, current_time, seed, min_tx_powers):
    """
    Vectorized RSRP calculation for all UE-cell pairs
    """
    num_ues = ue_positions.shape[0]
    num_cells = cell_positions.shape[0]
    rsrp_matrix = np.zeros((num_ues, num_cells), dtype=np.float64)
    
    for ue_idx in range(num_ues):
        ue_x, ue_y = ue_positions[ue_idx]
        ue_id = ue_ids[ue_idx]
        
        for cell_idx in range(num_cells):
            cell_x, cell_y = cell_positions[cell_idx]
            
            # Calculate distance
            dx = ue_x - cell_x
            dy = ue_y - cell_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate path loss
            path_loss = calculate_path_loss(distance, frequency, ue_id, current_time, seed)
            
            # RSRP with measurement variation
            rsrp_seed = seed + 6000 + ue_id + cell_idx + int(math.floor(current_time * 100))
            np.random.seed(rsrp_seed)
            
            rsrp = tx_powers[cell_idx] - path_loss + np.random.randn() * 1.5
            
            # Power penalty for low transmit power
            if tx_powers[cell_idx] <= min_tx_powers[cell_idx] + 2:
                power_penalty = (min_tx_powers[cell_idx] + 2 - tx_powers[cell_idx]) * 8.0
                rsrp = rsrp - power_penalty
                rsrp = rsrp + np.random.randn() * 3.0
            
            rsrp_matrix[ue_idx, cell_idx] = rsrp
    
    return rsrp_matrix


@jit(nopython=True, cache=True)
def calculate_rsrq_sinr(rsrp, rsrp_measurement_threshold):
    """
    Calculate RSRQ and SINR from RSRP
    Matches MATLAB calculation exactly
    """
    if rsrp >= (rsrp_measurement_threshold - 5):
        # RSSI calculation
        rssi = rsrp + 10.0*math.log10(12.0) + np.random.randn() * 0.5
        
        # RSRQ calculation (clamped between -20 and -3 dB)
        rsrq = 10.0*math.log10(12.0) + rsrp - rssi
        rsrq = max(-20.0, min(-3.0, rsrq))
        
        # SINR calculation
        base_sinr = rsrp - (-110.0)  # Noise floor at -110 dBm
        sinr = base_sinr + np.random.randn() * 2.0
        
        return rsrq, sinr
    else:
        return np.nan, np.nan


@jit(nopython=True, cache=True)
def evaluate_handover_success_prob(serving_tx_power, serving_min_power, 
                                   target_tx_power, target_min_power,
                                   neighbor_rsrp, neighbor_sinr,
                                   target_cpu_usage, target_prb_usage):
    """
    Calculate handover success probability based on multiple factors
    Matches MATLAB evaluateHandoverSuccess logic
    """
    base_success_prob = 0.98
    
    # Source cell power penalty
    if serving_tx_power <= serving_min_power + 2:
        source_power_penalty = (serving_min_power + 2 - serving_tx_power) * 0.15
        base_success_prob = base_success_prob - source_power_penalty
    
    # Target cell power penalty
    if target_tx_power <= target_min_power + 3:
        target_power_penalty = (target_min_power + 3 - target_tx_power) * 0.10
        base_success_prob = base_success_prob - target_power_penalty
    
    # Signal quality bonus/penalty
    if neighbor_rsrp >= -75:
        signal_bonus = 0.02
    elif neighbor_rsrp >= -85:
        signal_bonus = 0.01
    elif neighbor_rsrp >= -95:
        signal_bonus = 0.0
    elif neighbor_rsrp >= -105:
        signal_bonus = -0.05
    else:
        signal_bonus = -0.15
    
    # SINR bonus/penalty
    if neighbor_sinr >= 15:
        sinr_bonus = 0.02
    elif neighbor_sinr >= 5:
        sinr_bonus = 0.01
    elif neighbor_sinr >= 0:
        sinr_bonus = 0.0
    elif neighbor_sinr >= -5:
        sinr_bonus = -0.03
    else:
        sinr_bonus = -0.10
    
    # Both cells at minimum power penalty
    if (serving_tx_power <= serving_min_power + 1 and 
        target_tx_power <= target_min_power + 1):
        base_success_prob = base_success_prob - 0.20
    
    # Congestion penalty
    if target_cpu_usage > 85 or target_prb_usage > 85:
        congestion_penalty = 0.05
        base_success_prob = base_success_prob - congestion_penalty
    
    # Final probability calculation
    final_success_prob = base_success_prob + signal_bonus + sinr_bonus
    final_success_prob = max(0.25, min(0.98, final_success_prob))
    
    return final_success_prob


@jit(nopython=True, cache=True)
def calculate_drop_probability(cell_tx_power, cell_min_power, cell_max_power,
                               ue_sinr, ue_rsrp, cell_cpu_usage, cell_prb_usage,
                               cell_current_load, cell_max_capacity):
    """
    Calculate drop probability for a UE based on multiple factors
    Matches MATLAB updateUEDropEvents logic exactly
    """
    drop_prob = 0.001  # Base drop probability (0.1%)
    
    # Power penalty - matches MATLAB thresholds
    power_range = cell_max_power - cell_min_power
    if cell_tx_power <= cell_min_power + power_range/4:
        power_penalty = (cell_max_power - (cell_min_power + power_range/4) + 1) * 0.08
        drop_prob = drop_prob + power_penalty
    elif cell_tx_power <= cell_min_power + power_range/2:
        power_penalty = (cell_max_power - (cell_min_power + power_range/2) + 1) * 0.03
        drop_prob = drop_prob + power_penalty
    
    # Signal quality factor - SINR
    if not np.isnan(ue_sinr):
        if ue_sinr < -10:
            drop_prob = drop_prob + 0.15 + abs(ue_sinr + 10) * 0.02
        elif ue_sinr < -5:
            drop_prob = drop_prob + 0.08 + abs(ue_sinr + 5) * 0.015
        elif ue_sinr < 0:
            drop_prob = drop_prob + 0.04
        elif ue_sinr < 5:
            drop_prob = drop_prob + 0.01
    
    # Signal quality factor - RSRP
    if not np.isnan(ue_rsrp):
        if ue_rsrp < -120:
            drop_prob = drop_prob + 0.12 + abs(ue_rsrp + 120) * 0.01
        elif ue_rsrp < -115:
            drop_prob = drop_prob + 0.06 + abs(ue_rsrp + 115) * 0.008
        elif ue_rsrp < -110:
            drop_prob = drop_prob + 0.03
    
    # Cell congestion factor - CPU
    if cell_cpu_usage > 95:
        drop_prob = drop_prob + (cell_cpu_usage - 95) * 0.015
    elif cell_cpu_usage > 90:
        drop_prob = drop_prob + (cell_cpu_usage - 90) * 0.01
    
    # Cell congestion factor - PRB
    if cell_prb_usage > 95:
        drop_prob = drop_prob + (cell_prb_usage - 95) * 0.012
    elif cell_prb_usage > 90:
        drop_prob = drop_prob + (cell_prb_usage - 90) * 0.008
    
    # Traffic load factor
    load_ratio = cell_current_load / max(1.0, cell_max_capacity)
    if load_ratio > 0.98:
        drop_prob = drop_prob + (load_ratio - 0.98) * 1.0
    elif load_ratio > 0.95:
        drop_prob = drop_prob + (load_ratio - 0.95) * 0.6
    elif load_ratio > 0.90:
        drop_prob = drop_prob + (load_ratio - 0.90) * 0.2
    
    # Critical condition penalty
    if cell_tx_power <= cell_max_power and (ue_rsrp < -110 or ue_sinr < -5):
        drop_prob = drop_prob + 0.25
    
    # Clamp to maximum 45%
    drop_prob = min(0.45, drop_prob)
    
    return drop_prob


@jit(nopython=True, cache=True)
def calculate_cell_cpu_usage(power_ratio, num_ues, load_ratio):
    """
    Calculate cell CPU usage based on power, UEs, and load
    Matches MATLAB updateCellResourceUsage
    """
    base_cpu = 10.0 + power_ratio * 5.0
    per_ue_cpu = num_ues * 2.5
    load_cpu = load_ratio * 50.0
    cpu_usage = min(95.0, base_cpu + per_ue_cpu + load_cpu)
    return cpu_usage


@jit(nopython=True, cache=True)
def calculate_cell_prb_usage(num_ues, load_ratio):
    """
    Calculate cell PRB usage based on UEs and load
    Matches MATLAB updateCellResourceUsage
    """
    base_prb = num_ues * 3.0
    load_prb = load_ratio * 60.0
    prb_usage = min(95.0, base_prb + load_prb)
    return prb_usage


@jit(nopython=True, cache=True)
def calculate_cell_energy_consumption(base_energy, tx_power, num_ues, load_ratio):
    """
    Calculate total cell energy consumption
    Matches MATLAB updateCellResourceUsage
    """
    # Convert dBm to watts for power consumption
    tx_power_watts = 10.0 ** ((tx_power - 30.0) / 10.0)
    per_ue_energy = num_ues * 15.0
    load_energy = load_ratio * 200.0
    total_energy = base_energy + tx_power_watts + per_ue_energy + load_energy
    return total_energy


@jit(nopython=True, cache=True)
def calculate_theoretical_drop_rate(base_drop_rate, tx_power, min_tx_power, 
                                   max_tx_power, cpu_usage, prb_usage, 
                                   avg_sinr, valid_sinr_count):
    """
    Calculate theoretical drop rate for a cell
    Matches MATLAB updateCellResourceUsage logic
    """
    drop_rate = base_drop_rate
    
    # Power drop penalty
    power_range = max_tx_power - min_tx_power
    if tx_power <= min_tx_power + power_range/4:
        power_drop_penalty = (max_tx_power - (min_tx_power + power_range/4) + 1) * 4.0
        drop_rate += power_drop_penalty
    elif tx_power <= min_tx_power + power_range/2:
        power_drop_penalty = (max_tx_power - (min_tx_power + power_range/2) + 1) * 1.5
        drop_rate += power_drop_penalty
    
    # Congestion factors - CPU
    if cpu_usage > 90:
        drop_rate += (cpu_usage - 90) * 0.4
    elif cpu_usage > 85:
        drop_rate += (cpu_usage - 85) * 0.2
    
    # Congestion factors - PRB
    if prb_usage > 90:
        drop_rate += (prb_usage - 90) * 0.3
    elif prb_usage > 85:
        drop_rate += (prb_usage - 85) * 0.15
    
    # Signal quality factor
    if valid_sinr_count > 0:
        if avg_sinr < 0:
            drop_rate += abs(avg_sinr) * 0.2
        elif avg_sinr < 5:
            drop_rate += (5 - avg_sinr) * 0.1
    
    # Clamp to maximum 25%
    drop_rate = min(25.0, drop_rate)
    
    return drop_rate


@jit(nopython=True, cache=True)
def calculate_cell_latency(base_latency, load_ratio, num_ues, 
                          tx_power, min_tx_power, max_tx_power, power_ratio):
    """
    Calculate cell average latency
    Matches MATLAB updateCellResourceUsage
    """
    latency = base_latency
    load_latency = load_ratio * 25.0
    ue_latency = min(15.0, num_ues * 0.8)
    
    # Power latency penalty
    power_range = max_tx_power - min_tx_power
    if tx_power <= min_tx_power + power_range/4:
        power_latency_penalty = (max_tx_power - (min_tx_power + power_range/4) + 1) * 15.0
    elif tx_power <= min_tx_power + power_range/2:
        power_latency_penalty = (max_tx_power - (min_tx_power + power_range/2) + 1) * 8.0
    else:
        power_latency_penalty = (1.0 - power_ratio) * 3.0
    
    latency = latency + load_latency + ue_latency + power_latency_penalty
    return latency

@jit(nopython=True, cache=True)
def process_measurements_batch(rsrp_matrix, tx_powers, min_tx_powers, 
                               rsrp_measurement_threshold):
    """
    Takes a matrix of RSRP values and calculates RSRQ and SINR matrices.
    This function contains all the per-pair logic that can be JIT-compiled.
    """
    num_ues, num_cells = rsrp_matrix.shape
    
    # Create empty matrices to store the results
    rsrq_matrix = np.full((num_ues, num_cells), np.nan, dtype=np.float64)
    sinr_matrix = np.full((num_ues, num_cells), np.nan, dtype=np.float64)

    for ue_idx in prange(num_ues):  # Use prange for potential parallel execution
        for cell_idx in range(num_cells):
            rsrp = rsrp_matrix[ue_idx, cell_idx]

            # Only process measurements that are strong enough
            if rsrp >= (rsrp_measurement_threshold - 5.0):
                # --- Replicate the RSRQ/SINR logic from the original numba_utils ---
                # RSSI calculation
                # Note: To avoid seeding issues inside a JIT function, we use small constants
                # for random variation. This is a common pattern for performance.
                rssi = rsrp + 10.0 * math.log10(12.0) + (np.random.randn() * 0.5)
        
                # RSRQ calculation (clamped between -20 and -3 dB)
                rsrq = 10.0 * math.log10(12.0) + rsrp - rssi
                rsrq = max(-20.0, min(-3.0, rsrq))
                
                # SINR calculation
                base_sinr = rsrp - (-110.0)  # Noise floor at -110 dBm
                sinr = base_sinr + (np.random.randn() * 2.0)
                
                # --- Replicate the SINR penalty logic from simulation_logic ---
                tx_power = tx_powers[cell_idx]
                min_tx_power = min_tx_powers[cell_idx]
                if tx_power <= min_tx_power + 2.0:
                    sinr_penalty = (min_tx_power + 2.0 - tx_power) * 6.0
                    sinr = sinr - sinr_penalty
                
                # Store the final results in the matrices
                rsrq_matrix[ue_idx, cell_idx] = rsrq
                sinr_matrix[ue_idx, cell_idx] = sinr
                
    return rsrq_matrix, sinr_matrix