# numba_utils.py

import numpy as np
import numba
from numba import int32, float64, jit
from typing import List
import math

from fiveg_objects import UE, Cell
# ------------------------------
# Lightweight Numba data containers
# ------------------------------
@numba.experimental.jitclass([
    ('id', int32),
    ('x', float64),
    ('y', float64),
    ('velocity', float64),
    ('direction', float64),
    ('servingCell', int32),
    ('rsrp', float64),
    ('rsrq', float64),
    ('sinr', float64),
    ('trafficDemand', float64),
    ('pauseTimer', float64),
    ('lastDirectionChange', float64)
])
class NumbaUE:
    def __init__(self):
        self.id = 0
        self.x = 0.0
        self.y = 0.0
        self.velocity = 0.0
        self.direction = 0.0
        self.servingCell = -1  # -1 means no serving cell
        self.rsrp = -200.0  # Default invalid value
        self.rsrq = -40.0   # Default invalid value
        self.sinr = -40.0   # Default invalid value
        self.trafficDemand = 0.0
        self.pauseTimer = 0.0
        self.lastDirectionChange = 0.0

@numba.experimental.jitclass([
    ('id', int32),
    ('x', float64),
    ('y', float64),
    ('frequency', float64),
    ('txPower', float64),
    ('minTxPower', float64),
    ('maxTxPower', float64),
    ('cpuUsage', float64),
    ('prbUsage', float64),
    ('maxCapacity', float64),
    ('currentLoad', float64),
    ('baseEnergyConsumption', float64),
    ('energyConsumption', float64),
    ('dropRate', float64),
    ('avgLatency', float64)
])
class NumbaCell:
    def __init__(self):
        self.id = 0
        self.x = 0.0
        self.y = 0.0
        self.frequency = 3.5e9
        self.txPower = 46.0
        self.minTxPower = 30.0
        self.maxTxPower = 46.0
        self.cpuUsage = 0.0
        self.prbUsage = 0.0
        self.maxCapacity = 250.0
        self.currentLoad = 0.0
        self.baseEnergyConsumption = 1000.0
        self.energyConsumption = 1000.0
        self.dropRate = 0.0
        self.avgLatency = 0.0

# Convert Python objects to Numba objects
def convert_ues_to_numba(ues: List[UE]) -> List[NumbaUE]:
    """Convert regular UE objects to Numba-optimized objects."""
    numba_ues = []
    for ue in ues:
        n_ue = NumbaUE()
        n_ue.id = ue.id
        n_ue.x = ue.x
        n_ue.y = ue.y
        n_ue.velocity = ue.velocity
        n_ue.direction = ue.direction
        n_ue.servingCell = ue.servingCell if ue.servingCell is not None else -1
        n_ue.rsrp = ue.rsrp if not np.isnan(ue.rsrp) else -200.0
        n_ue.rsrq = ue.rsrq if not np.isnan(ue.rsrq) else -40.0
        n_ue.sinr = ue.sinr if not np.isnan(ue.sinr) else -40.0
        n_ue.trafficDemand = ue.trafficDemand
        n_ue.pauseTimer = getattr(ue, 'pauseTimer', 0.0)
        n_ue.lastDirectionChange = getattr(ue, 'lastDirectionChange', 0.0)
        numba_ues.append(n_ue)
    return numba_ues

def convert_cells_to_numba(cells: List[Cell]) -> List[NumbaCell]:
    """Convert regular Cell objects to Numba-optimized objects."""
    numba_cells = []
    for cell in cells:
        n_cell = NumbaCell()
        n_cell.id = cell.id
        n_cell.x = cell.x
        n_cell.y = cell.y
        n_cell.frequency = cell.frequency
        n_cell.txPower = cell.txPower
        n_cell.minTxPower = cell.minTxPower
        n_cell.maxTxPower = cell.maxTxPower
        n_cell.cpuUsage = cell.cpuUsage
        n_cell.prbUsage = cell.prbUsage
        n_cell.maxCapacity = cell.maxCapacity
        n_cell.currentLoad = cell.currentLoad
        n_cell.baseEnergyConsumption = cell.baseEnergyConsumption
        n_cell.energyConsumption = cell.energyConsumption
        n_cell.dropRate = cell.dropRate
        n_cell.avgLatency = cell.avgLatency
        numba_cells.append(n_cell)
    return numba_cells

def convert_numba_to_regular(numba_ues: List[NumbaUE], numba_cells: List[NumbaCell],
                           original_ues: List[UE], original_cells: List[Cell]) -> None:
    """Convert Numba objects back to regular Python objects."""
    # Create mapping for quick lookup
    cell_map = {cell.id: cell for cell in original_cells}
    ue_map = {ue.id: ue for ue in original_ues}

    # Update UE objects
    for n_ue in numba_ues:
        if n_ue.id in ue_map:
            ue = ue_map[n_ue.id]
            ue.x = n_ue.x
            ue.y = n_ue.y
            ue.direction = n_ue.direction
            ue.servingCell = n_ue.servingCell if n_ue.servingCell != -1 else None
            ue.rsrp = n_ue.rsrp if n_ue.rsrp > -199.0 else np.nan
            ue.rsrq = n_ue.rsrq if n_ue.rsrq > -39.0 else np.nan
            ue.sinr = n_ue.sinr if n_ue.sinr > -39.0 else np.nan
            ue.trafficDemand = n_ue.trafficDemand
            ue.pauseTimer = n_ue.pauseTimer
            ue.lastDirectionChange = n_ue.lastDirectionChange

    # Update Cell objects
    for n_cell in numba_cells:
        if n_cell.id in cell_map:
            cell = cell_map[n_cell.id]
            cell.txPower = n_cell.txPower
            cell.cpuUsage = n_cell.cpuUsage
            cell.prbUsage = n_cell.prbUsage
            cell.currentLoad = n_cell.currentLoad
            cell.energyConsumption = n_cell.energyConsumption
            cell.dropRate = n_cell.dropRate
            cell.avgLatency = n_cell.avgLatency
            
            # Update power_ratio for metrics
            power_range = max(1e-6, cell.maxTxPower - cell.minTxPower)
            cell.power_ratio = (cell.txPower - cell.minTxPower) / power_range

@jit(nopython=True)
def numba_clip_angle(a: float) -> float:
    """Clip angle to [-π, π] range."""
    a = a % (2 * math.pi)
    if a > math.pi:
        a -= 2 * math.pi
    elif a < -math.pi:
        a += 2 * math.pi
    return a

@jit(nopython=True)
def numba_move_ue(ue: NumbaUE.class_type.instance_type, distance: float):
    """Move UE in its current direction."""
    ue.x += distance * math.cos(ue.direction)
    ue.y += distance * math.sin(ue.direction)