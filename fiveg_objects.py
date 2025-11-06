import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# --- Data Classes for Simulation Events and Measurements ---

@dataclass
class NeighborMeasurement:
    """Represents a measurement of a neighbor cell."""
    cell_id: int
    rsrp: float
    rsrq: float
    sinr: float

@dataclass
class HandoverEvent:
    """Logs the details of a handover attempt."""
    ue_id: int
    cell_source: int
    cell_target: int
    rsrp_source: float
    rsrp_target: float
    rsrq_source: float
    rsrq_target: float
    sinr_source: float
    sinr_target: float
    a3_offset: float
    ttt: float
    ho_success: bool
    timestamp: float

# --- Core Simulation Objects ---

class UE:
    """
    Represents a User Equipment. Attributes are a 1:1 match with the MATLAB UE struct.
    """
    def __init__(self, **params: Dict[str, Any]):
        # Core Properties
        self.id: int = int(params['id'])
        self.x: float = float(params['x'])
        self.y: float = float(params['y'])
        self.velocity: float = float(params.get('velocity', 0.0))
        self.direction: float = float(params.get('direction', 0.0))
        self.mobility_pattern: str = params.get('mobility_pattern', 'pedestrian')
        self.deployment_scenario: str = params.get('deployment_scenario', 'dense_urban')

        # Network State
        self.serving_cell: Optional[int] = None
        self.rsrp: float = np.nan
        self.rsrq: float = np.nan
        self.sinr: float = np.nan
        self.neighbor_measurements: List[NeighborMeasurement] = []

        # Traffic & QoS State
        self.traffic_demand: float = 0.0
        self.is_dropped: bool = False
        self.drop_count: int = 0
        self.session_active: bool = False
        self.qos_latency: float = 0.0 # From MATLAB createUEStruct
        
        # Handover State
        self.ho_timer: float = 0.0
        self.handover_history: List[HandoverEvent] = []
        
        # Connection State Timers
        self.disconnection_timer: float = 0.0
        self.connection_timer: float = 0.0

        # Internal Mobility State
        self.pause_timer: float = 0.0
        self.rng_seed: int = int(params.get('rng_seed', 42))
        self.step_counter: int = 0
        self.last_direction_change: float = 0.0

        # Scenario-specific attributes for high-speed train
        self.in_train: bool = params.get('in_train', False)
        self.track_length: float = params.get('track_length', 0.0)
        self.train_start_x: float = params.get('train_start_x', 0.0)
        self.position_in_train: float = params.get('position_in_train', 0.0)
        
        # Scenario-specific attributes for highway
        self.on_highway: bool = params.get('on_highway', False)
        self.highway_length: float = params.get('highway_length', 0.0)
        self.num_lanes: int = params.get('num_lanes', 3)
        self.lane_width: float = params.get('lane_width', 3.5)
        self.lane: int = params.get('lane', 1)
        self.is_forward: bool = params.get('is_forward', True)

class Cell:
    """
    Represents a network cell. Attributes are a 1:1 match with the MATLAB Cell struct.
    """
    def __init__(self, **params: Dict[str, Any]):
        # Core Properties
        self.id: int = int(params['id'])
        self.site_id: int = int(params['site_id'])
        self.sector_id: int = int(params['sector_id'])
        self.x: float = float(params['x'])
        self.y: float = float(params['y'])
        self.frequency: float = float(params.get('frequency', 3.5e9))
        self.azimuth: float = float(params.get('azimuth', 0.0))
        self.antenna_height: float = float(params.get('antenna_height', 25.0))
        self.is_omnidirectional: bool = bool(params.get('is_omnidirectional', False))
        self.site_type: str = str(params.get('site_type', 'macro'))
        
        # Power Properties
        self.tx_power: float = float(params.get('tx_power', 46.0))
        self.min_tx_power: float = float(params.get('min_tx_power', 30.0))
        self.max_tx_power: float = float(params.get('max_tx_power', 46.0))
        
        # Energy Properties
        self.base_energy_consumption: float = float(params.get('base_energy_consumption', 1000.0))
        self.idle_energy_consumption: float = float(params.get('idle_energy_consumption', 250.0))
        self.energy_consumption: float = self.base_energy_consumption

        # Capacity & Load
        self.max_capacity: float = float(params.get('max_capacity', 250.0))
        self.current_load: float = 0.0
        self.cell_radius: float = float(params.get('cell_radius', 200.0))
        
        # Handover Parameters
        self.ttt: float = float(params.get('ttt', 0.008)) # Time To Trigger (seconds)
        self.a3_offset: float = float(params.get('a3_offset', 3.0)) # A3 Handover Offset (dB)

        # Dynamic Metrics
        self.cpu_usage: float = 0.0
        self.prb_usage: float = 0.0
        self.avg_latency: float = 10.0
        self.connected_ues: List[int] = []
        self.avg_sinr: float = 0.0

        # QoS Tracking (matches MATLAB struct)
        self.actual_drop_count: int = 0
        self.total_drop_events: int = 0
        self.total_connection_time: int = 0
        self.theoretical_drop_rate: float = 0.0
        self.drop_rate: float = 0.0

@dataclass
class Site:
    """Represents a physical site location which can host multiple cells."""
    id: int
    x: float
    y: float
    type: str

@dataclass
class SimulationParams:
    """Manages all simulation parameters, loaded from scenario configs."""
    name: str = "default"
    description: str = "Default simulation scenario"
    deploymentScenario: str = "urban_macro"
    layout: str = "hexagonal_grid"
    
    # Network Topology
    numSites: int = 7
    numSectors: int = 3
    isd: float = 200.0
    antennaHeight: float = 25.0
    cellRadius: float = 200.0
    max_radius: float = 500.0
    
    # RF Parameters
    carrierFrequency: float = 3.5e9
    systemBandwidth: float = 100e6
    
    # User Parameters
    numUEs: int = 210
    userDistribution: str = "Uniform/macro"
    ueSpeed: float = 3.0
    indoorRatio: float = 0.8
    outdoorSpeed: float = 30.0
    
    # Scenario Specific
    trainLength: float = 200.0
    trackLength: float = 10000.0
    highway_length: float = 10000.0
    num_lanes: int = 3
    lane_width: float = 3.5
    
    # Power Parameters
    minTxPower: float = 30.0
    maxTxPower: float = 46.0
    basePower: float = 800.0
    idlePower: float = 200.0
    
    # Simulation Time
    simTime: float = 600.0
    timeStep: float = 1.0
    
    # Thresholds & Triggers
    rsrpServingThreshold: float = -110.0
    rsrpTargetThreshold: float = -100.0
    rsrpMeasurementThreshold: float = -115.0
    dropCallThreshold: float = 1.0
    latencyThreshold: float = 50.0
    cpuThreshold: float = 80.0
    prbThreshold: float = 80.0
    
    # Traffic Model
    trafficLambda: float = 30.0
    peakHourMultiplier: float = 1.5
    
    # Derived Parameters
    total_steps: int = 600
    expected_cells: int = 21

    def __post_init__(self):
        self.total_steps = int(self.simTime / self.timeStep)
        self.expected_cells = self.numSites * self.numSectors
