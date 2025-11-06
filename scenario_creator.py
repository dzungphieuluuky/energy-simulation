"""
Scenario creation: sites, cells, UEs initialization.
Matches MATLAB scenario configurations and initialization scripts exactly.
"""

import numpy as np
import math
import json
from typing import List, Dict, Any
from fiveg_objects import Site, Cell, UE, SimulationParams


def load_scenario_config(scenario_name: str) -> SimulationParams:
    """Load scenario configuration from predefined MATLAB-equivalent functions."""
    return _get_predefined_scenario(scenario_name)

def _get_predefined_scenario(scenario_name: str) -> SimulationParams:
    """Get predefined scenario parameters, matching MATLAB defaults."""
    scenarios = {
        'indoor_hotspot': SimulationParams(name="indoor_hotspot", deploymentScenario="indoor_hotspot", numSites=12, numSectors=1, isd=30, antennaHeight=3, cellRadius=50, carrierFrequency=3.5e9, numUEs=80, ueSpeed=3, minTxPower=20, maxTxPower=30, basePower=400, idlePower=100, max_radius=100),
        'dense_urban': SimulationParams(name="dense_urban", deploymentScenario="dense_urban", numSites=7, numSectors=3, isd=200, antennaHeight=25, cellRadius=200, numUEs=210, ueSpeed=3, indoorRatio=0.8, outdoorSpeed=30, max_radius=500),
        'rural': SimulationParams(name="rural", deploymentScenario="rural", numSites=7, numSectors=3, isd=500, antennaHeight=35, cellRadius=1000, minTxPower=35, maxTxPower=49, basePower=1200, idlePower=300, numUEs=100, ueSpeed=60, max_radius=2000),
        'urban_macro': SimulationParams(name="urban_macro", deploymentScenario="urban_macro", numSites=7, numSectors=3, isd=300, antennaHeight=25, cellRadius=300, numUEs=250, ueSpeed=30, indoorRatio=0.8, max_radius=800),
        'high_speed': SimulationParams(name="high_speed", deploymentScenario="high_speed", numSites=10, numSectors=2, isd=1000, antennaHeight=35, cellRadius=1000, minTxPower=40, maxTxPower=49, basePower=1200, idlePower=300, numUEs=300, ueSpeed=500, trainLength=200, trackLength=10000, max_radius=1500),
        'extreme_rural': SimulationParams(name="extreme_rural", deploymentScenario="extreme_rural", numSites=3, numSectors=3, isd=5000, antennaHeight=45, cellRadius=50000, minTxPower=43, maxTxPower=49, basePower=1500, idlePower=400, numUEs=50, ueSpeed=120, max_radius=50000),
        'highway': SimulationParams(name="highway", deploymentScenario="highway", numSites=12, numSectors=3, isd=866, antennaHeight=35, cellRadius=866, minTxPower=40, maxTxPower=49, basePower=1200, idlePower=300, numUEs=200, ueSpeed=120, highway_length=10000, num_lanes=3, lane_width=3.5, max_radius=1732)
    }
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    return scenarios[scenario_name]

def create_sites(params: SimulationParams, seed: int) -> List[Site]:
    """Create site layout based on deployment scenario, matching MATLAB createLayout."""
    np.random.seed(seed + 1000)
    layout_funcs = {
        'indoor_hotspot': _create_indoor_layout,
        'dense_urban': lambda p, s: _create_hex_layout(p, s, 'urban_macro'),
        'rural': lambda p, s: _create_hex_layout(p, s, 'rural_macro'),
        'urban_macro': lambda p, s: _create_hex_layout(p, s, 'urban_macro'),
        'extreme_rural': lambda p, s: _create_hex_layout(p, s, 'extreme_rural_macro'),
        'high_speed': _create_high_speed_layout,
        'highway': _create_highway_layout,
    }
    func = layout_funcs.get(params.deploymentScenario, lambda p, s: _create_hex_layout(p, s, 'macro'))
    return func(params, seed)

def _create_indoor_layout(params: SimulationParams, seed: int) -> List[Site]:
    sites = []
    site_id = 1
    for row in range(1, 4):
        for col in range(1, 5):
            if site_id <= params.numSites:
                sites.append(Site(id=site_id, x=col * 24.0, y=row * 12.5, type='indoor_trxp'))
                site_id += 1
    return sites

def _create_hex_layout(params: SimulationParams, seed: int, site_type: str) -> List[Site]:
    sites = [Site(id=1, x=0, y=0, type=site_type)]
    if params.numSites == 1: return sites
    site_idx = 2
    ring = 1
    while site_idx <= params.numSites:
        angle_step = math.pi / 3
        for side in range(6):
            for pos_in_side in range(ring):
                if site_idx > params.numSites: break
                angle = side * angle_step
                x = ring * params.isd * math.cos(angle) + pos_in_side * params.isd * math.cos(angle + angle_step)
                y = ring * params.isd * math.sin(angle) + pos_in_side * params.isd * math.sin(angle + angle_step)
                sites.append(Site(id=site_idx, x=x, y=y, type=site_type))
                site_idx += 1
            if site_idx > params.numSites: break
        ring += 1
    return sites

def _create_high_speed_layout(params: SimulationParams, seed: int) -> List[Site]:
    start_pos = -(params.numSites - 1) * params.isd / 2.0
    return [Site(id=i + 1, x=start_pos + i * params.isd, y=100.0, type='high_speed_rrh') for i in range(params.numSites)]

def _create_highway_layout(params: SimulationParams, seed: int) -> List[Site]:
    spacing = params.highway_length / (params.numSites - 1)
    start_pos = -params.highway_length / 2.0
    return [Site(id=i + 1, x=start_pos + i * spacing, y=50.0 * ((i % 2) * 2 - 1), type='highway_macro') for i in range(params.numSites)]

def configure_cells(sites: List[Site], params: SimulationParams) -> List[Cell]:
    """Configure cells for all sites, matching MATLAB configureCells."""
    cells = []
    cell_id = 1
    for site in sites:
        num_sectors = {
            'indoor_hotspot': 1, 'high_speed': 2
        }.get(params.deploymentScenario, params.numSectors)
        
        config_func = {
            'indoor_trxp': _get_indoor_cell_config, 'high_speed_rrh': _get_high_speed_cell_config,
            'rural_macro': _get_rural_macro_cell_config, 'urban_macro': _get_urban_macro_cell_config,
            'extreme_rural_macro': _get_extreme_rural_cell_config, 'highway_macro': _get_highway_cell_config
        }.get(site.type, _get_macro_cell_config)
        
        cell_config = config_func(params)
        
        for sector_id in range(1, num_sectors + 1):
            azimuth = (sector_id - 1) * (360.0 / num_sectors)
            cells.append(Cell(
                id=cell_id, site_id=site.id, sector_id=sector_id,
                azimuth=azimuth, x=site.x, y=site.y,
                is_omnidirectional=(num_sectors == 1), site_type=site.type, **cell_config
            ))
            cell_id += 1
    return cells

def _get_common_params(p: SimulationParams) -> Dict[str, Any]:
    return {'frequency': p.carrierFrequency, 'min_tx_power': p.minTxPower, 'max_tx_power': p.maxTxPower}
def _get_indoor_cell_config(p: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_params(p), 'tx_power': 23, 'antenna_height': 3, 'cell_radius': 50, 'base_energy_consumption': 400, 'idle_energy_consumption': 100, 'max_capacity': 50, 'ttt': 0.004, 'a3_offset': 6}
def _get_macro_cell_config(p: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_params(p), 'tx_power': 43, 'antenna_height': 25, 'cell_radius': 200, 'base_energy_consumption': 800, 'idle_energy_consumption': 200, 'max_capacity': 200, 'ttt': 0.008, 'a3_offset': 8}
def _get_rural_macro_cell_config(p: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_params(p), 'tx_power': 46, 'antenna_height': 35, 'cell_radius': 1000, 'base_energy_consumption': 1200, 'idle_energy_consumption': 300, 'max_capacity': 150, 'ttt': 0.012, 'a3_offset': 10}
def _get_urban_macro_cell_config(p: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_params(p), 'tx_power': 43, 'antenna_height': 25, 'cell_radius': 300, 'base_energy_consumption': 1000, 'idle_energy_consumption': 250, 'max_capacity': 250, 'ttt': 0.008, 'a3_offset': 8}
def _get_high_speed_cell_config(p: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_params(p), 'tx_power': 46, 'antenna_height': 35, 'cell_radius': 1000, 'base_energy_consumption': 1200, 'idle_energy_consumption': 300, 'max_capacity': 300, 'ttt': 0.04, 'a3_offset': 3}
def _get_extreme_rural_cell_config(p: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_params(p), 'tx_power': 46, 'antenna_height': 45, 'cell_radius': 50000, 'base_energy_consumption': 1500, 'idle_energy_consumption': 400, 'max_capacity': 100, 'ttt': 0.016, 'a3_offset': 12}
def _get_highway_cell_config(p: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_params(p), 'tx_power': 46, 'antenna_height': 35, 'cell_radius': 866, 'base_energy_consumption': 1200, 'idle_energy_consumption': 300, 'max_capacity': 300, 'ttt': 0.04, 'a3_offset': 3}

def initialize_ues(params: SimulationParams, sites: List[Site], seed: int) -> List[UE]:
    """Initialize UEs based on deployment scenario, matching MATLAB initializeUEs."""
    np.random.seed(seed + 2000)
    init_funcs = {
        'indoor_hotspot': _initialize_indoor_ues, 'dense_urban': _initialize_dense_urban_ues,
        'rural': _initialize_rural_ues, 'urban_macro': _initialize_urban_macro_ues,
        'high_speed': _initialize_high_speed_ues, 'extreme_rural': _initialize_extreme_rural_ues,
        'highway': _initialize_highway_ues
    }
    func = init_funcs.get(params.deploymentScenario, _initialize_dense_urban_ues)
    return func(params, sites, seed)

def _create_ue_common(ue_id, x, y, mobility_pattern, velocity, seed, scenario):
    return UE(id=ue_id, x=x, y=y, mobility_pattern=mobility_pattern, velocity=velocity, 
              direction=np.random.random() * 2 * math.pi, rng_seed=seed + ue_id * 100, deployment_scenario=scenario)

def _initialize_indoor_ues(p, sites, seed):
    mobility = {'stationary': (0, 0.4), 'slow_walk': (0.5, 0.4), 'normal_walk': (1.5, 0.2)}
    patterns, props = zip(*mobility.items()); velocities, weights = zip(*props)
    ues = []
    for i in range(1, p.numUEs + 1):
        idx = np.random.choice(len(patterns), p=weights)
        x, y = 10 + np.random.random() * 100, 5 + np.random.random() * 40
        ues.append(_create_ue_common(i, x, y, patterns[idx], velocities[idx], seed, p.deploymentScenario))
    return ues

def _initialize_dense_urban_ues(p, sites, seed):
    ues = []
    num_indoor = int(round(p.numUEs * p.indoorRatio))
    for i in range(1, p.numUEs + 1):
        site = sites[np.random.randint(len(sites))]
        angle = np.random.random() * 2 * math.pi
        is_indoor = i <= num_indoor
        distance = abs(np.random.randn()) * 30 if is_indoor else 50 + np.random.random() * 100
        velocity = (p.ueSpeed if is_indoor else p.outdoorSpeed) / 3.6
        pattern = 'indoor_pedestrian' if is_indoor else 'outdoor_vehicle'
        x, y = site.x + distance * math.cos(angle), site.y + distance * math.sin(angle)
        ues.append(_create_ue_common(i, x, y, pattern, velocity, seed, p.deploymentScenario))
    return ues

def _initialize_rural_ues(p, sites, seed):
    mobility = {'stationary': (0, 0.1), 'pedestrian': (1.0, 0.4), 'slow_vehicle': (30/3.6, 0.3), 'fast_vehicle': (60/3.6, 0.2)}
    patterns, props = zip(*mobility.items()); velocities, weights = zip(*props)
    ues = []
    for i in range(1, p.numUEs + 1):
        if np.random.random() < 0.6: # Clustered
            site = sites[np.random.randint(len(sites))]
            angle, distance = np.random.random() * 2 * math.pi, np.random.random() * p.isd
            x, y = site.x + distance * math.cos(angle), site.y + distance * math.sin(angle)
        else: # Uniform
            angle = np.random.random() * 2 * math.pi
            radius = p.max_radius * math.sqrt(np.random.random())
            x, y = radius * math.cos(angle), radius * math.sin(angle)
        idx = np.random.choice(len(patterns), p=weights)
        ues.append(_create_ue_common(i, x, y, patterns[idx], velocities[idx], seed, p.deploymentScenario))
    return ues

def _initialize_urban_macro_ues(p, sites, seed):
    mobility = {'pedestrian': (1.5, 0.6), 'slow_vehicle': (15/3.6, 0.2), 'vehicle': (30/3.6, 0.2)}
    patterns, props = zip(*mobility.items()); velocities, weights = zip(*props)
    ues = []
    for i in range(1, p.numUEs + 1):
        site = sites[np.random.randint(len(sites))]
        angle = np.random.random() * 2 * math.pi
        distance = abs(np.random.randn()) * (p.cellRadius * 0.3) if np.random.random() < p.indoorRatio else p.cellRadius * math.sqrt(np.random.random())
        idx = np.random.choice(len(patterns), p=weights)
        x, y = site.x + distance * math.cos(angle), site.y + distance * math.sin(angle)
        ues.append(_create_ue_common(i, x, y, patterns[idx], velocities[idx], seed, p.deploymentScenario))
    return ues

def _initialize_high_speed_ues(p, sites, seed):
    start_x = -p.trackLength / 2.0
    ues = []
    for i in range(1, p.numUEs + 1):
        pos_in_train = np.random.random() * p.trainLength
        x = start_x + pos_in_train
        y = (np.random.random() - 0.5) * 4
        ue = UE(id=i, x=x, y=y, velocity=p.ueSpeed/3.6, direction=0, mobility_pattern='high_speed_train',
                rng_seed=seed + i * 100, deployment_scenario=p.deploymentScenario, in_train=True,
                track_length=p.trackLength, train_start_x=start_x, position_in_train=pos_in_train)
        ues.append(ue)
    return ues

def _initialize_extreme_rural_ues(p, sites, seed):
    mobility = {'stationary': (0, 0.2), 'slow_vehicle': (60/3.6, 0.3), 'fast_vehicle': (120/3.6, 0.3), 'extreme_vehicle': (160/3.6, 0.2)}
    patterns, props = zip(*mobility.items()); velocities, weights = zip(*props)
    ues = []
    for i in range(1, p.numUEs + 1):
        angle, radius = np.random.random() * 2 * math.pi, p.max_radius * math.sqrt(np.random.random())
        idx = np.random.choice(len(patterns), p=weights)
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        ues.append(_create_ue_common(i, x, y, patterns[idx], velocities[idx], seed, p.deploymentScenario))
    return ues

def _initialize_highway_ues(p, sites, seed):
    ues = []
    num_forward = int(math.ceil(p.numUEs / 2.0))
    for i in range(1, p.numUEs + 1):
        is_forward = i <= num_forward
        lane = np.random.randint(1, p.num_lanes + 1)
        x = np.random.random() * p.highway_length - p.highway_length / 2.0
        y = ((lane - 0.5) * p.lane_width) * (1 if is_forward else -1)
        velocity = p.ueSpeed/3.6 + (np.random.random() - 0.5) * 20
        ue = UE(id=i, x=x, y=y, velocity=velocity, direction=0 if is_forward else math.pi, mobility_pattern='highway_vehicle',
                rng_seed=seed + i * 100, deployment_scenario=p.deploymentScenario, on_highway=True,
                highway_length=p.highway_length, num_lanes=p.num_lanes, lane_width=p.lane_width, lane=lane, is_forward=is_forward)
        ues.append(ue)
    return ues