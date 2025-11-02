import math
from typing import List, Dict, Any
import numpy as np
from fiveg_objects import UE, Cell
from simulation_logic import rng_for
# ------------------------------
# Site / topology / scenario helpers
# ------------------------------
def create_hex_layout(num_sites: int, isd: float, seed: int) -> List[Dict[str, Any]]:
    sites = []
    sites.append({'id': 1, 'x': 0.0, 'y': 0.0, 'type': 'macro'})
    if num_sites == 1:
        return sites
    idx = 2
    ring = 1
    rng = rng_for(seed)
    while idx <= num_sites and ring <= 5:
        for side in range(6):
            for pos in range(ring):
                if idx > num_sites:
                    break
                angle = side * math.pi / 3.0
                x = isd * ring * math.cos(angle) + pos * isd * math.cos(angle + math.pi / 3.0)
                y = isd * ring * math.sin(angle) + pos * isd * math.sin(angle + math.pi / 3.0)
                sites.append({'id': idx, 'x': x, 'y': y, 'type': 'macro'})
                idx += 1
            if idx > num_sites:
                break
        ring += 1
    while idx <= num_sites:
        a = rng.rand() * 2 * math.pi
        r = rng.rand() * isd * 2.0
        sites.append({'id': idx, 'x': r * math.cos(a), 'y': r * math.sin(a), 'type': 'macro'})
        idx += 1
    return sites

def configure_cells_from_sites(config: Dict[str, Any], sites: List[Dict[str, Any]]) -> List[Cell]:
    cell_list = []
    cid = 1
    cell_config = {
        'frequency': config.get('carrierFrequency', 3.5e9),
        'initialTxPower': config.get('maxTxPower', 46.0),
        'minTxPower': config.get('minTxPower', 30.0),
        'maxTxPower': config.get('maxTxPower', 46.0),
        'cellRadius': config.get('cellRadius', 300.0),
        'basePower': config.get('basePower', 1000.0),
        'idlePower': config.get('idlePower', 250.0),
        'maxCapacity': config.get('maxCapacity', 250)
    }
    for site in sites:
        num_sectors = 1 if config.get('deploymentScenario') == 'indoor_hotspot' else config.get('numSectors', 3)
        for sec in range(num_sectors):
            az = sec * (360.0 / max(1, num_sectors))
            d = {
                'id': cid,
                'siteId': site['id'],
                'sectorId': sec + 1,
                'x': site['x'],
                'y': site['y'],
                'frequency': cell_config['frequency'],
                'txPower': cell_config['initialTxPower'],
                'minTxPower': cell_config['minTxPower'],
                'maxTxPower': cell_config['maxTxPower'],
                'baseEnergyConsumption': cell_config['basePower'],
                'idleEnergyConsumption': cell_config['idlePower'],
                'maxCapacity': cell_config['maxCapacity'],
                'ttt': config.get('ttt', 8.0),
                'a3Offset': config.get('a3Offset', 8.0)
            }
            cell_list.append(Cell(d))
            cid += 1
    return cell_list

# ------------------------------
# UE initializers (kept simple & faithful to earlier translation)
# ------------------------------
def initialize_ues_from_config(config: Dict[str, Any], sites: List[Dict[str, Any]], seed: int) -> List[UE]:
    scen = config.get('deploymentScenario', 'indoor_hotspot')
    if scen == 'indoor_hotspot':
        return _init_indoor_hotspot(config, sites, seed)
    elif scen == 'dense_urban':
        return _init_dense_urban(config, sites, seed)
    elif scen == 'rural':
        return _init_rural(config, sites, seed)
    elif scen == 'high_speed':
        return _init_high_speed(config, sites, seed)
    elif scen == 'highway':
        return _init_highway(config, sites, seed)
    else:
        return _init_default(config, sites, seed)

def _init_indoor_hotspot(config, sites, seed):
    rng = rng_for(seed + 2000)
    num_ues = int(config.get('numUEs', 210))
    width = config.get('dimensions', {}).get('width', 120)
    height = config.get('dimensions', {}).get('height', 50)
    ues = []
    for uid in range(1, num_ues + 1):
        site = sites[rng.randint(0, len(sites))]
        angle = rng.rand() * 2.0 * math.pi
        r = min(width, height) * 0.5 * rng.rand()
        x = site['x'] + r * math.cos(angle)
        y = site['y'] + r * math.sin(angle)
        velocity = config.get('ueSpeed', 3.0) / 3.6
        direction = rng.rand() * 2.0 * math.pi
        ues.append(UE({'id': uid, 'x': x, 'y': y, 'velocity': velocity, 'direction': direction, 'mobilityPattern': 'indoor_pedestrian', 'rngS': seed + 1000 + uid}))
    return ues

def _init_dense_urban(config, sites, seed):
    rng = rng_for(seed + 2000)
    num_ues = int(config.get('numUEs', 210))
    indoor_ratio = config.get('indoorRatio', 0.8)
    indoor_count = int(round(num_ues * indoor_ratio))
    ues = []
    uid = 1
    for _ in range(indoor_count):
        site = sites[rng.randint(0, len(sites))]
        angle = rng.rand() * 2.0 * math.pi
        r = min(50.0, rng.rand() * 50.0)
        x = site['x'] + r * math.cos(angle)
        y = site['y'] + r * math.sin(angle)
        velocity = 3.0 / 3.6
        ues.append(UE({'id': uid, 'x': x, 'y': y, 'velocity': velocity, 'direction': rng.rand() * 2.0 * math.pi, 'mobilityPattern': 'indoor_pedestrian', 'rngS': seed + 1000 + uid}))
        uid += 1
    for _ in range(num_ues - indoor_count):
        site = sites[rng.randint(0, len(sites))]
        angle = rng.rand() * 2.0 * math.pi
        r = rng.rand() * config.get('isd', 200.0)
        x = site['x'] + r * math.cos(angle)
        y = site['y'] + r * math.sin(angle)
        velocity = 50.0 / 3.6
        ues.append(UE({'id': uid, 'x': x, 'y': y, 'velocity': velocity, 'direction': rng.rand() * 2.0 * math.pi, 'mobilityPattern': 'vehicle', 'rngS': seed + 1000 + uid}))
        uid += 1
    return ues

def _init_rural(config, sites, seed):
    rng = rng_for(seed + 2000)
    num_ues = int(config.get('numUEs', 210))
    max_r = config.get('maxRadius', 5000)
    ues = []
    for uid in range(1, num_ues + 1):
        angle = rng.rand() * 2.0 * math.pi
        r = rng.rand() * max_r
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        velocity = 80.0 / 3.6
        ues.append(UE({'id': uid, 'x': x, 'y': y, 'velocity': velocity, 'direction': rng.rand() * 2.0 * math.pi, 'mobilityPattern': 'vehicle', 'rngS': seed + 1000 + uid}))
    return ues

def _init_high_speed(config, sites, seed):
    num_ues = int(config.get('numUEs', 210))
    train_length = config.get('trainLength', 200)
    track_length = config.get('trackLength', 10000)
    train_start_x = - track_length / 2.0
    train_y = 0.0
    velocity = config.get('ueSpeed', 500.0) / 3.6
    rng = rng_for(seed + 2000)
    ues = []
    for uid in range(1, num_ues + 1):
        pos_in_train = (uid - 1) / max(1, num_ues - 1) * train_length
        x = train_start_x + pos_in_train
        y = train_y + (rng.rand() - 0.5) * 4.0
        u = UE({'id': uid, 'x': x, 'y': y, 'velocity': velocity, 'direction': 0.0, 'mobilityPattern': 'high_speed_train', 'rngS': seed + 1000 + uid})
        u.inTrain = True
        u.trainStartX = train_start_x
        u.trackLength = track_length
        u.positionInTrain = pos_in_train
        ues.append(u)
    return ues

def _init_highway(config, sites, seed):
    rng = rng_for(seed + 2000)
    num_ues = int(config.get('numUEs', 210))
    highway_length = config.get('highwayLength', 10000)
    num_lanes = config.get('numLanes', 3)
    lane_width = config.get('laneWidth', 3.5)
    ues = []
    for uid in range(1, num_ues + 1):
        lane = rng.randint(0, num_lanes)
        x = -highway_length / 2.0 + rng.rand() * highway_length
        y = (lane - (num_lanes - 1) / 2.0) * lane_width + (rng.rand() - 0.5) * 0.5
        velocity = config.get('ueSpeed', 100.0) / 3.6
        ues.append(UE({'id': uid, 'x': x, 'y': y, 'velocity': velocity, 'direction': 0.0, 'mobilityPattern': 'highway_vehicle', 'rngS': seed + 1000 + uid}))
    return ues

def _init_default(config, sites, seed):
    rng = rng_for(seed + 2000)
    num_ues = int(config.get('numUEs', 210))
    ues = []
    for uid in range(1, num_ues + 1):
        site = sites[rng.randint(0, len(sites))]
        angle = rng.rand() * 2.0 * math.pi
        r = rng.rand() * min(100.0, config.get('isd', 200.0))
        x = site['x'] + r * math.cos(angle)
        y = site['y'] + r * math.sin(angle)
        velocity = config.get('ueSpeed', 3.0) / 3.6
        ues.append(UE({'id': uid, 'x': x, 'y': y, 'velocity': velocity, 'direction': rng.rand() * 2.0 * math.pi, 'mobilityPattern': 'pedestrian', 'rngS': seed + 1000 + uid}))
    return ues
