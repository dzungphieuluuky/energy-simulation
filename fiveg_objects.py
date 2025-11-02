# fiveg_objects.py
import numpy as np
from typing import List, Dict, Any, Optional
class UE:
    """Represents a User Equipment in the simulation."""
    def __init__(self, params: Dict[str, Any]):
        self.id: int = int(params['id'])
        self.x: float = float(params['x'])
        self.y: float = float(params['y'])
        self.velocity: float = float(params.get('velocity', 0.0))
        self.direction: float = float(params.get('direction', 0.0))
        self.servingCell: Optional[int] = None
        self.rsrp: float = np.nan
        self.rsrq: float = np.nan
        self.sinr: float = np.nan
        self.trafficDemand: float = 0.0
        self.pauseTimer: float = 0.0
        self.lastDirectionChange: float = 0.0
        self.sessionActive: bool = True
        self.mobilityPattern: str = params.get('mobilityPattern', 'pedestrian')
        self.rngS: int = int(params.get('rngS', 42))
        # Scenario-specific attributes
        self.inTrain: bool = False
        self.trainStartX: float = 0.0
        self.trackLength: float = 0.0
        self.positionInTrain: float = 0.0

class Cell:
    """Represents a network cell in the simulation."""
    def __init__(self, params: Dict[str, Any]):
        self.id: int = int(params['id'])
        self.siteId: int = int(params['siteId'])
        self.sectorId: int = int(params['sectorId'])
        self.x: float = float(params['x'])
        self.y: float = float(params['y'])
        self.frequency: float = float(params.get('frequency', 3.5e9))
        self.txPower: float = float(params.get('txPower', 46.0))
        self.minTxPower: float = float(params.get('minTxPower', 30.0))
        self.maxTxPower: float = float(params.get('maxTxPower', 46.0))
        self.baseEnergyConsumption: float = float(params.get('baseEnergyConsumption', 1000.0))
        self.idleEnergyConsumption: float = float(params.get('idleEnergyConsumption', 250.0))
        self.maxCapacity: float = float(params.get('maxCapacity', 250.0))
        self.ttt: float = float(params.get('ttt', 8.0))
        self.a3Offset: float = float(params.get('a3Offset', 8.0))
        # Dynamic attributes
        self.cpuUsage: float = 0.0
        self.prbUsage: float = 0.0
        self.currentLoad: float = 0.0
        self.energyConsumption: float = self.baseEnergyConsumption
        self.dropRate: float = 0.0
        self.avgLatency: float = 0.0
        self.connectedUEs: List[UE] = []
        self.power_ratio: float = 1.0