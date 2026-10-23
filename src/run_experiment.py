# src/run_experiment.py
from .config import HARDWARE_MODE, SIMULATION_CSV, SERIAL_PORT
from .hal import SimulatedHAL, RealHAL
from .controller import IrrigationController
from .predict import SoilPredictor


def get_hal():
    if HARDWARE_MODE == "simulated":
        return SimulatedHAL(SIMULATION_CSV)
    return RealHAL(SERIAL_PORT)


def run_policy(policy, days=7):
    hal = get_hal()
    soil_model = SoilPredictor.load()
    controller = IrrigationController(hal, hal, policy, soil_model)

    log = []
    for step in range(days * 24 * 60):  # 1-min steps
        decision = controller.decide()
        water = controller.execute(decision)
        log.append({**decision, "water_ml": water,
                    "total_ml": hal.get_total_water_used_ml()})
    return log