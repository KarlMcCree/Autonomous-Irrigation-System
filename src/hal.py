
from abc import ABC, abstractmethod
import os
import time
import numpy as np
import pandas as pd


class SensorHAL(ABC):
    @abstractmethod
    def read_soil_moisture(self) -> float: ...
    @abstractmethod
    def read_temperature(self) -> float: ...
    @abstractmethod
    def read_humidity(self) -> float: ...
    @abstractmethod
    def read_plant_image(self): ...


class ActuatorHAL(ABC):
    @abstractmethod
    def pump_on(self, duration_seconds: float) -> float: ...
    @abstractmethod
    def pump_off(self): ...
    @abstractmethod
    def get_total_water_used_ml(self) -> float: ...


class SimulatedHAL(SensorHAL, ActuatorHAL):

    # How long a watering event affects soil (decay time constant)
    WATER_DECAY_HOURS = 6.0
    # Max soil boost from a full watering (5 sec pump = 25 ml)
    # Tuned so a typical 25ml event raises soil by ~0.10
    WATER_BOOST_PER_ML = 0.004
    # Extra evaporation when temp > 25C (per degree per hour)
    EVAPORATION_PER_DEGREE = 0.0015

    def __init__(self,
                 csv_path: str,
                 noise_std: float = 0.02,
                 drift_per_hour: float = 0.001,
                 pump_ml_per_second: float = 5.0,
                 minutes_per_step: float = 1.0):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.idx = 0
        self.noise_std = noise_std
        self.drift_per_hour = drift_per_hour
        self.pump_ml_per_second = pump_ml_per_second
        self.minutes_per_step = minutes_per_step

        self.simulated_minutes = 0.0

        # Water tracking
        self.total_water_ml = 0.0
        self.pump_is_on = False

        # Water-balance state
        # Each entry: (sim_minute_watered, ml_added)
        # We sum their decaying contributions at each step
        self.irrigation_events = []

        # Cached current row
        self._current_row_cache = None
        self._current_temp_cache = None

        self._detect_columns()
        print(f" SimulatedHAL ready: {len(self.df)} rows from "
              f"{os.path.basename(csv_path)}")
        print(f"   Water-balance model: decay={self.WATER_DECAY_HOURS}h, "
              f"boost={self.WATER_BOOST_PER_ML}/ml")

    def _detect_columns(self):
        cols = {c.lower().strip(): c for c in self.df.columns}
        def find(cands):
            for c in cands:
                if c in cols:
                    return cols[c]
            return None
        self.col_soil = find(["soil_moisture", "soilmoisture", "moisture",
                              "soil_moisture_%", "sm"])
        self.col_temp = find(["temperature", "temp", "air_temperature",
                              "temp_c"])
        self.col_humid = find(["humidity", "humid", "rh"])
        if self.col_soil is None:
            raise ValueError(f"No soil column found in {list(cols)}")

    def advance(self):
        """Advance simulated clock by minutes_per_step."""
        self.simulated_minutes += self.minutes_per_step
        self._current_row_cache = None
        self._current_temp_cache = None
        self.idx += 1

    def get_simulated_time_hours(self) -> float:
        return self.simulated_minutes / 60.0

    def _row(self):
        if self._current_row_cache is None:
            self._current_row_cache = self.df.iloc[self.idx % len(self.df)]
        return self._current_row_cache

    def _irrigation_boost(self) -> float:
        """
        Compute the current soil-moisture boost from past irrigation events.

        Each event contributes a decaying amount:
            boost(t) = ml * WATER_BOOST_PER_ML * exp(-(t - t_watered) / TAU)

        Returns total boost at current simulated time.
        """
        now_min = self.simulated_minutes
        tau_min = self.WATER_DECAY_HOURS * 60.0
        total = 0.0
        for (t_watered, ml) in self.irrigation_events:
            dt = now_min - t_watered
            if dt < 0:
                continue
            total += ml * self.WATER_BOOST_PER_ML * np.exp(-dt / tau_min)
        return total

    def _evaporation_loss(self, temp_c: float, hours_elapsed: float) -> float:
        """
        Extra drying when it's hot. Baseline CSV already includes typical
        drying; this is ADDITIONAL loss for heat above 25°C.
        """
        excess_temp = max(0.0, temp_c - 25.0)
        return excess_temp * self.EVAPORATION_PER_DEGREE * hours_elapsed

    def read_soil_moisture(self) -> float:
        raw = float(self._row()[self.col_soil])
        base = raw / 100.0 if raw > 1.5 else raw

        # Apply irrigation boost from past events
        boost = self._irrigation_boost()

        # Apply evaporation penalty (hot days dry faster)
        temp = self.read_temperature()
        evap = self._evaporation_loss(temp, self.minutes_per_step / 60.0)

        # Combine
        value = base + boost - evap
        value += np.random.normal(0, self.noise_std)

        return float(np.clip(value, 0.0, 1.0))

    def read_temperature(self) -> float:
        if self.col_temp is None:
            return 25.0
        raw = float(self._row()[self.col_temp])
        return float(raw + np.random.normal(0, self.noise_std))

    def read_humidity(self) -> float:
        if self.col_humid is None:
            return 50.0
        raw = float(self._row()[self.col_humid])
        return float(np.clip(raw + np.random.normal(0, self.noise_std),
                             0.0, 100.0))

    def read_plant_image(self):
        return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    def pump_on(self, duration_seconds: float) -> float:
        ml = duration_seconds * self.pump_ml_per_second
        self.total_water_ml += ml
        self.pump_is_on = True
        # Record the event — it will boost soil moisture going forward
        self.irrigation_events.append((self.simulated_minutes, ml))
        return ml

    def pump_off(self):
        self.pump_is_on = False

    def get_total_water_used_ml(self) -> float:
        return self.total_water_ml

class RealHAL(SensorHAL, ActuatorHAL):
    """Unchanged — talks to ESP32 over serial."""
    def __init__(self, serial_port: str, baud: int = 115200, timeout: float = 2.0):
        import serial
        self.ser = serial.Serial(serial_port, baud, timeout=timeout)
        self.total_water_ml = 0.0
        time.sleep(2)
        print(f" RealHAL connected on {serial_port} @ {baud}")

    def advance(self):
        """No-op for real hardware — time advances naturally."""
        pass

    def get_simulated_time_hours(self):
        return time.time() / 3600.0

    def _cmd(self, command: str) -> str:
        self.ser.write(f"{command}\n".encode())
        return self.ser.readline().decode(errors="ignore").strip()

    def _parse(self, response: str) -> float:
        try:
            return float(response.split(":")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Bad ESP32 response: {response!r}")

    def read_soil_moisture(self): return self._parse(self._cmd("READ_MOISTURE"))
    def read_temperature(self):   return self._parse(self._cmd("READ_TEMP"))
    def read_humidity(self):      return self._parse(self._cmd("READ_HUMID"))

    def read_plant_image(self):
        return None

    def pump_on(self, duration_seconds: float) -> float:
        ml = self._parse(self._cmd(f"PUMP_ON:{duration_seconds}"))
        self.total_water_ml += ml
        return ml

    def pump_off(self): self._cmd("PUMP_OFF")
    def get_total_water_used_ml(self): return self.total_water_ml


def build_hal(mode: str = None):
    from . import config
    if mode is None:
        mode = config.HARDWARE_MODE
    if mode == "simulated":
        return SimulatedHAL(
            csv_path=config.SIMULATION_CSV,
            noise_std=config.SIMULATION_NOISE,
            drift_per_hour=config.SIMULATION_DRIFT,
        )
    elif mode == "real":
        return RealHAL(config.SERIAL_PORT, config.SERIAL_BAUD)
    raise ValueError(f"Unknown mode: {mode!r}")