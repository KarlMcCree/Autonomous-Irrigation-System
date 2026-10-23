import numpy as np

from collections import deque

from .hal import SensorHAL, ActuatorHAL


class IrrigationController:
    def __init__(self,
                 sensor_hal: SensorHAL,
                 actuator_hal: ActuatorHAL,
                 policy: str = "C",
                 soil_model=None,
                 plant_model=None):
        from . import config

        self.sensor = sensor_hal
        self.actuator = actuator_hal
        self.policy = policy.upper()
        self.soil_model = soil_model
        self.plant_model = plant_model
        self.cfg = config

        self.history = deque(maxlen=config.WINDOW_SIZE)

        # Time bookkeeping — all in SIMULATED hours
        self.last_water_time_h = 0.0
        self.sim_start_h = None
        self.last_water_time_any_h = -1e9
        self.day_start_h = 0.0
        self.water_today_ml = 0.0

        if self.policy not in ("A", "B", "C", "D"):
            raise ValueError(f"Unknown policy: {policy!r}")

    def _now_hours(self) -> float:
        
        return self.sensor.get_simulated_time_hours()

    def step(self, water_duration_sec: float = 5.0) -> dict:
        now_h = self._now_hours()

        # Reset daily counter every 24 simulated hours
        if now_h - self.day_start_h >= 24.0:
            self.water_today_ml = 0.0
            self.day_start_h = now_h

        # Read sensors
        soil = self.sensor.read_soil_moisture()
        temp = self.sensor.read_temperature()
        humid = self.sensor.read_humidity()

        self.history.append([soil, temp, humid])

        record = {
            "sim_hours": now_h,
            "soil": soil,
            "temp": temp,
            "humidity": humid,
            "policy": self.policy,
            "action": "wait",
            "reason": "",
            "water_ml": 0.0,
            "total_ml": self.actuator.get_total_water_used_ml(),
        }

        # Apply policy
        if self.policy == "A":
            self._policy_threshold(record)
        elif self.policy == "B":
            self._policy_scheduled(record, now_h)
        elif self.policy == "C":
            self._policy_predictive(record, now_h)
        elif self.policy == "D":
            record["reason"] = "no irrigation (baseline)"    

        # Safety
        if record["action"] == "water" and not self._safety_ok(now_h):
            record["action"] = "blocked"
            record["reason"] += " [SAFETY]"

        # Execute
        if record["action"] == "water":
            ml = self.actuator.pump_on(water_duration_sec)
            self.actuator.pump_off()
            record["water_ml"] = ml
            record["total_ml"] = self.actuator.get_total_water_used_ml()
            self.water_today_ml += ml
            self.last_water_time_any_h = now_h
            if self.policy == "B":
                self.last_water_time_h = now_h

        # Advance simulated clock AFTER the step
        self.sensor.advance()

        if not hasattr(self, "decisions"):
            self.decisions = []
        self.decisions.append(record)

        return record

    def _policy_threshold(self, record):
        thr = self.cfg.POLICY_A_THRESHOLD
        if record["soil"] < thr:
            record["action"] = "water"
            record["reason"] = f"soil {record['soil']:.2f} < {thr:.2f}"
        else:
            record["reason"] = f"soil {record['soil']:.2f} >= {thr:.2f}"

    def _policy_scheduled(self, record, now_h: float):
        interval = self.cfg.POLICY_B_INTERVAL_HOURS
        hours_since = now_h - self.last_water_time_h
        if hours_since >= interval:
            record["action"] = "water"
            record["reason"] = f"scheduled ({hours_since:.1f}h)"
        else:
            record["reason"] = f"next in {interval - hours_since:.1f}h"

    def _policy_predictive(self, record, now_h: float):

        # ── Condition 1: enough history ──
        if len(self.history) < self.cfg.WINDOW_SIZE:
            record["reason"] = (f"warming up "
                                f"({len(self.history)}/{self.cfg.WINDOW_SIZE})")
            return

        # ── Compute predicted soil ──
        predicted = None

        if self.soil_model is not None:
            # Use trained LSTM
            try:
                predicted = self.soil_model.predict(list(self.history))
            except Exception as e:
                record["reason"] = f"predict error: {e}"
                return
        else:
            # Fallback: linear extrapolation of recent trend
            recent = [h[0] for h in list(self.history)[-15:]]
            x = np.arange(len(recent))
            slope = float(np.polyfit(x, recent, 1)[0])

            # Require meaningful drying before predicting danger
            MIN_DRYING_SLOPE = -0.002
            if slope > MIN_DRYING_SLOPE:
                record["reason"] = f"soil trend stable (slope={slope:.4f})"
                return

            horizon_steps = self.cfg.PREDICT_HORIZON
            predicted = recent[-1] + slope * horizon_steps
            predicted = float(max(0.0, min(1.0, predicted)))

        # Safety: if somehow predicted is still None, bail out
        if predicted is None:
            record["reason"] = "prediction unavailable"
            return

        record["predicted_soil"] = predicted
        current_soil = record["soil"]

        # ── Condition 2: current-soil guard ──
        # Stricter: require soil to be NEAR the threshold before we even consider
        SAFE_CEILING = 0.35    # don't consider watering if current soil > 0.35
        if current_soil > SAFE_CEILING:
            record["reason"] = f"current {current_soil:.2f} > {SAFE_CEILING} (buffer)"
            return
        
        # ── Vision check ──
        plant_ok = True
        plant_status = "unknown"
        if self.plant_model is not None:
            img = self.sensor.read_plant_image()
            if img is not None:
                try:
                    result = self.plant_model.predict_array(img)
                    plant_status = result["display_name"]
                    if result["class"] in self.cfg.NO_WATER_CLASSES:
                        plant_ok = False
                except Exception:
                    plant_ok = True
        record["plant_status"] = plant_status

        if not plant_ok:
            record["reason"] = f"plant diseased ({plant_status})"
            return

        # ── Condition 3: prediction below threshold ──
        if predicted < self.cfg.POLICY_C_PREDICT_THRESH:
            record["action"] = "water"
            record["reason"] = (f"predicted {predicted:.2f} < "
                                f"{self.cfg.POLICY_C_PREDICT_THRESH:.2f} "
                                f"(current={current_soil:.2f})")
        else:
            record["reason"] = f"predicted {predicted:.2f} ok"

    def _safety_ok(self, now_h: float) -> bool:
        if self.water_today_ml >= self.cfg.MAX_WATER_PER_DAY_ML:
            return False
        hours_since = now_h - self.last_water_time_any_h
        if hours_since < self.cfg.MIN_INTERVAL_BETWEEN_ML / 60.0:
            return False
        return True

    def summary(self) -> dict:
        
        if not hasattr(self, "decisions") or not self.decisions:
            return {
                "policy": self.policy,
                "steps": 0,
                "water_events": 0,
                "total_water_ml": self.actuator.get_total_water_used_ml(),
                "avg_soil": 0.0,
                "min_soil": 0.0,
            }

        watered = [d for d in self.decisions if d["action"] == "water"]
        soils = [d["soil"] for d in self.decisions]

        return {
            "policy": self.policy,
            "steps": len(self.decisions),
            "water_events": len(watered),
            "total_water_ml": self.actuator.get_total_water_used_ml(),
            "avg_soil": float(sum(soils) / len(soils)),
            "min_soil": float(min(soils)),
        }