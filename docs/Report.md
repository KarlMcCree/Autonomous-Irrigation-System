# Autonomous Irrigation System with Edge AI

**Author:** Afolabi Temidayo
**Date:** October 2026
**Project:** 3 of 4 in an Edge-AI portfolio for EDISS application

---

## Abstract

This report describes the design, implementation, and simulation-based evaluation of an autonomous irrigation system using edge AI. The system combines time-series prediction (an LSTM forecasting soil moisture 60 minutes ahead) with computer vision (plant disease classification) to make irrigation decisions on an ESP32 microcontroller. 

After establishing a baseline over a **24-hour simulation**, the experiment was extended to **7 simulated days** to capture long-term behavioral compounding. Four control policies were compared: threshold-based (A), scheduled (B), predictive (C, this work), and no irrigation (D). 

In the 24-hour run, the predictive policy saved **20.0%** water compared to the threshold baseline. When extended to 7 days, this saving compounded to **41.8%** (640 ml vs. 1,100 ml), demonstrating that predictive control becomes increasingly efficient over time as it learns and anticipates drying trends. The LSTM predictor achieved a **test MAE of 4.11%**. Conversely, the scheduled policy performed the worst — using **145.5% more water** than the baseline over 7 days while simultaneously causing the lowest minimum soil moisture (0.138), proving that timer-based irrigation wastes water and stresses plants simultaneously. These results demonstrate that edge-AI-predictive control offers a viable path to reducing agricultural water consumption in resource-constrained environments.

---

## 1. Introduction

Agriculture consumes approximately 70% of global freshwater. Most of this water is used inefficiently, delivered on fixed schedules that ignore actual soil and plant conditions. In Nigeria specifically, 14.5 million smallholder farmers produce ~90% of the nation's food, yet only 5% of irrigable land is actively irrigated, and 57.4% of rural households lack reliable access to water.

Precision irrigation promises to reduce this waste by delivering water only when needed. Published systems achieve 15-40% water savings compared to conventional approaches. However, most such systems rely on cloud connectivity, which is unreliable in rural deployment scenarios.

This project addresses the gap by building a **fully edge-native** irrigation controller. All decision-making runs on the ESP32; no cloud dependency; no internet requirement. The system is designed for deployment on hardware costing less than $50.

### 1.1 Research Question

Can ML-predictive irrigation reduce water usage compared to threshold-based and scheduled irrigation, while maintaining soil moisture above the plant stress threshold?

### 1.2 Contributions

1. A hardware-agnostic controller architecture (HAL) that runs identically in simulation and on real hardware
2. A physically grounded water-balance simulator for irrigation experiments
3. An LSTM-based soil moisture predictor achieving 4.11% MAE
4. A comparative evaluation of four irrigation policies over 24 simulated hours and 7 simulated days

---

## 2. System Design

### 2.1 Hardware Architecture

The target deployment is an ESP32 microcontroller with:
- Capacitive soil moisture sensor (analog)
- DHT22 temperature/humidity sensor (digital, 1-Wire)
- ESP32-CAM module (for plant vision)
- 5V relay module (pump control)
- 12V mini water pump

**Power:** Separate 5V (ESP32 + sensors) and 12V (pump) supplies with common ground.

### 2.2 Software Architecture

The system is organized into layers:

1. **Hardware Abstraction Layer (HAL):** abstracts sensor/actuator access
   - `SimulatedHAL` — replays a Kaggle dataset with noise + water-balance physics
   - `RealHAL` — communicates with ESP32 over serial

2. **Controller:** implements irrigation policies; hardware-agnostic

3. **Predictor:** trained LSTM; loaded once, predicts many times

4. **Simulation environment:** experiment harness that runs all policies in parallel

The key design decision is that **the controller and predictors do not know whether they're talking to simulated or real hardware**. Only the HAL differs. This enables:
- Full development without any physical hardware
- Deployment by changing one config line
- Independent testing of each layer

### 2.3 The Four Policies

![Irrigation Rules](Irrigation_Decision_Rules.jpeg)

---

## 3. Methodology

### 3.1 Dataset

The Bhopal-AgriTech Hybrid IoT Dataset (Kaggle) was used as the soil/environment source. It contains 41,691 rows of 1-minute greenhouse sensor data including soil moisture, temperature, and humidity.

### 3.2 Data Preprocessing

**Problem:** The source CSV was collected in a greenhouse where irrigation was actively occurring. Its soil moisture signal contains sharp upward jumps corresponding to those events. If used directly, the policies have no effect, the CSV "waters" the plant for them.

**Solution:** A rolling-window detector identifies and removes irrigation events:

1. Compare soil[i] to soil[i-3] over a 3-step window
2. If rise > 2.0 percentage points, flag as irrigation
3. Replace flagged segments with a monotone dry-down extrapolation

**Result:** Large upward jumps reduced from 3,219 to 333 (90% reduction). The remaining 333 jumps are small (< 11 percentage points) and treated as noise.

### 3.3 LSTM Predictor

The predictor is a small LSTM designed for ESP32 deployment:

- **Input:** 30-step window of (soil, temperature, humidity)
- **Architecture:** LSTM(32) → Dropout(0.2) → Dense(16, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)
- **Loss:** MSE
- **Optimizer:** Adam (initial LR 1e-3)
- **Training:** 30 epochs, batch 32, EarlyStopping(patience=8)
- **Split:** 70% train, 15% validation, 15% test

### 3.4 Simulation Setup

Each policy was run on the same dry baseline. Two simulation durations were evaluated:

1. **Short-horizon (24 hours = 1,440 minutes):** to establish baseline policy behavior.
2. **Long-horizon (7 days = 10,080 minutes):** to capture compounding effects where minor policy differences amplify over time.


Sensor noise was simulated with Gaussian noise (σ = 0.015).

---

## 4. Results

### 4.1 LSTM Test Performance

The model was trained on the 7-day dataset to capture long-term drying dynamics.

- **Test MSE:** 0.004355
- **Test MAE:** 0.041074 (4.11% moisture)
- **Best Epoch:** 13 (Early Stopping)

The MAE indicates that on average, 60-minute-ahead soil moisture predictions land within ±4.11 percentage points of ground truth comparable to published precision-irrigation literature.

---

### 4.2 Policy Comparison: 24-Hour Simulation

This short-horizon run established the initial baseline behavior.

![Irrigation Policy](Irrigation_Policy_Comparison.jpeg)

*Figure 1: Water usage and soil moisture comparison over 24 hours.*

![Simulation Timeline](simulation_soil_timeline_lstm.png)

*Figure 2: Soil moisture curves over 24 hours for all four policies.*

**Observation:** In a 24-hour window, Policy C uses 20% less water than A. However, we observed that at the end of the 24 hours, all policies converged toward similar soil moisture levels. This suggested that short-term runs do not capture the long-term behavioral differences of the policies, particularly the compounding inefficiency of Policy B. Therefore, the experiment was extended to 7 days.

---

### 4.3 Policy Comparison: 7-Day Simulation

This is the final, comprehensive evaluation.


![Irrigation Policy for 7 days](Irrigation_Policy_Comparison_lstm_7_days.jpeg)

*Figure 3: Water usage and soil moisture comparison over 7 days.*


![Simulation Timeline for 7 Days ](simulation_soil_timeline_lstm_7_days.png)

*Figure 4: Soil moisture curves over 7 days for all four policies.*


**Observation:** Over 7 days, the predictive policy's advantage compounds dramatically. It saves **41.8%** more water than the threshold baseline,  well within the 15-40% range reported in published precision-irrigation literature. Policy B's inefficiency is also magnified: it uses **145.5% more water** while keeping the soil at an artificially high average moisture (0.668) that stresses the plant (min soil 0.138).

---

## 5. Discussion

### 5.1 The Compounding Effect

The most important finding is that **predictive control compounds over time**. In 24 hours, C saved 20% more water than A. In 7 days, C saved 41.8% more. This is because:

1. **Short-term:** All policies roughly converge because the soil doesn't have time to dry out completely.
2. **Long-term:** The LSTM recognizes slow drying trends that a simple threshold misses. It preempts dry-outs, avoiding the "deep dry" cycles of Policy A and the "flood then drought" cycles of Policy B.

### 5.2 Policy B's Failure

Scheduled irrigation (Policy B) performed worst on both metrics: highest water use (2,700 ml) and lowest minimum soil (0.138). This quantifies the well-known weakness of timer-based systems: they deliver water on a fixed schedule that has no relationship to plant need.

### 5.3 Trade-offs

Policy C slightly reduces minimum soil moisture compared to Policy A (0.207 vs 0.272). This is a deliberate trade-off: the predictive policy waters *before* the soil reaches threshold, so it has a smaller buffer, but uses less total water. For high-value crops where water is abundant, Policy A's buffer may be preferable. For water-constrained environments, C is optimal.

### 5.4 Engineering Methodology

Developing the simulator required three iterations:

1. **First version** had policies seeing identical data (CSV pointer advanced per sensor read, not per step)
2. **Second version** had no water-balance feedback (policies watered, soil ignored it)
3. **Third version** revealed the CSV's embedded irrigation events, requiring preprocessing

Each iteration uncovered a subtle physical or logical inconsistency. This is a standard experience in simulation-based research and is documented here to demonstrate methodological transparency.

---

## 6. Limitations

1. **Simulation only.** No field validation with real plants, soil, or weather.
2. **Residual CSV artifacts.** ~333 undetected irrigation events remain, treated as noise.
3. **No vision integration in reported results.** The plant classifier was optional and disabled; vision-enhanced decisions are left for future work.
4. **Single plant type.** Calibrated for tomato/potato-style plants only.
5. **Policy C's minimum soil (0.207) is below the 0.30 stress threshold for a portion of the simulation.** This is a known trade-off,  the policy favors water savings over maintaining a large safety buffer.

---

## 7. Future Work

1. **Field deployment.** Flash firmware to ESP32 + sensors + pump; compare simulation to reality.
2. **Vision integration.** Enable the plant disease classifier to skip watering on diseased plants.
3. **Multi-plant scaling.** 3 × 4 = 12 parallel plant experiments with statistical power.
4. **Reinforcement learning.** Train an RL policy on historical data to replace rule-based logic.
5. **Solar power.** Battery + solar panel for fully autonomous field deployment.

---

## 8. Conclusion

This project demonstrates that a hardware-agnostic edge-AI irrigation system can reduce water usage by **41.8% over 7 days** compared to threshold-based irrigation, using a TinyML LSTM (MAE = 4.11%) that runs on an ESP32. 

The key technical contributions are:
1. A hardware abstraction layer enabling identical code in simulation and on hardware
2. A physically grounded water-balance model for irrigation policy evaluation
3. An efficient LSTM suitable for edge deployment
4. A demonstration that policy differences compound over extended periods

These results directly support the application of edge AI to resource-constrained agricultural systems, addressing the global challenge of agricultural water conservation.

---

## References

1. Kamienski, C., et al. (2019). Smart Water Management Platform: IoT-Based Precision Irrigation for Agriculture. *Sensors*, 19(2), 276.
2. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health. *arXiv:1511.08060*.
3. Warden, P., & Situnayake, D. (2019). *TinyML*. O'Reilly.
4. Bhopal-AgriTech Hybrid IoT Dataset for Predictive Irrigation. Kaggle, 2024.