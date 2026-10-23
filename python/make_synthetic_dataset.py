# python/make_synthetic_dataset.py
"""Generate synthetic soil dataset if Kaggle download fails."""
import os
import numpy as np
import pandas as pd

os.makedirs("../data/raw", exist_ok=True)
np.random.seed(42)

N = 10000
soil, temp, humid = [0.6], [25.0], [55.0]
for i in range(1, N):
    hour = (i / 60) % 24
    # Temperature: sine wave day/night
    t = 25 + 8 * np.sin(2 * np.pi * (hour - 6) / 24) + np.random.normal(0, 0.3)
    # Humidity: inversely correlated with temperature
    h = 70 - 25 * np.sin(2 * np.pi * (hour - 6) / 24) + np.random.normal(0, 1)
    # Soil: dry-out (evaporation) + occasional watering
    dry_rate = 0.0008 * (1 + 0.03 * (t - 25))
    s = soil[-1] - dry_rate + np.random.normal(0, 0.002)
    if s < 0.25:  # simulate watering
        s = 0.7
    soil.append(np.clip(s, 0.05, 0.95))
    temp.append(t)
    humid.append(np.clip(h, 20, 95))

df = pd.DataFrame({
    "Timestamp": pd.date_range("2025-01-01", periods=N, freq="1min"),
    "Soil_Moisture": np.array(soil) * 100,
    "Temperature": temp,
    "Humidity": humid,
    "irrigation_status": [1 if s > 0.65 and soil[i-1] < 0.3 else 0
                          for i, s in enumerate(soil)],
})
df["soil_moisture_next_1m"] = df["Soil_Moisture"].shift(-1)
df = df.dropna()

out = "../data/raw/bhopal_agritech.csv"
df.to_csv(out, index=False)
print(f"   Synthetic dataset: {out}")
print(f"   Rows: {len(df)}")