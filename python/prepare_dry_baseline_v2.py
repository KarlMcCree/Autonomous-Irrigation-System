
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def detect_and_remove_irrigation(df, soil_col,
                                 rise_threshold=2.0,
                                 window=3,
                                 decay_window=60,
                                 floor=5.0):
  
    soil = df[soil_col].values.astype(float).copy()
    n = len(soil)

    # Compare soil[i] to soil[i - window]: if rise > threshold → irrigation
    detected = []
    i = window
    while i < n:
        rise = soil[i] - soil[i - window]
        if rise > rise_threshold:
            detected.append(i)
            i += window  
        else:
            i += 1

    print(f"Detected {len(detected)} irrigation events "
          f"(rise > {rise_threshold} over {window} steps)")

    
    for idx in detected:
        # Estimate pre-jump baseline as median of the window before the jump
        pre_window = soil[max(0, idx - 30):idx]
        if len(pre_window) == 0:
            continue
        pre_baseline = np.median(pre_window)

        # Estimate drying slope from the pre-jump window
        if len(pre_window) >= 5:
            x = np.arange(len(pre_window))
            slope, _ = np.polyfit(x, pre_window, 1)
            if slope > 0:
                slope = -abs(slope)  # force drying
        else:
            slope = -0.01  # default per-step drying

        # Extend a monotone dry-down from pre_baseline
        for k in range(idx, min(idx + decay_window, n)):
            expected = pre_baseline + slope * (k - idx + 1)
            if expected < floor:
                expected = floor
            soil[k] = expected

    # Final light smoothing to remove jagged edges
    soil = pd.Series(soil).rolling(5, center=True,
                                  min_periods=1).mean().values

    df_out = df.copy()
    df_out[soil_col] = soil
    return df_out, len(detected)


def main():
    in_path = config.SIMULATION_CSV
    out_path = os.path.join(config.RAW_DIR, "bhopal_agritech_dry_v2.csv")

    if in_path == out_path:
        raise ValueError("in_path == out_path — refuse to overwrite original!")

    print(f" Input:  {in_path}")
    print(f" Output: {out_path}\n")

    df = pd.read_csv(in_path)
    print(f"   Rows: {len(df)}")

    # Detect soil column
    cols_lower = {c.lower(): c for c in df.columns}
    soil_col = next((cols_lower[k] for k in
                     ["soil_moisture", "soilmoisture", "moisture",
                      "soil_moisture_%"]
                     if k in cols_lower), None)
    if soil_col is None:
        raise ValueError(f"No soil column in {list(df.columns)}")

    # Aggressive detection
    df_dry, n_events = detect_and_remove_irrigation(
        df, soil_col,
        rise_threshold=1.5,  
        window=3,            
        decay_window=60,
        floor=5.0,
    )

    # Stats
    orig = df[soil_col].values
    dry  = df_dry[soil_col].values
    orig_jumps = (np.diff(orig) > 1.0).sum()
    dry_jumps  = (np.diff(dry)  > 1.0).sum()

    print(f"\n Removed {n_events} irrigation events")
    print(f"   Original: mean={orig.mean():.2f} std={orig.std():.2f} "
          f"jumps>1.0={orig_jumps} max_rise={np.diff(orig).max():.2f}")
    print(f"   Dry:      mean={dry.mean():.2f} std={dry.std():.2f} "
          f"jumps>1.0={dry_jumps} max_rise={np.diff(dry).max():.2f}")

    df_dry.to_csv(out_path, index=False)
    print(f"\n Saved: {out_path}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(orig[:2880], color="steelblue", alpha=0.8)
    axes[0].set_title(f"Original — mean={orig.mean():.2f}, "
                      f"max rise={np.diff(orig).max():.2f}")
    axes[0].grid(alpha=0.3)

    axes[1].plot(dry[:2880], color="darkred", alpha=0.8)
    axes[1].set_title(f"Dry baseline — mean={dry.mean():.2f}, "
                      f"max rise={np.diff(dry).max():.2f}")
    axes[1].set_xlabel("Simulated minute")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(config.DOCS_DIR, exist_ok=True)
    plot_path = os.path.join(config.DOCS_DIR, "dry_baseline_comparison.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f" Plot: {plot_path}")


if __name__ == "__main__":
    main()