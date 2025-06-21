# === test_extract_features_from_named_waveform.py ===
import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress RuntimeWarnings (e.g., divide by zero, mean of empty slice)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Allow running directly without PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_engine.extract_features import extract_features

# === Parameters ===
INPUT_PATH = 'C:/Users/mlopi/Projects/SparcVI-Core/src/sparcvi/data/AI test_Waveform_Event_10sec.csv'
OUTPUT_PATH = 'C:/Users/mlopi/Projects/SparcVI-Core/src/sparcvi/data/extracted_features_Event_matrix.xlsx'
SAMPLING_RATE = 1024
CYCLE_SAMPLES = SAMPLING_RATE

# === Load CSV File ===
df = pd.read_csv(INPUT_PATH)

# Column names expected
required_cols = [
    "Waveform Phase V1N", "Waveform Phase V2N", "Waveform Phase V3N",
    "Waveform Phase I1", "Waveform Phase I2", "Waveform Phase I3"
]

# Validate presence of required columns
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Extract 3-phase waveforms
Va = df["Waveform Phase V1N"].values
Vb = df["Waveform Phase V2N"].values
Vc = df["Waveform Phase V3N"].values

Ia = df["Waveform Phase I1"].values
Ib = df["Waveform Phase I2"].values
Ic = df["Waveform Phase I3"].values

vabc = np.vstack([Va, Vb, Vc])
iabc = np.vstack([Ia, Ib, Ic])

# === Sliding Window Feature Extraction ===
features_list = []
num_cycles = len(Va) // CYCLE_SAMPLES

for i in range(num_cycles):
    start = i * CYCLE_SAMPLES
    end = start + CYCLE_SAMPLES

    v_win = vabc[:, start:end]
    i_win = iabc[:, start:end]

    if np.any(np.isnan(v_win)) or np.any(np.isnan(i_win)):
        print(f"⚠️ Skipping cycle {i} due to NaNs.")
        continue

    if v_win.shape[1] < 4 or i_win.shape[1] < 4:
        print(f"⚠️ Skipping cycle {i} due to too few samples.")
        continue

    voltage = v_win[0]
    current = i_win[0]

    features = extract_features(voltage, current, vabc=v_win, iabc=i_win, sampling_rate=SAMPLING_RATE)
    features_list.append(features)
    print(f"✅ Extracted cycle {i+1}/{num_cycles}")

# === Save to Excel ===
df_out = pd.DataFrame(features_list)
df_out.to_excel(OUTPUT_PATH, index=False)
print(f"\n✅ Feature matrix saved to: {OUTPUT_PATH}")
