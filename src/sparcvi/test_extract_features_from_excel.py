# === test_extract_features_from_excel.py ===
import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress RuntimeWarnings from numpy (e.g., divide by zero, mean of empty slice)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Allow running directly without PYTHONPATH by adjusting sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_engine.extract_features import extract_features

# === Parameters ===
INPUT_PATH = 'C:/Users/mlopi/Projects/SparcVI-Core/src/sparcvi/data/test_gradual_phase_imbalance.csv'
OUTPUT_PATH = 'C:/Users/mlopi/Projects/SparcVI-Core/src/sparcvi/data/extracted_features_matrix.xlsx'
SAMPLING_RATE = 1024
CYCLE_SAMPLES = SAMPLING_RATE  # 1-cycle windows

# === Load Signal ===
df = pd.read_csv(INPUT_PATH)
columns = df.columns.str.lower()

# Validate format
if len(columns) not in [2, 6, 8]:
    raise ValueError("Unsupported file format. Expecting 2, 6, or 8 columns.")

time = df.iloc[:, 0].values

if len(columns) == 2:
    Va = df.iloc[:, 1].values
    Ia = np.zeros_like(Va)
    vabc = np.vstack([Va, Va, Va])
    iabc = np.vstack([Ia, Ia, Ia])
elif len(columns) == 6:
    Va, Vb, Vc, Ia, Ib, Ic = [df[col].values for col in df.columns[1:]]
    vabc = np.vstack([Va, Vb, Vc])
    iabc = np.vstack([Ia, Ib, Ic])
elif len(columns) == 8:
    Va, Vb, Vc, _, Ia, Ib, Ic = [df[col].values for col in df.columns[1:]]
    vabc = np.vstack([Va, Vb, Vc])
    iabc = np.vstack([Ia, Ib, Ic])

voltage = Va
current = Ia

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
