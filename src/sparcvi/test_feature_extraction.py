import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
from sparcvi.feature_engine.extract_features import extract_features


# Create simulated waveforms
fs = 1024  # Sampling rate
t = np.linspace(0, 1, fs, endpoint=False)
voltage = np.sin(2 * np.pi * 50 * t)
current = 0.8 * np.sin(2 * np.pi * 50 * t + np.pi / 6)  # With phase shift

# Run extractor (uses src/sparcvi/config/features_list.csv)
results = extract_features(voltage, current, sampling_rate=fs)

# Print results
print("\nExtracted Features:\n" + "-"*30)
for feature, value in results.items():
    print(f"{feature:25} = {value:.4f}")
