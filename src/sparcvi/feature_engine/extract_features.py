# === extract_features.py ===
import os
import pandas as pd
import inspect
import re

from sparcvi.feature_engine.features import (
    time_features, phase_features, power_features,
    harmonic_features, frequency_features
)

# Mapping from category prefix to module
category_modules = {
    'time': time_features,
    'phase': phase_features,
    'power': power_features,
    'harmonic': harmonic_features,
    'frequency': frequency_features
}

# Normalization for feature function keys
def normalize_key(name):
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')

def extract_features(voltage, current, vabc=None, iabc=None, sampling_rate=1024):
    features_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'features_list.csv')
    df = pd.read_csv(features_path)
    results = {}

    for _, row in df.iterrows():
        feature_name = row['Feature Name']
        category = row['Category'].lower()
        key = normalize_key(feature_name)

        module = category_modules.get(category)
        if not module:
            print(f"[SKIP] Unknown category: {category} for feature: {feature_name}")
            results[feature_name] = float('nan')
            continue

        try:
            func = getattr(module, key)
            sig = inspect.signature(func)
            kwargs = {
                k: v for k, v in {
                    'voltage': voltage,
                    'current': current,
                    'vabc': vabc,
                    'iabc': iabc,
                    'sampling_rate': sampling_rate
                }.items() if k in sig.parameters
            }
            results[feature_name] = func(**kwargs)
        except AttributeError:
            print(f"[MISSING] {feature_name} → {key} not implemented in {module.__name__}")
            results[feature_name] = float('nan')
        except Exception as e:
            print(f"[ERROR] {feature_name} → {key} → {module.__name__} failed: {e}")
            results[feature_name] = float('nan')

    return results
