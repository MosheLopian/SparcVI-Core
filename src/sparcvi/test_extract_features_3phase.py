# === test_extract_features_3phase.py ===
import os
import sys
import numpy as np
from rich.console import Console
from rich.table import Table

# Allow direct run without PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_engine.extract_features import extract_features

# Simulation parameters
fs = 1024
f = 50  # Hz
cycles = 1
samples = fs * cycles
T = np.linspace(0, cycles / f, samples, endpoint=False)

# === Clean Signal ===
Va = np.sin(2 * np.pi * f * T)
Vb = np.sin(2 * np.pi * f * T - 2 * np.pi / 3)
Vc = np.sin(2 * np.pi * f * T + 2 * np.pi / 3)
Ia = 0.8 * np.sin(2 * np.pi * f * T + np.pi / 6)
Ib = 0.8 * np.sin(2 * np.pi * f * T - 2 * np.pi / 3 + np.pi / 6)
Ic = 0.8 * np.sin(2 * np.pi * f * T + 2 * np.pi / 3 + np.pi / 6)

vabc_clean = np.vstack([Va, Vb, Vc])
iabc_clean = np.vstack([Ia, Ib, Ic])
voltage_clean = Va
current_clean = Ia
features_clean = extract_features(voltage_clean, current_clean, vabc=vabc_clean, iabc=iabc_clean, sampling_rate=fs)

# === Faulty Signal: Phase Imbalance + 5th Harmonic on Vc ===
Va_f = Va
Vb_f = 0.9 * Vb
Vc_f = 0.7 * Vc + 0.1 * np.sin(2 * np.pi * 5 * f * T)  # Add 5th harmonic to Vc
Ia_f = Ia
Ib_f = 0.85 * Ib
Ic_f = 0.6 * Ic

vabc_fault = np.vstack([Va_f, Vb_f, Vc_f])
iabc_fault = np.vstack([Ia_f, Ib_f, Ic_f])
voltage_fault = Va_f
current_fault = Ia_f
features_fault = extract_features(voltage_fault, current_fault, vabc=vabc_fault, iabc=iabc_fault, sampling_rate=fs)

# === Report Comparison Table ===
console = Console()
table = Table(title="⚡ SPARCVI Feature Drift Report", show_lines=True)
table.add_column("Feature", style="bold cyan")
table.add_column("Clean", justify="right", style="green")
table.add_column("Faulty", justify="right", style="yellow")
table.add_column("Δ Delta", justify="right", style="magenta")

all_keys = sorted(set(features_clean.keys()) | set(features_fault.keys()))
for k in all_keys:
    c = features_clean.get(k, np.nan)
    f = features_fault.get(k, np.nan)
    d = f - c if isinstance(c, (int, float, np.float64)) and isinstance(f, (int, float, np.float64)) else np.nan
    table.add_row(
        k,
        f"{c:.4f}" if isinstance(c, (int, float, np.float64)) else str(c),
        f"{f:.4f}" if isinstance(f, (int, float, np.float64)) else str(f),
        f"{d:+.4f}" if isinstance(d, (int, float, np.float64)) else "nan"
    )

console.print(table)
