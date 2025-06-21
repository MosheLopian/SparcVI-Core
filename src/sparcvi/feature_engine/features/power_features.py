# === power_features.py ===
import numpy as np


def real_power_3_phase(vabc, iabc, sampling_rate=1024, **kwargs):
    try:
        return sum(np.mean(np.real(v * np.conj(i))) for v, i in zip(vabc, iabc))
    except Exception:
        return np.nan


def reactive_power_3_phase(vabc, iabc, sampling_rate=1024, **kwargs):
    try:
        return sum(np.mean(np.imag(v * np.conj(i))) for v, i in zip(vabc, iabc))
    except Exception:
        return np.nan


def apparent_power(vabc, iabc, sampling_rate=1024, **kwargs):
    try:
        s_values = [np.sqrt(np.mean(np.abs(v)**2)) * np.sqrt(np.mean(np.abs(i)**2)) for v, i in zip(vabc, iabc)]
        return sum(s_values)
    except Exception:
        return np.nan


def displacement_power_factor(vabc, iabc, sampling_rate=1024, **kwargs):
    try:
        phi = [np.angle(np.vdot(v, i)) for v, i in zip(vabc, iabc)]
        cos_phi = [np.cos(p) for p in phi]
        return np.mean(cos_phi)
    except Exception:
        return np.nan


def true_power_factor(vabc, iabc, sampling_rate=1024, **kwargs):
    try:
        p = real_power_3_phase(vabc, iabc)
        s = apparent_power(vabc, iabc)
        return p / s if s != 0 else np.nan
    except Exception:
        return np.nan


def energy_drift_per_phase(vabc, iabc, sampling_rate=1024, **kwargs):
    try:
        drift = []
        for v, i in zip(vabc, iabc):
            inst_power = np.real(v * np.conj(i))
            cumulative_energy = np.cumsum(inst_power) / sampling_rate  # Joules
            drift.append(np.std(np.diff(cumulative_energy)))
        return np.mean(drift)
    except Exception:
        return np.nan
