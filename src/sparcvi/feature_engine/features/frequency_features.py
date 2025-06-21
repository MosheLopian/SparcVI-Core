# === frequency_features.py ===
import numpy as np


def fundamental_frequency(vabc, sampling_rate=1024):
    try:
        dominant_freqs = []
        for v in vabc:
            fft_vals = np.fft.rfft(v)
            freqs = np.fft.rfftfreq(len(v), d=1/sampling_rate)
            idx = np.argmax(np.abs(fft_vals[1:])) + 1
            dominant_freqs.append(freqs[idx])
        return np.mean(dominant_freqs) if dominant_freqs else np.nan
    except Exception:
        return np.nan


def frequency_deviation(vabc, sampling_rate=1024, nominal=50.0):
    try:
        f0 = fundamental_frequency(vabc, sampling_rate)
        return abs(f0 - nominal)
    except Exception:
        return np.nan


def frequency_stability_index(vabc, sampling_rate=1024):
    try:
        variations = []
        for v in vabc:
            segments = np.array_split(v, 4)
            freqs = []
            for seg in segments:
                fft_vals = np.fft.rfft(seg)
                f = np.fft.rfftfreq(len(seg), d=1/sampling_rate)
                idx = np.argmax(np.abs(fft_vals[1:])) + 1
                freqs.append(f[idx])
            variations.append(np.std(freqs))
        return np.mean(variations) if variations else np.nan
    except Exception:
        return np.nan


def oscillatory_component_index(vabc, sampling_rate=1024):
    try:
        oci = []
        for v in vabc:
            fft_vals = np.abs(np.fft.rfft(v))
            total_power = np.sum(fft_vals[1:]**2)
            band_power = np.sum(fft_vals[20:]**2)
            ratio = band_power / (total_power + 1e-12)
            oci.append(ratio)
        return np.mean(oci) if oci else np.nan
    except Exception:
        return np.nan


def zero_crossing_frequency_estimate(vabc, sampling_rate=1024):
    try:
        freqs = []
        for v in vabc:
            zero_crossings = np.where(np.diff(np.signbit(v)))[0]
            if len(zero_crossings) < 2:
                continue
            periods = np.diff(zero_crossings) / sampling_rate
            if len(periods) == 0:
                continue
            avg_period = np.mean(periods)
            if avg_period == 0:
                continue
            freq = 1.0 / (2 * avg_period)  # 2 zero crossings per full cycle
            freqs.append(freq)
        return np.mean(freqs) if freqs else np.nan
    except Exception:
        return np.nan
