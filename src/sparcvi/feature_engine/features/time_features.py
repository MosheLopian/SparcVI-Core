# === time_features.py ===
import numpy as np
import numpy as np
import scipy.signal as signal
from scipy.stats import skew, kurtosis, entropy


def voltage_rms(voltage, current=None, sampling_rate=1024, **kwargs):
    try:
        return np.sqrt(np.mean(voltage ** 2))
    except Exception:
        return np.nan


def current_rms(voltage, current, sampling_rate=1024, **kwargs):
    try:
        return np.sqrt(np.mean(current ** 2))
    except Exception:
        return np.nan


def voltage_peak(voltage, current=None, sampling_rate=1024, **kwargs):
    try:
        return np.max(np.abs(voltage))
    except Exception:
        return np.nan


def current_peak(voltage, current, sampling_rate=1024, **kwargs):
    try:
        return np.max(np.abs(current))
    except Exception:
        return np.nan


def voltage_std_dev(voltage, current=None, sampling_rate=1024, **kwargs):
    try:
        return np.std(voltage)
    except Exception:
        return np.nan


def current_std_dev(voltage, current, sampling_rate=1024, **kwargs):
    try:
        return np.std(current)
    except Exception:
        return np.nan


def voltage_skewness(voltage, current=None, sampling_rate=1024, **kwargs):
    try:
        return skew(voltage)
    except Exception:
        return np.nan


def current_skewness(voltage, current, sampling_rate=1024, **kwargs):
    try:
        return skew(current)
    except Exception:
        return np.nan


def voltage_kurtosis(voltage, current=None, sampling_rate=1024, **kwargs):
    try:
        return kurtosis(voltage)
    except Exception:
        return np.nan


def current_kurtosis(voltage, current, sampling_rate=1024, **kwargs):
    try:
        return kurtosis(current)
    except Exception:
        return np.nan


def voltage_crest_factor(voltage, current=None, sampling_rate=1024, **kwargs):
    try:
        rms = voltage_rms(voltage)
        return np.max(np.abs(voltage)) / rms if rms != 0 else np.nan
    except Exception:
        return np.nan


def current_crest_factor(voltage, current, sampling_rate=1024, **kwargs):
    try:
        rms = current_rms(voltage, current)
        return np.max(np.abs(current)) / rms if rms != 0 else np.nan
    except Exception:
        return np.nan


def voltage_entropy(voltage, current=None, sampling_rate=1024, **kwargs):
    try:
        hist, _ = np.histogram(voltage, bins=64, density=True)
        return entropy(hist + 1e-12)
    except Exception:
        return np.nan


def current_entropy(voltage, current, sampling_rate=1024, **kwargs):
    try:
        hist, _ = np.histogram(current, bins=64, density=True)
        return entropy(hist + 1e-12)
    except Exception:
        return np.nan


def entropy_delta(voltage, current, sampling_rate=1024, **kwargs):
    try:
        v_entropy = voltage_entropy(voltage)
        i_entropy = current_entropy(voltage, current)
        return abs(v_entropy - i_entropy)
    except Exception:
        return np.nan


def voltage_zero_crossings(voltage, current=None, sampling_rate=1024, **kwargs):
    try:
        return ((voltage[:-1] * voltage[1:]) < 0).sum()
    except Exception:
        return np.nan


def current_zero_crossings(voltage, current, sampling_rate=1024, **kwargs):
    try:
        return ((current[:-1] * current[1:]) < 0).sum()
    except Exception:
        return np.nan


def zero_crossing_irregularity(voltage, current=None, sampling_rate=1024, **kwargs):
    try:
        zc = np.where(np.diff(np.signbit(voltage)))[0]
        if len(zc) < 2:
            return np.nan
        intervals = np.diff(zc) / sampling_rate
        return np.std(intervals)
    except Exception:
        return np.nan


def cycle_stability(voltage, current=None, sampling_rate=1024, **kwargs):
    try:
        cycles = int(len(voltage) / sampling_rate)
        rms_values = [np.sqrt(np.mean(voltage[i * sampling_rate:(i + 1) * sampling_rate] ** 2))
                      for i in range(cycles)]
        return np.std(rms_values) if len(rms_values) > 1 else np.nan
    except Exception:
        return np.nan
    
def rise_time_v(voltage, sampling_rate=1024):
    try:
        v = np.array(voltage)
        v_min = np.min(v)
        v_max = np.max(v)
        threshold_low = v_min + 0.1 * (v_max - v_min)
        threshold_high = v_min + 0.9 * (v_max - v_min)
        indices = np.where((v >= threshold_low) & (v <= threshold_high))[0]
        if len(indices) < 2:
            return float('nan')
        rise_samples = indices[-1] - indices[0]
        return rise_samples / sampling_rate
    except:
        return float('nan')

def rise_time_i(current, sampling_rate=1024):
    try:
        i = np.array(current)
        i_min = np.min(i)
        i_max = np.max(i)
        threshold_low = i_min + 0.1 * (i_max - i_min)
        threshold_high = i_min + 0.9 * (i_max - i_min)
        indices = np.where((i >= threshold_low) & (i <= threshold_high))[0]
        if len(indices) < 2:
            return float('nan')
        rise_samples = indices[-1] - indices[0]
        return rise_samples / sampling_rate
    except:
        return float('nan')

def load_skew_i(current):
    try:
        i = np.array(current)
        rms = np.sqrt(np.mean(i ** 2))
        avg = np.mean(i)
        return avg / rms if rms != 0 else float('nan')
    except:
        return float('nan')

def signal_to_noise_ratio(voltage):
    try:
        signal_power = np.mean(np.square(voltage))
        noise_power = np.var(voltage - np.mean(voltage))
        snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')
        return snr
    except:
        return float('nan')

def voltage_flicker_index(voltage, sampling_rate=1024):
    try:
        voltage = np.array(voltage)
        window_size = int(sampling_rate * 0.1)
        if window_size < 1 or len(voltage) < window_size:
            return float('nan')
        smoothed = signal.convolve(voltage, np.ones(window_size)/window_size, mode='valid')
        flicker = np.std(smoothed)
        return flicker
    except:
        return float('nan')

def current_flicker_index(current, sampling_rate=1024):
    try:
        current = np.array(current)
        window_size = int(sampling_rate * 0.1)
        if window_size < 1 or len(current) < window_size:
            return float('nan')
        smoothed = signal.convolve(current, np.ones(window_size)/window_size, mode='valid')
        flicker = np.std(smoothed)
        return flicker
    except:
        return float('nan')

def dc_offset_v(voltage):
    try:
        return float(np.mean(voltage))
    except:
        return float('nan')

def dc_offset_i(current):
    try:
        return float(np.mean(current))
    except:
        return float('nan')

def voltage_sag_duration(voltage, sampling_rate=1024, threshold=0.9):
    try:
        v = np.abs(voltage)
        nominal = np.max(v)
        sag_mask = v < threshold * nominal
        sag_samples = np.sum(sag_mask)
        return sag_samples / sampling_rate
    except:
        return float('nan')

def voltage_swell_duration(voltage, sampling_rate=1024, threshold=1.1):
    try:
        v = np.abs(voltage)
        nominal = np.max(v)
        swell_mask = v > threshold * nominal
        swell_samples = np.sum(swell_mask)
        return swell_samples / sampling_rate
    except:
        return float('nan')

def current_sag_duration(current, sampling_rate=1024, threshold=0.9):
    try:
        i = np.abs(current)
        nominal = np.max(i)
        sag_mask = i < threshold * nominal
        sag_samples = np.sum(sag_mask)
        return sag_samples / sampling_rate
    except:
        return float('nan')

def current_swell_duration(current, sampling_rate=1024, threshold=1.1):
    try:
        i = np.abs(current)
        nominal = np.max(i)
        swell_mask = i > threshold * nominal
        swell_samples = np.sum(swell_mask)
        return swell_samples / sampling_rate
    except:
        return float('nan')

def waveform_asymmetry_index(voltage):
    try:
        v = np.array(voltage)
        pos = np.sum(v[v > 0])
        neg = np.abs(np.sum(v[v < 0]))
        return abs(pos - neg) / (pos + neg + 1e-12)
    except:
        return float('nan')

def flat_top_distortion_index(voltage):
    try:
        v = np.array(voltage)
        peak = np.max(np.abs(v))
        near_peak = v[np.abs(v) > 0.95 * peak]
        return len(near_peak) / len(v)
    except:
        return float('nan')

def transient_spike_count(voltage, threshold_factor=3):
    try:
        v = np.array(voltage)
        diff = np.diff(v)
        std = np.std(diff)
        spikes = np.where(np.abs(diff) > threshold_factor * std)[0]
        return len(spikes)
    except:
        return float('nan')

def rms_stability(voltage, window=128):
    try:
        v = np.array(voltage)
        if len(v) < window * 2:
            return float('nan')
        rms_vals = [np.sqrt(np.mean(v[i:i+window]**2)) for i in range(0, len(v)-window, window)]
        return np.std(rms_vals)
    except:
        return float('nan')

def peak_dv_dt(voltage, sampling_rate=1024):
    try:
        dv = np.diff(voltage) * sampling_rate
        return np.max(np.abs(dv))
    except:
        return float('nan')

def peak_di_dt(current, sampling_rate=1024):
    try:
        di = np.diff(current) * sampling_rate
        return np.max(np.abs(di))
    except:
        return float('nan')

def waveform_smoothness(voltage):
    try:
        diff2 = np.diff(voltage, n=2)
        return np.mean(diff2**2)
    except:
        return float('nan')

def time_symmetry_index(voltage):
    try:
        midpoint = len(voltage) // 2
        v1 = np.array(voltage[:midpoint])
        v2 = np.array(voltage[-midpoint:])[::-1]
        return np.mean(np.abs(v1 - v2)) / (np.mean(np.abs(v1)) + 1e-12)
    except:
        return float('nan')

def voltage_crest_to_entropy_ratio(voltage):
    try:
        peak = np.max(np.abs(voltage))
        entropy = -np.sum((p := np.histogram(voltage, bins=32, density=True)[0]) * np.log2(p + 1e-12))
        return peak / (entropy + 1e-6)
    except:
        return float('nan')

def current_crest_to_entropy_ratio(current):
    try:
        peak = np.max(np.abs(current))
        entropy = -np.sum((p := np.histogram(current, bins=32, density=True)[0]) * np.log2(p + 1e-12))
        return peak / (entropy + 1e-6)
    except:
        return float('nan')

def voltage_skew_to_kurtosis_ratio(voltage):
    try:
        from scipy.stats import skew, kurtosis
        return skew(voltage) / (kurtosis(voltage) + 1e-6)
    except:
        return float('nan')

def current_skew_to_kurtosis_ratio(current):
    try:
        from scipy.stats import skew, kurtosis
        return skew(current) / (kurtosis(current) + 1e-6)
    except:
        return float('nan')

def entropy_delta_entropy_mean(voltage, current):
    try:
        from scipy.stats import entropy
        hist_v, _ = np.histogram(voltage, bins=64, density=True)
        hist_i, _ = np.histogram(current, bins=64, density=True)
        ent_v = entropy(hist_v + 1e-12)
        ent_i = entropy(hist_i + 1e-12)
        delta = abs(ent_v - ent_i)
        mean = (ent_v + ent_i) / 2.0
        return delta / (mean + 1e-6)
    except:
        return float('nan')

def zero_crossing_irregularity_rms(voltage, current):
    try:
        def zero_cross_irreg(sig):
            crossings = np.where(np.diff(np.signbit(sig)))[0]
            intervals = np.diff(crossings)
            return np.std(intervals)
        irr_v = zero_cross_irreg(voltage)
        rms_v = np.sqrt(np.mean(np.square(voltage)))
        return irr_v / (rms_v + 1e-6)
    except:
        return float('nan')

def rise_time_ratio_v_i(voltage, current, sampling_rate=1024):
    try:
        def rise_time(sig):
            max_val = np.max(sig)
            idx_10 = np.argmax(sig >= 0.1 * max_val)
            idx_90 = np.argmax(sig >= 0.9 * max_val)
            return (idx_90 - idx_10) / sampling_rate
        rt_v = rise_time(voltage)
        rt_i = rise_time(current)
        return rt_v / (rt_i + 1e-6)
    except:
        return float('nan')

def rms_stability_index(voltage, current):
    try:
        def moving_rms(sig, window=64):
            return np.array([np.sqrt(np.mean(sig[i:i+window]**2)) for i in range(len(sig)-window)])
        v_rms_series = moving_rms(np.array(voltage))
        return np.std(v_rms_series) / (np.mean(v_rms_series) + 1e-6)
    except:
        return float('nan')

def peak_di_dt_rms_i(current, sampling_rate=1024):
    try:
        di = np.diff(current) * sampling_rate
        peak = np.max(np.abs(di))
        rms = np.sqrt(np.mean(np.square(current)))
        return peak / (rms + 1e-6)
    except:
        return float('nan')

def current_to_voltage_rms_ratio(voltage, current):
    try:
        rms_v = np.sqrt(np.mean(np.square(voltage)))
        rms_i = np.sqrt(np.mean(np.square(current)))
        return rms_i / (rms_v + 1e-6)
    except:
        return float('nan')

def voltage_flicker_to_rms_ratio(voltage):
    try:
        flicker = np.std(voltage) / (np.mean(np.abs(voltage)) + 1e-6)
        rms = np.sqrt(np.mean(np.square(voltage)))
        return flicker / (rms + 1e-6)
    except:
        return float('nan')
