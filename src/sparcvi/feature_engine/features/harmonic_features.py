# === harmonic_features.py ===
import numpy as np
from numpy.fft import rfft, rfftfreq


def _get_fft(signal, sampling_rate):
    N = len(signal)
    if N < 2:
        return np.array([]), np.array([])
    freqs = rfftfreq(N, d=1/sampling_rate)
    fft_vals = rfft(signal)
    return freqs, fft_vals


def voltage_thd(vabc, sampling_rate=1024):
    try:
        thds = []
        for v in vabc:
            freqs, fft_vals = _get_fft(v, sampling_rate)
            if len(fft_vals) < 2:
                continue
            fund = np.abs(fft_vals[1])
            harmonics = np.abs(fft_vals[2:])
            thd = np.sqrt(np.sum(harmonics**2)) / (fund + 1e-12)
            thds.append(thd)
        return np.mean(thds) if thds else np.nan
    except Exception:
        return np.nan


def current_thd(iabc, sampling_rate=1024):
    try:
        thds = []
        for i in iabc:
            freqs, fft_vals = _get_fft(i, sampling_rate)
            if len(fft_vals) < 2:
                continue
            fund = np.abs(fft_vals[1])
            harmonics = np.abs(fft_vals[2:])
            thd = np.sqrt(np.sum(harmonics**2)) / (fund + 1e-12)
            thds.append(thd)
        return np.mean(thds) if thds else np.nan
    except Exception:
        return np.nan


def nth_harmonic_v(vabc, n=3, sampling_rate=1024):
    try:
        mags = []
        for v in vabc:
            _, fft_vals = _get_fft(v, sampling_rate)
            if len(fft_vals) > n:
                mags.append(np.abs(fft_vals[n]))
        return np.mean(mags) if mags else np.nan
    except Exception:
        return np.nan


def nth_harmonic_i(iabc, n=3, sampling_rate=1024):
    try:
        mags = []
        for i in iabc:
            _, fft_vals = _get_fft(i, sampling_rate)
            if len(fft_vals) > n:
                mags.append(np.abs(fft_vals[n]))
        return np.mean(mags) if mags else np.nan
    except Exception:
        return np.nan


def harmonic_3rd_v(vabc, sampling_rate=1024):
    return nth_harmonic_v(vabc, n=3, sampling_rate=sampling_rate)


def harmonic_3rd_i(iabc, sampling_rate=1024):
    return nth_harmonic_i(iabc, n=3, sampling_rate=sampling_rate)


def harmonic_5th_v(vabc, sampling_rate=1024):
    return nth_harmonic_v(vabc, n=5, sampling_rate=sampling_rate)


def harmonic_5th_i(iabc, sampling_rate=1024):
    return nth_harmonic_i(iabc, n=5, sampling_rate=sampling_rate)


def harmonic_7th_v(vabc, sampling_rate=1024):
    return nth_harmonic_v(vabc, n=7, sampling_rate=sampling_rate)


def harmonic_7th_i(iabc, sampling_rate=1024):
    return nth_harmonic_i(iabc, n=7, sampling_rate=sampling_rate)


def interharmonic_energy_v(vabc, sampling_rate=1024):
    try:
        energies = []
        for v in vabc:
            freqs, fft_vals = _get_fft(v, sampling_rate)
            if len(fft_vals) < 20:
                continue
            total_energy = np.sum(np.abs(fft_vals[2:])**2)
            harmonic_energy = np.sum([np.abs(fft_vals[h])**2 for h in range(2, 20)])
            interharmonic = total_energy - harmonic_energy
            energies.append(interharmonic)
        return np.mean(energies) if energies else np.nan
    except Exception:
        return np.nan


def interharmonic_energy_i(iabc, sampling_rate=1024):
    try:
        energies = []
        for i in iabc:
            freqs, fft_vals = _get_fft(i, sampling_rate)
            if len(fft_vals) < 20:
                continue
            total_energy = np.sum(np.abs(fft_vals[2:])**2)
            harmonic_energy = np.sum([np.abs(fft_vals[h])**2 for h in range(2, 20)])
            interharmonic = total_energy - harmonic_energy
            energies.append(interharmonic)
        return np.mean(energies) if energies else np.nan
    except Exception:
        return np.nan


def harmonic_spectral_flatness(vabc, iabc, sampling_rate=1024):
    try:
        flatness = []
        for sig in vabc + iabc:
            _, fft_vals = _get_fft(sig, sampling_rate)
            if len(fft_vals) < 2:
                continue
            mag = np.abs(fft_vals[1:]) + 1e-12
            flatness.append(np.exp(np.mean(np.log(mag))) / np.mean(mag))
        return np.mean(flatness) if flatness else np.nan
    except Exception:
        return np.nan


def cumulative_harmonic_power(vabc, iabc, sampling_rate=1024):
    try:
        total_power = 0
        for v, i in zip(vabc, iabc):
            freqs, v_fft = _get_fft(v, sampling_rate)
            _, i_fft = _get_fft(i, sampling_rate)
            if len(freqs) < 3:
                continue
            harmonic_bins = np.arange(2, len(freqs))
            harmonic_power = [np.real(v_fft[k] * np.conj(i_fft[k])) for k in harmonic_bins if k < len(v_fft) and k < len(i_fft)]
            total_power += np.sum(harmonic_power)
        return total_power if total_power > 0 else np.nan
    except Exception:
        return np.nan
