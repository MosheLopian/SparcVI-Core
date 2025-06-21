# === phase_features.py ===
import numpy as np
from scipy.stats import entropy
from numpy.fft import rfft


def positive_sequence_current_ratio(iabc):
    iA, iB, iC = iabc
    i_alpha = (2/3) * (iA - 0.5*iB - 0.5*iC)
    i_beta = (2/3) * ((np.sqrt(3)/2)*iB - (np.sqrt(3)/2)*iC)
    i_pos_seq = np.sqrt(i_alpha**2 + i_beta**2)
    i_total_rms = np.sqrt(np.mean(iA**2) + np.mean(iB**2) + np.mean(iC**2))
    return np.mean(i_pos_seq) / (i_total_rms + 1e-12)


def negative_sequence_current_ratio(iabc):
    return 1 - positive_sequence_current_ratio(iabc)


def inter_phase_voltage_difference(vabc):
    try:
        diffs = [np.abs(np.real(vabc[i] - vabc[j])) for i in range(3) for j in range(i+1, 3)]
        return np.mean([np.mean(d) for d in diffs])
    except Exception:
        return np.nan


def inter_phase_thd_skew(vabc):
    try:
        thd = []
        for v in vabc:
            fft = np.abs(rfft(v))
            fund = fft[1] if len(fft) > 1 else 1
            harmonics = np.sqrt(np.sum(fft[2:]**2))
            thd.append(harmonics / (fund + 1e-12))
        return np.std(thd)
    except Exception:
        return np.nan


def voltage_unbalance_ratio(vabc):
    try:
        mags = [np.sqrt(np.mean(v**2)) for v in vabc]
        return np.std(mags) / (np.mean(mags) + 1e-12)
    except Exception:
        return np.nan


def current_unbalance_ratio(iabc):
    try:
        mags = [np.sqrt(np.mean(i**2)) for i in iabc]
        return np.std(mags) / (np.mean(mags) + 1e-12)
    except Exception:
        return np.nan


def voltage_angle_symmetry_index(vabc):
    try:
        angles = [np.angle(np.mean(v)) for v in vabc]
        return np.std(np.diff(sorted(angles)))
    except Exception:
        return np.nan


def current_angle_symmetry_index(iabc):
    try:
        angles = [np.angle(np.mean(i)) for i in iabc]
        return np.std(np.diff(sorted(angles)))
    except Exception:
        return np.nan


def phase_entropy_v(vabc):
    try:
        hist = np.histogram(np.real(np.concatenate(vabc)), bins=32)[0]
        return entropy(hist + 1e-12)
    except Exception:
        return np.nan


def phase_entropy_i(iabc):
    try:
        hist = np.histogram(np.real(np.concatenate(iabc)), bins=32)[0]
        return entropy(hist + 1e-12)
    except Exception:
        return np.nan


def sequence_component_distortion_index(vabc):
    try:
        fft = np.abs(rfft(np.concatenate(vabc)))
        fund = fft[1] if len(fft) > 1 else 1
        distortion = np.sqrt(np.sum(fft[2:]**2))
        return distortion / (fund + 1e-12)
    except Exception:
        return np.nan


def phase_dominance_index_v(vabc):
    try:
        rms = [np.sqrt(np.mean(v**2)) for v in vabc]
        return np.max(rms) / (np.sum(rms) + 1e-12)
    except Exception:
        return np.nan


def phase_dominance_index_i(iabc):
    try:
        rms = [np.sqrt(np.mean(i**2)) for i in iabc]
        return np.max(rms) / (np.sum(rms) + 1e-12)
    except Exception:
        return np.nan


def phase_voltage_angle_spread(vabc):
    try:
        angles = [np.angle(np.mean(v)) for v in vabc]
        return np.ptp(angles)
    except Exception:
        return np.nan


def phase_current_angle_spread(iabc):
    try:
        angles = [np.angle(np.mean(i)) for i in iabc]
        return np.ptp(angles)
    except Exception:
        return np.nan


def voltage_cross_phase_correlation(vabc):
    try:
        corrs = [np.corrcoef(np.real(vabc[i]), np.real(vabc[j]))[0,1] for i in range(3) for j in range(i+1, 3)]
        return np.nanmean(corrs)
    except Exception:
        return np.nan


def current_cross_phase_correlation(iabc):
    try:
        corrs = [np.corrcoef(np.real(iabc[i]), np.real(iabc[j]))[0,1] for i in range(3) for j in range(i+1, 3)]
        return np.nanmean(corrs)
    except Exception:
        return np.nan


def cross_phase_thd_differential(vabc):
    try:
        thd = []
        for v in vabc:
            fft = np.abs(rfft(v))
            fund = fft[1] if len(fft) > 1 else 1
            harmonics = np.sqrt(np.sum(fft[2:]**2))
            thd.append(harmonics / (fund + 1e-12))
        return np.ptp(thd)
    except Exception:
        return np.nan


def unbalanced_power_factor_spread(vabc, iabc):
    try:
        pf = [np.mean(v * i) / (np.sqrt(np.mean(v**2)) * np.sqrt(np.mean(i**2)) + 1e-12) for v, i in zip(vabc, iabc)]
        return np.std(pf)
    except Exception:
        return np.nan


def sequence_angle_drift(vabc, iabc):
    try:
        vang = [np.angle(np.mean(v)) for v in vabc]
        iang = [np.angle(np.mean(i)) for i in iabc]
        return np.mean([np.abs(v - i) for v, i in zip(vang, iang)])
    except Exception:
        return np.nan


def phase_voltage_unbalance(vabc):
    return voltage_unbalance_ratio(vabc)


def phase_current_unbalance(iabc):
    return current_unbalance_ratio(iabc)


def positive_sequence_voltage_ratio(vabc):
    try:
        vA, vB, vC = vabc
        alpha = (2/3) * (vA - 0.5*vB - 0.5*vC)
        beta = (2/3) * ((np.sqrt(3)/2)*vB - (np.sqrt(3)/2)*vC)
        pos_seq = np.sqrt(alpha**2 + beta**2)
        total_rms = np.sqrt(np.mean(vA**2) + np.mean(vB**2) + np.mean(vC**2))
        return np.mean(pos_seq) / (total_rms + 1e-12)
    except Exception:
        return np.nan


def negative_sequence_voltage_ratio(vabc):
    try:
        return 1 - positive_sequence_voltage_ratio(vabc)
    except Exception:
        return np.nan


def zero_sequence_voltage_ratio(vabc):
    try:
        zero_seq = np.mean(np.sum(vabc, axis=0))
        total = np.mean([np.mean(np.abs(v)) for v in vabc]) + 1e-12
        return np.abs(zero_seq) / total
    except Exception:
        return np.nan


def phase_angle_shift_v(vabc):
    try:
        ref = np.angle(np.mean(vabc[0]))
        return np.mean([np.abs(np.angle(np.mean(v)) - ref) for v in vabc[1:]])
    except Exception:
        return np.nan


def phase_angle_shift_i(iabc):
    try:
        ref = np.angle(np.mean(iabc[0]))
        return np.mean([np.abs(np.angle(np.mean(i)) - ref) for i in iabc[1:]])
    except Exception:
        return np.nan


def phase_rms_deviation_v(vabc):
    try:
        rms = [np.sqrt(np.mean(v**2)) for v in vabc]
        return np.std(rms)
    except Exception:
        return np.nan


def phase_rms_deviation_i(iabc):
    try:
        rms = [np.sqrt(np.mean(i**2)) for i in iabc]
        return np.std(rms)
    except Exception:
        return np.nan


def reactive_power_imbalance(vabc, iabc):
    try:
        q = [np.mean(np.real(v) * np.imag(i) - np.imag(v) * np.real(i)) for v, i in zip(vabc, iabc)]
        return np.std(q)
    except Exception:
        return np.nan
