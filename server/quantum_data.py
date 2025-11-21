from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import io
from pathlib import Path

import math
import base64

import pandas as pd
pd.set_option("display.max_columns", None)

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample_poly, correlate
from sklearn.preprocessing import StandardScaler, MinMaxScaler

MOTOR_UPDRS_MIN = 5.0377
MOTOR_UPDRS_MAX = 39.511
TOTAL_UPDRS_MIN = 7.0
TOTAL_UPDRS_MAX = 54.992
MOTOR_UPDRS_MEAN = 21.296
MOTOR_UPDRS_STD = 8.129
TOTAL_UPDRS_MEAN = 29.019
TOTAL_UPDRS_STD = 10.700

WAV_PATH: Path = Path(__file__).with_name("test_audio.wav")
FEATURES_SIGNIFICANT = [0, 1, 2, 3, 4, 6, 9, 11, 12, 15, 16, 17, 18]
X_SCALER = None
Y_SCALER = None

# voice features: subject#,age,sex,test_time,motor_UPDRS,total_UPDRS,Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP,Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA,NHR,HNR,RPDE,DFA,PPE
PARKINSONS_TELEMONITORING_DATA_PATH = "../data/parkinsons_telemonitoring_colab.csv"

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def normalize(x: float, x_min: float, x_max: float) -> float:
    return np.clip((x - x_min) / (x_max - x_min), 0, 1)

# Calculate Parkinson's Disease probability from motor_UPDRS and total_UPDRS. The method is one of ['normalized', 'sigmoid', 'hybrid']
def parkinson_probability(motor_updrs: float, total_updrs: float, method: str = "hybrid") -> float:
    print("motor_updrs: ", motor_updrs)
    print("total_updrs: ", total_updrs)
    p_motor_norm = normalize(motor_updrs, MOTOR_UPDRS_MIN, MOTOR_UPDRS_MAX)
    print("p_motor_norm: ", p_motor_norm)
    p_total_norm = normalize(total_updrs, TOTAL_UPDRS_MIN, TOTAL_UPDRS_MAX)
    print("p_total_norm: ", p_total_norm)
    if method == "normalized":
        return float((p_motor_norm + p_total_norm) / 2)
    p_motor_sig = sigmoid((motor_updrs - MOTOR_UPDRS_MEAN) / MOTOR_UPDRS_STD)
    print("p_motor_sig: ", p_motor_sig)
    p_total_sig = sigmoid((total_updrs - TOTAL_UPDRS_MEAN) / TOTAL_UPDRS_STD)
    print("p_total_sig: ", p_total_sig)
    if method == "sigmoid":
        return float((p_motor_sig + p_total_sig) / 2)
    else: # hybrid
        p_combined = (p_motor_norm + p_motor_sig + p_total_norm + p_total_sig) / 8
        return float(p_combined)

def get_scalers(X, Y):
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    Y_scaled = y_scaler.fit_transform(Y)
    return x_scaler, y_scaler

def init_foundation_data():
    global X_SCALER
    global Y_SCALER

    print("Data source path: ", PARKINSONS_TELEMONITORING_DATA_PATH)
    dataset = pd.read_csv(PARKINSONS_TELEMONITORING_DATA_PATH)
    print(dataset[0:30])
    print("Columns Names: ",dataset.columns)
    print("Columns Count: ",len(dataset.columns))
    print(dataset.shape)
    dataset = dataset.drop(["subject#"], axis=1) # !!! Drop ID column
    
    dataset.isnull().sum()
    array = dataset.values
    X1 = array[:,0:4]
    X2 = array[:,6:]
    X = np.hstack((X1,X2)) # Feature matrix (input data)
    X = pd.DataFrame(X)
    X = X[FEATURES_SIGNIFICANT]
    X = X.values
    print("Input shape: ", X.shape)
    Y = array[:,4:6] # Label matrix (output data) for motor_UPDRS (4) and total_UPDRS (5).
    print("Ouput shape: ", Y.shape)
    
    X_SCALER, Y_SCALER = get_scalers(X, Y)
    
    return X, Y, dataset

def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Decode audio stream from bytes. Return mono float32 in [-1, 1] and sample rate.
    Resamples to target_sr if needed. If stereo, averages channels.
    """
    #sr, x = target_sr, np.array(audio_bytes, dtype=np.float32)
    #x = x.astype(np.float32) / 32768.0
    
    x = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    #x = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 2147483648.0
    sr = target_sr

    # Remove DC and very-low drift, gentle high-pass at 30 Hz
    b, a = butter(2, 30.0 / (0.5 * sr), btype="highpass")
    x = filtfilt(b, a, x).astype(np.float32)

    # Normalize to fixed RMS target for stability
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    if rms > 0:
        x = x / max(0.01, rms)

    return x, sr

def load_wav_from_bytes(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Decode PCM WAV from bytes. Return mono float32 in [-1, 1] and sample rate.
    Resamples to target_sr if needed. If stereo, averages channels.
    """
    sr, x = wavfile.read(io.BytesIO(audio_bytes))
    # Convert to float32 range [-1, 1]
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    else:
        x = x.astype(np.float32)

    # Mono
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Remove DC and very-low drift, gentle high-pass at 30 Hz
    b, a = butter(2, 30.0 / (0.5 * sr), btype="highpass")
    x = filtfilt(b, a, x).astype(np.float32)

    # Resample if needed
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        x = resample_poly(x, up, down).astype(np.float32)
        sr = target_sr

    # Normalize to fixed RMS target for stability
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    if rms > 0:
        x = x / max(0.01, rms)

    return x, sr

@dataclass
class FrameConfig:
    frame_ms: float = 40.0
    hop_ms: float = 10.0
    fmin: float = 50.0     # Hz
    fmax: float = 400.0    # Hz
    voicing_thresh: float = 0.3  # autocorr peak threshold for voicing


def framing(x: np.ndarray, sr: int, cfg: FrameConfig) -> Tuple[np.ndarray, int]:
    frame_len = int(round(cfg.frame_ms * 1e-3 * sr))
    hop_len = int(round(cfg.hop_ms * 1e-3 * sr))
    if frame_len <= 0:
        raise ValueError("frame_len must be positive")
    if len(x) < frame_len:
        # pad if too short
        pad = frame_len - len(x)
        x = np.pad(x, (0, pad))
    n_frames = 1 + (len(x) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return x[idx], hop_len


def hann_window(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / max(1, n - 1))

# F0 via autocorrelation
def f0_autocorr(x: np.ndarray, sr: int, cfg: FrameConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-frame F0 using time-domain autocorrelation.
    Returns f0_hz (nan for unvoiced) and corresponding period_s.
    """
    frames, _ = framing(x, sr, cfg)
    win = hann_window(frames.shape[1]).astype(np.float32)
    frames = frames * win

    min_lag = int(np.floor(sr / cfg.fmax))
    max_lag = int(np.ceil(sr / cfg.fmin))
    f0 = np.full(frames.shape[0], 0.0, dtype=np.float32) # np.nan
    period = np.full(frames.shape[0], 0.0, dtype=np.float32) # np.nan

    for i, f in enumerate(frames):
        if not np.any(f):
            continue
        # unbiased autocorrelation
        ac = correlate(f, f, mode='full')
        ac = ac[len(f)-1:]  # non-negative lags
        if ac[0] <= 0:
            continue

        # Normalize
        ac = ac / (ac[0] + 1e-12)

        # Search peak in plausible lag range
        search = ac[min_lag:max_lag]
        if len(search) == 0:
            continue
        peak_idx = np.argmax(search)
        peak_val = search[peak_idx]
        lag = min_lag + peak_idx

        if peak_val >= cfg.voicing_thresh:
            t0 = lag / sr
            period[i] = t0
            f0[i] = 1.0 / t0 if t0 > 0 else 0.0 # np.nan

    return f0, period

# Amplitude proxy per frame
def frame_rms(x: np.ndarray, sr: int, cfg: FrameConfig) -> np.ndarray:
    frames, _ = framing(x, sr, cfg)
    win = hann_window(frames.shape[1]).astype(np.float32)
    frames = frames * win
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12).astype(np.float32)



# Jitter metrics (period perturbation)
def jitter_metrics(period_s: np.ndarray) -> Dict[str, float]:
    """
    period_s: array of period estimates in seconds for voiced frames (nan for unvoiced)
    Jitter local (%) = mean(|diff(P)|) / mean(P) * 100
    Jitter abs (s)   = mean(|diff(P)|)
    RAP              = mean(|P_i - mean(P_{i-1:i+1})|) / mean(P)
    PPQ5             = mean(|P_i - mean(P_{i-2:i+2})|) / mean(P)
    DDP              = 3 * RAP
    """
    P = period_s[~np.isnan(period_s)]
    if len(P) < 6:
        return {"jitter_percent": 0.0, # np.nan
                "jitter_abs": 0.0, # np.nan
                "jitter_rap": 0.0, # np.nan
                "jitter_ppq5": 0.0, # np.nan
                "jitter_ddp": 0.0} # np.nan

    meanP = float(np.mean(P)) + 0.0000001
    dP = np.abs(np.diff(P))
    jitter_percent = float(np.mean(dP) / meanP * 100.0)
    jitter_abs = float(np.mean(dP))

    # RAP: 3-point moving average centered
    rap_terms = []
    for i in range(1, len(P) - 1):
        local_mean = (P[i - 1] + P[i] + P[i + 1]) / 3.0
        rap_terms.append(abs(P[i] - local_mean))
    jitter_rap = float(np.mean(rap_terms) / meanP) if rap_terms else 0.0 # np.nan

    # PPQ5: 5-point moving average centered
    ppq_terms = []
    for i in range(2, len(P) - 2):
        local_mean = np.mean(P[i - 2:i + 3])
        ppq_terms.append(abs(P[i] - local_mean))
    jitter_ppq5 = float(np.mean(ppq_terms) / meanP) if ppq_terms else 0.0 # np.nan

    jitter_ddp = 3.0 * jitter_rap if not np.isnan(jitter_rap) else 0.0 # np.nan

    return {
        "jitter_percent": jitter_percent,
        "jitter_abs": jitter_abs,
        "jitter_rap": jitter_rap,
        "jitter_ppq5": jitter_ppq5,
        "jitter_ddp": jitter_ddp,
    }


# Shimmer metrics (amplitude perturbation)
def shimmer_metrics(ampl: np.ndarray) -> Dict[str, float]:
    """
    ampl: amplitude proxy per voiced frame (RMS)
    Shimmer (local)       = mean(|diff(A)|) / mean(A)
    Shimmer (dB)          = 20 * log10( mean(A_i / A_{i+1}) ) using absolute ratio
    APQ3, APQ5, APQ11     = mean(|A_i - mean(A_neigh)|) / mean(A) with 3,5,11-term neighborhoods
    DDA                   = 3 * APQ3
    """
    A = ampl[~np.isnan(ampl)]
    if len(A) < 12:
        return {"shimmer": 0.0, # np.nan
                "shimmer_db": 0.0, # np.nan
                "shimmer_apq3": 0.0, # np.nan
                "shimmer_apq5": 0.0, # np.nan
                "shimmer_apq11": 0.0, # np.nan
                "shimmer_dda": 0.0} # np.nan

    meanA = float(np.mean(A))
    dA = np.abs(np.diff(A))
    shimmer = float(np.mean(dA) / meanA)

    # Use symmetric ratio, avoid log of zero
    ratios = (A[:-1] + 1e-12) / (A[1:] + 1e-12)
    shimmer_db = float(20.0 * np.log10(np.mean(np.abs(ratios))))

    def apq_k(k: int) -> float:
        terms = []
        half = k // 2
        for i in range(half, len(A) - half):
            neigh = np.concatenate([A[i - half:i], A[i + 1:i + 1 + half]])
            if len(neigh) != k:
                continue
            local_mean = np.mean(neigh)
            terms.append(abs(A[i] - local_mean))
        return float(np.mean(terms) / meanA) if terms else 0.0 # np.nan

    apq3 = apq_k(3)
    apq5 = apq_k(5)
    apq11 = apq_k(11)
    dda = 3.0 * apq3 if not np.isnan(apq3) else 0.0 # np.nan

    return {
        "shimmer": shimmer,
        "shimmer_db": shimmer_db,
        "shimmer_apq3": apq3,
        "shimmer_apq5": apq5,
        "shimmer_apq11": apq11,
        "shimmer_dda": dda,
    }


# HNR and NHR (autocorr-based)
def hnr_nhr(x: np.ndarray, sr: int, period_s: np.ndarray, cfg: FrameConfig) -> Tuple[float, float]:
    """
    Autocorr-based HNR per frame:
    HNR = 10*log10( Rxx(T0) / (Rxx(0) - Rxx(T0)) )
    NHR as linear ratio: NHR = (Rxx(0) - Rxx(T0)) / Rxx(T0)
    Returns mean over voiced frames.
    """
    frames, _ = framing(x, sr, cfg)
    win = hann_window(frames.shape[1]).astype(np.float32)
    hnr_vals = []
    nhr_vals = []
    min_lag = int(np.floor(sr / cfg.fmax))
    max_lag = int(np.ceil(sr / cfg.fmin))

    for i, f in enumerate(frames):
        if np.isnan(period_s[i]):
            continue
        f = f * win
        ac = correlate(f, f, mode='full')
        ac = ac[len(f)-1:]
        R0 = ac[0]
        if R0 <= 0:
            continue
        # pick lag around expected T0 for robustness
        exp_lag = int(round(period_s[i] * sr))
        lag_lo = max(min_lag, exp_lag - 1)
        lag_hi = min(max_lag, exp_lag + 1)
        Rtau = np.max(ac[lag_lo:lag_hi+1]) if lag_hi >= lag_lo else ac[exp_lag]
        Rtau = max(1e-12, Rtau)

        harm = Rtau
        noise = max(1e-12, R0 - Rtau)
        hnr_vals.append(10.0 * np.log10(harm / noise))
        nhr_vals.append(noise / harm)

    hnr_mean = float(np.mean(hnr_vals)) if hnr_vals else 0.0 # np.nan
    nhr_mean = float(np.mean(nhr_vals)) if nhr_vals else 0.0 # np.nan
    return hnr_mean, nhr_mean


# RPDE (recurrence period density entropy)
def rpde(x: np.ndarray, m: int = 3, tau: int = 2, eps_frac: float = 0.1, max_T: int = 50) -> float:
    """
    Approximate RPDE:
        1) time-delay embed
        2) build recurrence periods from nearest neighbors within epsilon
        3) compute normalized Shannon entropy of period histogram
    Returns value in [0, 1] (higher means more irregular).
    """
    N = len(x) - (m - 1) * tau
    if N <= max_T + 5:
        return 0.0 # np.nan
    X = np.stack([x[i:i + N] for i in range(0, m * tau, tau)], axis=1)
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-12)

    # epsilon threshold as fraction of median pairwise distance across a sample
    idx = np.random.default_rng(0).choice(len(X), size=min(2000, len(X)), replace=False)
    dists = []
    for i in range(len(idx) - 1):
        d = np.linalg.norm(X[idx[i]] - X[idx[i + 1]])
        dists.append(d)
    eps = eps_frac * (np.median(dists) if dists else 1.0)

    periods = []
    for i in range(len(X) - max_T - 1):
        # find first j>i such that ||X[j]-X[i]|| < eps, j-i within [1, max_T]
        found = False
        for T in range(1, max_T + 1):
            j = i + T
            if j >= len(X):
                break
            if np.linalg.norm(X[j] - X[i]) < eps:
                periods.append(T)
                found = True
                break
        if not found:
            periods.append(max_T)

    if not periods:
        return 0.0 # np.nan
    hist, _ = np.histogram(periods, bins=np.arange(1, max_T + 2), density=False)
    p = hist.astype(np.float64) / np.sum(hist)
    p = p[p > 0]
    H = -np.sum(p * np.log(p + 1e-12))
    Hmax = math.log(len(hist) + 1e-12)
    return float(H / Hmax) if Hmax > 0 else 0.0 # np.nan


# DFA (detrended fluctuation analysis)
def dfa_alpha(x: np.ndarray, min_win: int = 16, max_win: int = 1024, n_scales: int = 20) -> float:
    """
    Standard DFA implementation returning scaling exponent alpha.
    """
    x = x.astype(np.float64)
    x = x - np.mean(x)
    y = np.cumsum(x)

    scales = np.unique(np.logspace(np.log10(min_win), np.log10(max_win), n_scales).astype(int))
    F = []
    for s in scales:
        if s < 4:
            continue
        n_segs = len(y) // s
        if n_segs < 2:
            continue
        y_seg = y[:n_segs * s].reshape(n_segs, s)
        t = np.arange(s)
        # linear detrend each segment
        res = []
        for seg in y_seg:
            A = np.vstack([t, np.ones_like(t)]).T
            coef, _, _, _ = np.linalg.lstsq(A, seg, rcond=None)
            trend = A @ coef
            res.append(np.sqrt(np.mean((seg - trend)**2)))
        F.append((s, np.sqrt(np.mean(np.square(res)))))

    if len(F) < 2:
        return 0.0 # np.nan
    S = np.array(F, dtype=np.float64)
    xs = np.log(S[:, 0])
    ys = np.log(S[:, 1] + 1e-12)
    a, b = np.polyfit(xs, ys, 1)
    return float(a)

# PPE (pitch period entropy)
def ppe_from_f0(f0_hz: np.ndarray) -> float:
    """
    Pitch period entropy over semitone-normalized deviations.
    Steps:
        1) Convert F0 to cents relative to median voiced F0.
        2) Z-normalize deviations.
        3) Compute discrete entropy of histogram.
    """
    f0 = f0_hz[~np.isnan(f0_hz)]
    if len(f0) < 10:
        return 0.0 # np.nan
    f0_med = np.median(f0)
    if f0_med <= 0:
        return 0.0 # np.nan
    # semitone deviations
    dev_st = 12.0 * np.log2(f0 / f0_med + 1e-12)
    z = (dev_st - np.mean(dev_st)) / (np.std(dev_st) + 1e-12)
    # histogram with robust binning
    hist, _ = np.histogram(z, bins=30, range=(-3.0, 3.0), density=False)
    p = hist.astype(np.float64) / max(1, np.sum(hist))
    p = p[p > 0]
    H = -np.sum(p * np.log(p + 1e-12))
    # Normalize by log(number of non-empty bins) so PPE in [0,1]
    Hmax = math.log(len(p) + 1e-12)
    return float(H / Hmax) if Hmax > 0 else 0.0 # np.nan

# Main feature extractor
def compute_voice_features_from_bytes(
    audio_bytes: bytes,
    target_sr: int = 16000,
    cfg: Optional[FrameConfig] = None
) -> Dict[str, float]:
    """
    Compute the requested acoustic features from WAV bytes.
    Returns a dict with keys matching the table names.
    """
    if cfg is None:
        cfg = FrameConfig()

    x, sr = load_wav_from_bytes(audio_bytes, target_sr=target_sr)
    # x, sr = load_audio_from_bytes(audio_bytes, target_sr=target_sr)

    # F0 and period per frame
    f0_hz, period_s = f0_autocorr(x, sr, cfg)

    # Mask unvoiced frames using simple F0 presence
    voiced = ~np.isnan(f0_hz)

    # Amplitude proxy per frame
    amp = frame_rms(x, sr, cfg).astype(np.float32)
    amp_voiced = amp.copy()
    amp_voiced[~voiced] = 0.0 # np.nan

    # Jitter metrics
    jit = jitter_metrics(period_s)

    # Shimmer metrics
    shim = shimmer_metrics(amp_voiced)

    # HNR and NHR
    hnr, nhr = hnr_nhr(x, sr, period_s, cfg)

    # RPDE on band-limited, z-normalized signal
    # Use a light band-pass around typical voice bandwidth if desired.
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-12)
    rpde_val = rpde(x_norm)

    # DFA on amplitude envelope for robustness
    dfa_val = dfa_alpha(amp)

    # PPE from f0 contour
    ppe_val = ppe_from_f0(f0_hz)

    return {
        "Jitter(%)": jit["jitter_percent"],
        "Jitter(Abs)": jit["jitter_abs"],
        "Jitter:RAP": jit["jitter_rap"],
        "Jitter:PPQ5": jit["jitter_ppq5"],
        "Jitter:DDP": jit["jitter_ddp"],
        "Shimmer": shim["shimmer"],
        "Shimmer(dB)": shim["shimmer_db"],
        "Shimmer:APQ3": shim["shimmer_apq3"],
        "Shimmer:APQ5": shim["shimmer_apq5"],
        "Shimmer:APQ11": shim["shimmer_apq11"],
        "Shimmer:DDA": shim["shimmer_dda"],
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde_val,
        "DFA": dfa_val,
        "PPE": ppe_val,
    }

"""
# Test code:
if __name__ == "__main__":
    with open(WAV_PATH, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")
    audio_bytes = base64.b64decode(audio_b64, validate=True)
    feats = compute_voice_features_from_bytes(audio_bytes)
    print(feats)
"""

"""
| id | name          | role     | type       | description                                                                          |
|----|---------------|----------|------------|--------------------------------------------------------------------------------------|
| 1  | age           | Feature  | Integer    | Subject age                                                                          |
| 2  | sex           | Feature  | Binary     | Subject sex '0' - male, '1' - female                                                 |
| 3  | test_time     | Feature  | Continuous | Time since recruitment into the trial. The integer part is days since recruitment    |
| 4  | motor_UPDRS   | Target   | Continuous | Motor Unified Parkinson's Disease Rating Scale score, linearly interpolated          |
| 5  | total_UPDRS   | Target   | Continuous | Total Unified Parkinson's Disease Rating Scale score, linearly interpolated          |
| 6  | Jitter(%)     | Feature  | Continuous | Several measures of variation in fundamental frequency                               |
| 7  | Jitter(Abs)   | Feature  | Continuous | Several measures of variation in fundamental frequency                               |
| 8  | Jitter:RAP    | Feature  | Continuous | Several measures of variation in fundamental frequency                               |
| 9  | Jitter:PPQ5   | Feature  | Continuous | Several measures of variation in fundamental frequency                               |
| 10 | Jitter:DDP    | Feature  | Continuous | Several measures of variation in fundamental frequency                               |
| 11 | Shimmer       | Feature  | Continuous | Several measures of variation in amplitude                                           |
| 12 | Shimmer(dB)   | Feature  | Continuous | Several measures of variation in amplitude                                           |
| 13 | Shimmer:APQ3  | Feature  | Continuous | Several measures of variation in amplitude                                           |
| 14 | Shimmer:APQ5  | Feature  | Continuous | Several measures of variation in amplitude                                           |
| 15 | Shimmer:APQ11 | Feature  | Continuous | Several measures of variation in amplitude                                           |
| 16 | Shimmer:DDA   | Feature  | Continuous | Several measures of variation in amplitude                                           |
| 17 | NHR           | Feature  | Continuous | Two measures of ratio of noise to tonal components in the voice                      |
| 18 | HNR           | Feature  | Continuous | Two measures of ratio of noise to tonal components in the voice                      |
| 19 | RPDE          | Feature  | Continuous | A nonlinear dynamical complexity measure                                             |
| 20 | DFA           | Feature  | Continuous | Signal fractal scaling exponent                                                      |
| 21 | PPE           | Feature  | Continuous | A nonlinear measure of fundamental frequency variation                               |
"""
