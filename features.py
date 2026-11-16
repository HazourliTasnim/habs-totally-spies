"""Per-trial physiological features for one emotion-video recording.

Each video trial is cut into a 10 s *video* window [0, 10) s after onset and a
3 s *baseline* window [-3, 0) s just before it. For every modality we compute
interpretable scalar features in both windows, plus their difference
(video - baseline), so that slow drifts and electrode offsets cancel.

Modality      Features (per window)                          Source
-----------   --------------------------------------------   ------------------
EEG (62 ch)   log band power: delta theta alpha beta gamma   PREPROCESSED (ICA)
              + per-band spatial covariance (theta/alpha/beta, video only)
ECG           heart rate mean/slope, SDNN, RMSSD             R-peak detection
GSR (uS)      tonic level, max phasic rise, slope            10 Hz, low-passed
Respiration   amplitude (SD), rate (peaks/min)               0.1-1 Hz band
Pupil         diameter median/p90/slope, blink-dip fraction  native ~166 Hz
Gaze          dispersion (SD of x, y), mean speed            native ~166 Hz

Everything is computed from the trial's own samples; no information from
other trials, participants or labels enters a feature.
"""

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

FS = 1000                                   # master sampling rate (Hz)
BASE, VIDEO = (-3.0, 0.0), (0.0, 10.0)      # windows relative to video onset (s)
PAD = 5.0                                   # extra context cut around a trial (s)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
COV_BANDS = ("theta", "alpha", "beta")


def _sos(lo, hi, fs, order=4):
    if lo is None:
        return butter(order, hi, "lowpass", fs=fs, output="sos")
    return butter(order, (lo, hi), "bandpass", fs=fs, output="sos")


def _win(t, w):
    return (t >= w[0]) & (t < w[1])


def eeg_features(x, t):
    """x: (62, n) in uV on time axis t. Log band power per channel and window."""
    out = {}
    for wname, w in (("base", BASE), ("video", VIDEO)):
        seg = x[:, _win(t, w)]
        f, p = welch(seg, fs=FS, nperseg=min(1000, seg.shape[1]), axis=1)
        for b, (lo, hi) in BANDS.items():
            out[(wname, b)] = np.log10(p[:, (f >= lo) & (f < hi)].mean(axis=1) + 1e-12)
    covs = []
    for b in COV_BANDS:
        y = sosfiltfilt(_sos(*BANDS[b], FS), x, axis=1)[:, _win(t, VIDEO)]
        covs.append(np.cov(y))
    return out, np.stack(covs)


def ecg_features(ecg, t):
    """Heart rate from R peaks; polarity chosen so that R peaks are positive."""
    y = sosfiltfilt(_sos(5, 30, FS), ecg)
    if np.abs(np.percentile(y, 0.5)) > np.abs(np.percentile(y, 99.5)):
        y = -y
    peaks, _ = find_peaks(y, distance=int(0.33 * FS), height=0.4 * np.percentile(y, 99))
    out = {}
    if len(peaks) < 4:
        return {f"{w}_{k}": np.nan for w in ("base", "video") for k in ("hr", "hr_slope", "sdnn", "rmssd")}
    pt, ibi = t[peaks[1:]], np.diff(peaks) / FS
    med = np.median(ibi)
    ok = (ibi > 0.33) & (ibi < 1.5) & (np.abs(ibi - med) < 0.3 * med)   # 40-180 bpm, no missed/extra beats
    pt, ibi = pt[ok], ibi[ok]
    for wname, w in (("base", BASE), ("video", VIDEO)):
        m = _win(pt, w)
        hr = 60 / ibi[m]
        out[f"{wname}_hr"] = hr.mean() if m.sum() else np.nan
        out[f"{wname}_hr_slope"] = np.polyfit(pt[m], hr, 1)[0] if m.sum() > 2 else np.nan
        out[f"{wname}_sdnn"] = ibi[m].std() * 1e3 if m.sum() > 2 else np.nan
        out[f"{wname}_rmssd"] = np.sqrt(np.mean(np.diff(ibi[m]) ** 2)) * 1e3 if m.sum() > 2 else np.nan
    return out


def gsr_features(gsr, t):
    """Skin conductance (uS): level, steepest rise and slope, at 10 Hz."""
    y = sosfiltfilt(_sos(None, 1.0, FS, 2), gsr)[::100]
    tt = t[::100]
    out = {}
    for wname, w in (("base", BASE), ("video", VIDEO)):
        s = y[_win(tt, w)]
        out[f"{wname}_scl"] = s.mean()
        out[f"{wname}_slope"] = np.polyfit(np.arange(len(s)) / 10, s, 1)[0]
        out[f"{wname}_rise"] = (s - np.minimum.accumulate(s)).max()   # largest rise from a prior trough
    return out


def resp_features(resp, t):
    y = sosfiltfilt(_sos(0.1, 1.0, FS, 2), resp)[::100]
    tt = t[::100]
    out = {}
    for wname, w in (("base", BASE), ("video", VIDEO)):
        s = y[_win(tt, w)]
        out[f"{wname}_amp"] = s.std()
        pk, _ = find_peaks(s, distance=15, prominence=0.3 * s.std() if s.std() > 0 else None)
        out[f"{wname}_rate"] = 60 * len(pk) / (w[1] - w[0])
    return out


def trial_features(df, eeg_names, onset):
    """All scalar features (dict) and EEG covariances for one trial."""
    i0, i1 = int((onset + BASE[0] - PAD) * FS), int((onset + VIDEO[1] + PAD) * FS)
    if i0 < 0 or i1 > len(df):
        return None, None
    seg = df.iloc[i0:i1]
    t = np.arange(i1 - i0) / FS + BASE[0] - PAD
    row = {}
    eeg, cov = eeg_features(seg[eeg_names].to_numpy().T, t)
    for (w, b), v in eeg.items():
        row.update({f"eeg_{w}_{b}_{ch}": val for ch, val in zip(eeg_names, v)})
    for prefix, feats in (
        ("ecg", ecg_features(seg["ECG"].to_numpy(), t)),
        ("gsr", gsr_features(seg["GSR"].to_numpy(), t)),
        ("resp", resp_features(seg["Resp"].to_numpy(), t)),
    ):
        row.update({f"{prefix}_{k}": v for k, v in feats.items()})
    iu = np.triu_indices(len(eeg_names))
    return row, cov[:, iu[0], iu[1]].astype(np.float32)

