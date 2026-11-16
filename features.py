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
    iu = np.triu_indices(len(eeg_names))
    return row, cov[:, iu[0], iu[1]].astype(np.float32)

