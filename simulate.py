"""Synthetic results folder with a known answer, to check analysis.py.

    python simulate.py sim && python analysis.py sim

Planted truth: eye features carry only a video fingerprint (like luminance);
skin conductance carries only the person's own arousal deviation; EEG a weak
mix; heart and breathing are pure noise. A correct analysis must find exactly
that: an eye fingerprint (E1) that collapses on unseen videos (E2), and an
individual arousal signal in skin conductance (E3, E4), nothing for valence.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "sim")
out.mkdir(exist_ok=True)
rng = np.random.default_rng(1)
videos = [f"theme{t}_{i}" for t in range(84) for i in (1, 2, 3)]
fingerprint = {v: rng.normal(size=20) for v in videos}
mean_arousal = {v: rng.normal(5, 1.5) for v in videos}
mean_valence = {v: rng.normal(5, 1.5) for v in videos}

labels, feats = [], []
for p in range(200):
    pid, bias = f"P{p:03d}", rng.normal(0, 1)
    for j, v in enumerate(rng.choice(videos, 45, replace=False)):
        dev = rng.normal(0, 1.2)                                  # the person's own feeling
        labels.append(dict(participant_id=pid, day=1, start_marker=10000 + j, video=v,
                           video_theme=v.rsplit("_", 1)[0], cohort=1, flagged_outliers=0,
                           final_emotion_label="joy",
                           rating_arousal=np.clip(mean_arousal[v] + bias + dev, 1, 9),
                           rating_valence=np.clip(mean_valence[v] + rng.normal(0, 1.2), 1, 9)))
        row = dict(participant_id=pid, day=1, start_marker=10000 + j, onset_s=0, source="SIM")
        base = rng.normal(size=20)
        for i in range(20):
            row[f"eye_video_f{i}"], row[f"eye_base_f{i}"] = fingerprint[v][i] + rng.normal() + base[i], base[i]
            row[f"gsr_video_f{i}"] = (0.15 * dev if i < 5 else 0) + rng.normal()
            row[f"eeg_video_f{i}"] = 0.05 * dev + 0.1 * fingerprint[v][i] + rng.normal()
            row[f"ecg_video_f{i}"] = rng.normal()
            row[f"resp_video_f{i}"] = rng.normal()
            for m in ("gsr", "eeg", "ecg", "resp"):
                row[f"{m}_base_f{i}"] = 0.0
        feats.append(row)

pd.DataFrame(labels).to_csv(out / "labels_train.csv", index=False)
pd.DataFrame(feats).to_csv(out / "features.csv.gz", index=False)
print(f"wrote {len(feats)} simulated trials to {out}/")
