"""GPU-platform job: extract per-trial multimodal features for every video trial.

Run by the HABS platform as process(data_dir=/data, out_dir=...). Locally:
    habs run --data ./sample --out ./out

For every emotion-video recording (PREPROCESSED, i.e. ICA-cleaned EEG, falling
back to CURATED when absent) we find each labelled video onset in the BDF's own
annotations (marker code = labels_train.start_marker), compute features.py's
trial features, and write:

    out_dir/features.csv.gz   one row per trial: keys + ~1,300 features
    out_dir/eeg_cov.npy       (n_trials, 3 bands, 1953) EEG covariances, same row order
    out_dir/inventory.json    what was found, matched, skipped and why
    out_dir/*.csv             copies of the dataset's label tables

No label enters a feature: labels are used only to know where trials start.
"""

import json
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "starting_notebooks")]
from features import trial_features  # noqa: E402

BUDGET_S = 47 * 60          # stop scheduling new sessions after this (job limit is 1 h)
TASK = "emotion-video"


def available_memory_gb():
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(f).read_text().strip()
            if v.isdigit() and int(v) < 1 << 50:
                return int(v) / 1e9
        except OSError:
            pass
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def session_trials(bdf, markers):
    """Features for every labelled trial of one recording. Returns (rows, covs, log)."""
    import mne
    from bdf_loader import load_bdf
    from scipy.signal import butter, sosfiltfilt
    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")

    raw, df, channels = load_bdf(bdf)
    eeg = channels.loc[channels.mne_type == "eeg", "bdf_label"].tolist()
    eeg = [c for c in eeg if c not in ("T9", "T10")]          # as in PREPROCESSED
    if "CURATED" in bdf.parts:                                  # minimal equivalent cleaning
        x = df[eeg].to_numpy().T
        x = sosfiltfilt(butter(4, (0.5, 48), "bandpass", fs=1000, output="sos"), x, axis=1)
        df[eeg] = (x - x.mean(axis=0)).T

    onsets = {}
    for a in raw.annotations:
        onsets.setdefault(a["description"].strip(), []).append(float(a["onset"]))
    rows, covs, log = [], [], {"missing_marker": [], "repeated_marker": [], "out_of_range": []}
    for m in markers:
        hits = onsets.get(str(int(m)), [])
        if not hits:                                  # usually a trial from another day
            log["missing_marker"].append(int(m))
            continue
        if len(hits) > 1:
            log["repeated_marker"].append(int(m))
        row, cov = trial_features(df, eeg, hits[0])
        if row is None:
            log["out_of_range"].append(int(m))
            continue
        rows.append({"start_marker": int(m), "onset_s": hits[0], **row})
        covs.append(cov)
    log["eeg_channels"] = len(eeg)
    log["seconds"] = len(df) / 1000
    log["annotation_codes"] = sorted(onsets)[:200]
    return rows, covs, log


def run_session(args):
    bdf, pid, day, markers = args
    t = time.time()
    try:
        rows, covs, log = session_trials(Path(bdf), markers)
    except Exception:
        return pid, day, bdf, [], [], {"error": traceback.format_exc(limit=3)}, time.time() - t
    for r in rows:
        r.update(participant_id=pid, day=day, source=Path(bdf).parts[-3])
    return pid, day, bdf, rows, covs, log, time.time() - t


def process(data_dir: Path, out_dir: Path) -> None:
    t0 = time.time()
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    for name in ("labels_train.csv", "participants_train.csv", "markers.csv"):
        if (data_dir / name).is_file():
            (out_dir / name).write_bytes((data_dir / name).read_bytes())

    labels_path = data_dir / "labels_train.csv"
    labels = pd.read_csv(labels_path, na_values=["TESTSUBJECT"]) if labels_path.is_file() else None
    if labels is not None:
        labels["day"] = pd.to_numeric(labels.day.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        print(f"labels_train.csv: {len(labels)} rows, {labels.final_emotion_label.notna().sum()} labelled, "
              f"columns {list(labels.columns)}\n{labels.head(3).to_string()}", flush=True)
    counts = {}
    for p in data_dir.glob("*/P*/P*.bdf"):
        k = f"{p.parts[-3]} {p.stem.split('_task-')[1].split('_')[0]}"
        counts[k] = counts.get(k, 0) + 1
    print("BDF files:", json.dumps(counts), flush=True)

    # One job per recording, preferring the ICA-cleaned PREPROCESSED copy.
    found = {}
    for source in ("CURATED", "PREPROCESSED"):                 # PREPROCESSED overwrites
        for bdf in data_dir.glob(f"{source}/P*/P*_day-*_task-{TASK}_multimodal.bdf"):
            pid, day = bdf.name.split("_")[0], int(bdf.name.split("_")[1].split("-")[1])
            found[(pid, day)] = bdf
    skip = set()
    if (HERE / "done_sessions.txt").is_file():
        skip = {tuple(l.split()) for l in (HERE / "done_sessions.txt").read_text().split("\n") if l.strip()}

    jobs = []
    for (pid, day), bdf in sorted(found.items()):
        if (pid, str(day)) in skip:
            continue
        if labels is not None:
            # Match on participant only: in cohort 4 the labels' day column does
            # not match the recording day, but a start marker is unique per person.
            m = labels[labels.participant_id == pid]
            markers = sorted(m.start_marker.dropna().astype(int).unique())
        else:
            markers = []
        if not markers:                                        # fall back to the session metadata
            meta = bdf.parent / f"{pid}_day-{day}_metadata.json"
            if meta.is_file():
                events = json.loads(meta.read_text())["events"]
                markers = sorted({e["marker_id"] for e in events if e.get("trial_type") == "emotion_video"})
        jobs.append((str(bdf), pid, day, markers))
    if labels is None and not jobs:                            # sample/ has no video file: smoke-test on artefact
        for bdf in sorted(data_dir.glob("*/P*/P*_task-artefact_multimodal.bdf"))[-1:]:
            jobs.append((str(bdf), bdf.name[:4], 1, [10000, 10006, 10008, 10016]))

    mem, cpus = available_memory_gb(), os.cpu_count() or 1
    workers = max(1, min(cpus - 1, int(mem // 3.5), 48))
    print(f"{len(found)} {TASK} recordings, {len(jobs)} to do; {cpus} CPUs, {mem:.0f} GB -> {workers} workers",
          flush=True)

    rows, covs, inventory = [], [], {}
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(run_session, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            pid, day, bdf, r, c, log, dt = f.result()
            rows += r
            covs += c
            inventory[f"{pid} {day}"] = {"bdf": bdf, "trials": len(r), "secs": round(dt, 1), **log}
            if i % 10 == 0 or i == len(jobs) or "error" in log:
                print(f"[{time.time() - t0:6.0f}s] {i}/{len(jobs)} sessions, {len(rows)} trials"
                      + (f"  ERROR {pid} day {day}: {log['error'][-300:]}" if "error" in log else ""), flush=True)
            if time.time() - t0 > BUDGET_S:
                print("time budget reached: cancelling the remaining sessions", flush=True)
                for g in futures:
                    g.cancel()
                break

    feats = pd.DataFrame(rows)
    keys = ["participant_id", "day", "start_marker", "onset_s", "source"]
    feats = feats[keys + [c for c in feats.columns if c not in keys]]
    feats.to_csv(out_dir / "features.csv.gz", index=False, float_format="%.6g")
    np.save(out_dir / "eeg_cov.npy", np.stack(covs) if covs else np.zeros((0, 3, 0), np.float32))
    (out_dir / "inventory.json").write_text(json.dumps(inventory, indent=1, default=str))
    print(f"wrote {len(feats)} trials x {feats.shape[1]} columns from {len(inventory)} sessions "
          f"in {time.time() - t0:.0f}s", flush=True)
