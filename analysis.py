"""Do brain and body signals show how each person feels, or which video they watched?

Runs locally on the tables written by process.py (downloaded with `habs get`):

    python analysis.py results/          # writes results/*.csv and figures/

Three experiments, every one evaluated on participants never seen in training:

E1  Stimulus fingerprint (model-free). For pairs of trials from *different*
    people, are physiological responses more similar when both watched the same
    video? AUC of feature correlation for same- vs different-video pairs.

E2  Seen vs unseen videos. Participants and video themes are each split into 5
    folds. For fold k the model trains on (other participants x other themes)
    and is tested on held-out participants twice: on themes it saw in training
    and on themes it never saw. Training data are identical, so the gap between
    the two scores is what stimulus recognition contributes.

E3  The individual component. Target = how far a person's rating of a video
    deviates from everyone else's rating of that same video. Predicting it on
    unseen people (and unseen themes) requires reading the person, not the video.

Score: correlation between prediction and rating computed *within* each test
participant (removes personal scale use), Fisher-averaged over participants.
E3 chance level: deviations are shuffled among people who watched the *same
video*, which keeps everything tied to the video (including rating-scale
ceiling effects) and destroys only who-felt-what.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

MODALITIES = {"EEG": "eeg_", "ECG": "ecg_", "GSR": "gsr_", "Resp": "resp_", "Eye": "eye_"}
SETS = {**{m: [m] for m in MODALITIES}, "Body": ["ECG", "GSR", "Resp"], "All": list(MODALITIES)}
TARGETS = ("rating_arousal", "rating_valence")
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------- data
def load(results: Path, include_flagged=False):
    """Trials with labels and session-normalised features. Returns (meta, X, columns_by_modality)."""
    feats = pd.read_csv(results / "features.csv.gz")
    labels = pd.read_csv(results / "labels_train.csv", na_values=["TESTSUBJECT"], low_memory=False)
    labels["day"] = pd.to_numeric(labels.day.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    labels = labels.drop_duplicates(["participant_id", "day", "start_marker"], keep=False)
    df = feats.merge(labels, on=["participant_id", "day", "start_marker"], how="inner", validate="1:1")
    print(f"{len(feats)} feature rows, {len(df)} matched to labels "
          f"({df.participant_id.nunique()} participants, {df.video.nunique()} videos)")
    df = df.dropna(subset=list(TARGETS))
    if not include_flagged:
        df = df[df.flagged_outliers.fillna(0).astype(float) != 1]
    df = df.reset_index(drop=True)

    # Response = video window and its change from the pre-video baseline.
    cols, blocks = {}, []
    for mod, prefix in MODALITIES.items():
        video = [c for c in df.columns if c.startswith(prefix + "video_")]
        v = df[video].to_numpy(float)
        b = df[[c.replace("_video_", "_base_") for c in video]].to_numpy(float)
        block = pd.DataFrame(np.hstack([v, v - b]),
                             columns=video + [c.replace("_video_", "_delta_") for c in video])
        cols[mod] = list(block.columns)
        blocks.append(block)
    X = pd.concat(blocks, axis=1)

    # Robust z-score within each recording session, using only that session's
    # own (unlabelled) trials: removes electrode impedance, skin and caffeine
    # differences between people without touching any rating.
    sess = df.participant_id + "_" + df.day.astype(str)
    g = X.groupby(sess.values)
    med = g.transform("median")
    iqr = g.transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
    X = ((X - med) / iqr.replace(0, np.nan)).clip(-5, 5).fillna(0)
    return df, X.to_numpy(np.float32), cols


def col_index(cols, mods):
    names = [c for m in mods for c in cols[m]]
    allcols = [c for m in MODALITIES for c in cols[m]]
    pos = {c: i for i, c in enumerate(allcols)}
    return np.array([pos[c] for c in names])


# ------------------------------------------------------------------ scoring
def within_r(y, p, groups):
    """Fisher-averaged within-participant Pearson r."""
    d = pd.DataFrame({"g": groups, "y": y, "p": p})
    d[["y", "p"]] -= d.groupby("g")[["y", "p"]].transform("mean")
    g = d.assign(yp=d.y * d.p, yy=d.y ** 2, pp=d.p ** 2, n=1).groupby("g")[["yp", "yy", "pp", "n"]].sum()
    g = g[(g.n >= 5) & (g.yy > 0) & (g.pp > 0)]
    rs = (g.yp / np.sqrt(g.yy * g.pp)).clip(-0.999, 0.999)
    return float(np.tanh(np.arctanh(rs).mean())) if len(rs) else np.nan


def shuffle_within(y, groups):
    y = y.copy()
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        y[m] = y[RNG.permutation(m)]
    return y


def folds(df, k=5, seed=0):
    rng = np.random.default_rng(seed)
    people = df.participant_id.unique()
    themes = df.video_theme.fillna(df.video).unique()
    pf = dict(zip(people, rng.permutation(len(people)) % k))
    tf = dict(zip(themes, rng.permutation(len(themes)) % k))
    return df.participant_id.map(pf).to_numpy(), df.video_theme.fillna(df.video).map(tf).to_numpy()


def video_deviation(df, y, train):
    """Rating minus the mean rating others gave the same video (others = training people for train rows)."""
    out = np.full(len(df), np.nan)
    vid = df.video.to_numpy()
    tr = pd.Series(y[train]).groupby(vid[train])
    s, n = tr.sum().reindex(vid[train]).to_numpy(), tr.count().reindex(vid[train]).to_numpy()
    allg = pd.Series(y).groupby(vid)
    S, N = allg.transform("sum").to_numpy(), allg.transform("count").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out[train] = y[train] - (s - y[train]) / (n - 1)
        out[~train] = y[~train] - (S[~train] - y[~train]) / (N[~train] - 1)    # leave-one-out over everyone
    out[~np.isfinite(out)] = np.nan
    return out


# -------------------------------------------------------------- experiments
def e1_fingerprint(df, X, cols, n_pairs=200_000):
    """AUC: are cross-person feature correlations higher for the same video?"""
    pid, vid = df.participant_id.to_numpy(), df.video.to_numpy()
    by_vid = pd.Series(np.arange(len(df))).groupby(vid).apply(np.array)
    by_vid = by_vid[by_vid.map(len) > 1]
    i = RNG.integers(0, len(by_vid), n_pairs)
    a = np.array([RNG.choice(by_vid.iloc[k]) for k in i])
    b = np.array([RNG.choice(by_vid.iloc[k]) for k in i])
    c = RNG.integers(0, len(df), n_pairs)
    ok = (pid[a] != pid[b]) & (pid[a] != pid[c]) & (vid[a] != vid[c])
    a, b, c = a[ok], b[ok], c[ok]
    rows = []
    for name, mods in SETS.items():
        Z = X[:, col_index(cols, mods)]
        Z = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-9)
        same, diff = (Z[a] * Z[b]).mean(1), (Z[a] * Z[c]).mean(1)
        auc = roc_auc_score(np.r_[np.ones(len(same)), np.zeros(len(diff))], np.r_[same, diff])
        rows.append({"features": name, "same_video_auc": auc, "pairs": len(same)})
    return pd.DataFrame(rows)


def main(results: Path):
    df, X, cols = load(results)
    out = results / "analysis"
    out.mkdir(exist_ok=True)
    e1 = e1_fingerprint(df, X, cols)
    print(e1.round(3).to_string())
    e1.to_csv(out / "e1_fingerprint.csv", index=False)
    return e1


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
