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
    # A start marker is unique per person; the labels' day column is unreliable
    # in cohort 4, so the recording day comes from the feature file.
    labels = labels.drop(columns="day").drop_duplicates(["participant_id", "start_marker"], keep=False)
    feats = feats.copy().assign(cov_row=np.arange(len(feats)))                     # row in eeg_cov.npy
    feats = feats.drop_duplicates(["participant_id", "start_marker"], keep=False)   # P475: unlabelled anyway
    df = feats.merge(labels, on=["participant_id", "start_marker"], how="inner", validate="1:1")
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


def riemann_features(results: Path, df):
    """EEG covariances (theta, alpha, beta) mapped to the tangent space at each
    session's own mean covariance: a standard cross-subject EEG representation.
    Recentring per session uses no labels."""
    tri = np.load(results / "eeg_cov.npy", mmap_mode="r")
    n = int((np.sqrt(8 * tri.shape[2] + 1) - 1) / 2)
    iu = np.triu_indices(n)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))                 # keeps the Frobenius norm
    sess = (df.participant_id + "_" + df.day.astype(str)).to_numpy()
    out = np.zeros((len(df), tri.shape[1] * len(iu[0])), np.float32)

    def powm(C, f):
        e, V = np.linalg.eigh(C)
        return (V * f(e)[..., None, :]) @ np.swapaxes(V, -1, -2)

    for s in np.unique(sess):
        rows = np.flatnonzero(sess == s)
        for b in range(tri.shape[1]):
            C = np.zeros((len(rows), n, n))
            C[:, iu[0], iu[1]] = tri[df.cov_row.to_numpy()[rows], b]
            C = C + np.triu(C, 1).swapaxes(1, 2)
            C = C / np.trace(C, axis1=1, axis2=2)[:, None, None] + 1e-5 * np.eye(n)   # scale-free, regularised
            ref = powm(np.mean(C, axis=0), lambda e: e ** -0.5)
            T = powm(ref @ C @ ref, np.log)
            out[rows, b * len(iu[0]):(b + 1) * len(iu[0])] = T[:, iu[0], iu[1]] * w
    return out


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


def e4_synchrony(df, X, cols, n_perm=1000, min_viewers=5):
    """Does reacting like the crowd go with feeling like the crowd?

    For each trial: how far the person's response is from the average response
    of the other viewers of the same video (RMS distance of session-normalised
    features, log), and how far their rating is from the other viewers' mean
    rating (absolute). Score: within-person correlation of the two. Null:
    shuffle rating distances among viewers of the same video (keeps any
    video-level link, e.g. polarising videos), Holm-corrected afterwards.
    No model is trained, so nothing can leak."""
    vid, groups = df.video.to_numpy(), df.participant_id.to_numpy()
    n = pd.Series(vid).map(pd.Series(vid).value_counts()).to_numpy()
    ok = n >= min_viewers
    rows = []

    def loo(a):                                    # mean of the other viewers of the same video
        g = pd.DataFrame(a).groupby(vid)
        return (g.transform("sum").to_numpy() - a) / (n[:, None] - 1)

    for target in TARGETS:
        y = df[target].to_numpy(float)
        rating_gap = np.abs(y - loo(y[:, None])[:, 0])
        for name, mods in SETS.items():
            Z = X[:, col_index(cols, mods)].astype(float)
            react_gap = np.log(np.sqrt(((Z - loo(Z)) ** 2).mean(axis=1)) + 1e-6)
            r = within_r(rating_gap[ok], react_gap[ok], groups[ok])
            null = [within_r(shuffle_within(rating_gap[ok], vid[ok]), react_gap[ok], groups[ok])
                    for _ in range(n_perm)]
            rows.append({"target": target, "features": name, "r_sync": r,
                         "p_sync": (1 + np.sum(np.array(null) >= r)) / (n_perm + 1),
                         "null_mean": float(np.mean(null)), "trials": int(ok.sum())})
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out["q_sync"] = holm(out.p_sync)
    return out


def fit_predict(Xtr, ytr, Xte, model="ridge"):
    """Ridge (main model) or gradient-boosted trees (nonlinear check)."""
    if model == "trees":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0)
    else:
        m = RidgeCV(alphas=np.logspace(0, 5, 11))
    return m.fit(Xtr, ytr).predict(Xte)


def e2_e3(df, X, cols, n_perm=0, sets=SETS, model="ridge"):
    """Seen-vs-unseen rating decoding (E2) and individual-deviation decoding (E3)."""
    pf, tf = folds(df)
    groups = df.participant_id.to_numpy()
    rows = []
    for target in TARGETS:
        y = df[target].to_numpy(float)
        for name in sets:
            Xm = X[:, col_index(cols, SETS[name])]
            pred = {k: np.full(len(df), np.nan) for k in ("seen", "unseen", "dev_seen", "dev_unseen",
                                                          "video_mean_seen")}
            dev_true = {"dev_seen": np.full(len(df), np.nan), "dev_unseen": np.full(len(df), np.nan)}
            for k in range(5):
                train = (pf != k) & (tf != k)
                seen, unseen = (pf == k) & (tf != k), (pf == k) & (tf == k)
                p = fit_predict(Xm[train], y[train], Xm[seen | unseen], model)
                pred["seen"][seen | unseen] = p
                d = video_deviation(df, y, train)
                ok = train & np.isfinite(d)
                pd_ = fit_predict(Xm[ok], d[ok], Xm[seen | unseen], model)
                for key, m in (("dev_seen", seen), ("dev_unseen", unseen)):
                    pred[key][m] = pd_[m[seen | unseen]]
                    dev_true[key][m] = d[m]
                vm = pd.Series(y[train]).groupby(df.video.to_numpy()[train]).mean()
                pred["video_mean_seen"][seen] = df.video[seen].map(vm).to_numpy()
            seen_all, unseen_all = tf != pf, tf == pf
            res = {
                "target": target, "features": name, "model": model,
                "r_seen_videos": within_r(y[seen_all], pred["seen"][seen_all], groups[seen_all]),
                "r_unseen_videos": within_r(y[unseen_all], pred["seen"][unseen_all], groups[unseen_all]),
            }
            for key, m in (("dev_seen", seen_all), ("dev_unseen", unseen_all)):
                ok = m & np.isfinite(dev_true[key])
                res[f"r_{key}"] = within_r(dev_true[key][ok], pred[key][ok], groups[ok])
                if n_perm:
                    vids = df.video.to_numpy()[ok]
                    null = [within_r(shuffle_within(dev_true[key][ok], vids), pred[key][ok], groups[ok])
                            for _ in range(n_perm)]
                    res[f"p_{key}"] = (1 + np.sum(np.array(null) >= res[f"r_{key}"])) / (n_perm + 1)
            if name == "All":
                ok = seen_all & np.isfinite(pred["video_mean_seen"])
                res["r_video_mean_baseline"] = within_r(y[ok], pred["video_mean_seen"][ok], groups[ok])
            rows.append(res)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    return pd.DataFrame(rows)


def robustness(results: Path):
    """Same E2/E3 scores (all signals, ridge) within each cohort.
    (No trial is flagged in this release, so flagged_outliers changes nothing.)"""
    df, X, cols = load(results)
    rows = []
    for c in sorted(df.cohort.unique()):
        m = (df.cohort == c).to_numpy()
        if df[m].participant_id.nunique() >= 25:
            r = e2_e3(df[m].reset_index(drop=True), X[m], cols, sets=["All"])
            rows.append(r.assign(cohort=c, n_trials=int(m.sum()), n_people=df[m].participant_id.nunique()))
    return pd.concat(rows)


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def figures(e1, e23, out: Path, e4=None):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})
    BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
    names = list(SETS)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    d = e1.set_index("features").loc[names[::-1]]
    ax.barh(d.index, d.same_video_auc - 0.5, left=0.5, color=BLUE, height=0.55)
    ax.axvline(0.5, color=GRAY, lw=1, ls="--")
    pad = 0.02 * (d.same_video_auc.max() - 0.5)
    for y, v in enumerate(d.same_video_auc):
        ax.text(max(v, 0.5) + pad, y, f"{v:.3f}", va="center", color="#52514e")
    ax.set_xlim(0.5, 0.5 + 1.15 * (d.same_video_auc.max() - 0.5))
    ax.set_xlabel("same-video AUC (0.5 = no fingerprint)")
    ax.set_title("E1  Do two people's signals match\nwhen they watch the same video?", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_fingerprint.png", dpi=200)

    for fig_name, cols, labels, title in (
        ("fig2_seen_vs_unseen.png", ("r_seen_videos", "r_unseen_videos"), ("seen videos", "unseen videos"),
         "E2  Rating decoding on new people: videos seen vs never seen in training"),
        ("fig3_individual.png", ("r_dev_seen", "r_dev_unseen"), ("seen videos", "unseen videos"),
         "E3  Decoding how a person's rating differs from others on the same video"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
        for ax, target in zip(axes, TARGETS):
            d = e23[e23.target == target].set_index("features").loc[names]
            x = np.arange(len(names))
            for off, c, lab, col in ((-0.17, BLUE, labels[0], cols[0]), (0.17, ORANGE, labels[1], cols[1])):
                ax.bar(x + off, d[col], width=0.32, color=c, label=lab)
                q = d.get(col.replace("r_", "q_"))
                if q is not None:
                    for xi, (r, qi) in enumerate(zip(d[col], q)):
                        if qi < 0.05:
                            ax.text(xi + off, max(r, 0) + 0.004, "*", ha="center", color="#0b0b0b")
            ax.axhline(0, color=GRAY, lw=1)
            ax.set_xticks(x, names, rotation=0)
            ax.set_title(target.replace("rating_", ""), loc="left", fontsize=9)
        axes[0].set_ylabel("within-person r")
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=2, loc="lower center")
        fig.suptitle(title + ("  (* Holm p < .05)" if "dev" in cols[0] else ""), x=0.01, ha="left", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        fig.savefig(out / fig_name, dpi=200)
    if e4 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
        for ax, target in zip(axes, TARGETS):
            d = e4[e4.target == target].set_index("features").loc[names]
            x = np.arange(len(names))
            ax.bar(x, d.r_sync, width=0.55, color=BLUE, label="observed")
            ax.scatter(x, d.null_mean, marker="_", s=260, color="#0b0b0b", zorder=3, label="expected by chance")
            for xi, (r, q) in enumerate(zip(d.r_sync, d.q_sync)):
                if q < 0.05:
                    ax.text(xi, max(r, 0) + 0.002, "*", ha="center")
            ax.axhline(0, color=GRAY, lw=1)
            ax.set_xticks(x, names)
            ax.set_title(target.replace("rating_", ""), loc="left", fontsize=9)
        axes[0].set_ylabel("within-person r")
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=2, loc="lower center")
        fig.suptitle("E4  Do people who react unlike the crowd also rate unlike the crowd?", x=0.01, ha="left",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        fig.savefig(out / "fig4_synchrony.png", dpi=200)
    plt.close("all")


def main(results: Path, n_perm=1000):
    df, X, cols = load(results)
    out = results / "analysis"
    out.mkdir(exist_ok=True)
    e1 = e1_fingerprint(df, X, cols)
    print(e1.round(3).to_string())
    e1.to_csv(out / "e1_fingerprint.csv", index=False)
    e23 = e2_e3(df, X, cols, n_perm=n_perm)
    if n_perm:
        p = e23[["p_dev_seen", "p_dev_unseen"]].to_numpy().ravel()
        e23[["q_dev_seen", "q_dev_unseen"]] = holm(p).reshape(-1, 2)
    e23.to_csv(out / "e2_e3_decoding.csv", index=False)
    e4 = e4_synchrony(df, X, cols, n_perm=n_perm)
    e4.to_csv(out / "e4_synchrony.csv", index=False)
    robustness(results).to_csv(out / "robustness_by_cohort.csv", index=False)
    figures(e1, e23, Path("figures"), e4)
    return e1, e23, e4


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
