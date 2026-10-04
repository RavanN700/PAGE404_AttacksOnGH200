#!/usr/bin/env python3
"""
Evaluation suite for the prompt-length side-channel attack.

Imports the detection/regression pipeline (linear_regression.py, aliased `D`)
and produces the metrics a security/side-channel paper needs, in three blocks:

  BLOCK 1 — ESTIMATION RIGOR
     k-fold cross-validation (mean ± std) for R², MAE, RMSE, MAPE.
     Baselines: predict-the-mean, and ratio-anchor-only (no wave detection).
     Residual-vs-true plot (exposes the high-token degeneracy honestly).

  BLOCK 2 — DETECTION QUALITY  (separate from estimation)
     Localization error |detected_start - ratio_anchor_start| on held-out
     streams, and a detection-failure rate (how often blind detection lands
     far from where the ratio says the wave is).

  BLOCK 3 — ATTACK REALISM
     Token-resolution: smallest reliably-distinguishable token gap.
     Traces-needed curve: accuracy vs number of repeated traces averaged.
     Per-class error table.

Run after linear_regression.py has populated its cache (fast). Standalone:
    python3 evaluate.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

import linear_regression as D    # the working pipeline

OUT = D.OUT_DIR / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 10


# ──────────────────────────────────────────────────────────────────────────
# Build the full feature table ONCE (detect every stream, blind = whole stream)
# ──────────────────────────────────────────────────────────────────────────
def build_table():
    """For every stream: blind-detect the wave (no ratio), record start/end,
    the ratio-anchor position (for detection ground-truth + a baseline), and
    the token label. Cached to a CSV so re-runs are instant."""
    cache = OUT / "feature_table.csv"
    if cache.exists():
        return pd.read_csv(cache)

    from concurrent.futures import ThreadPoolExecutor
    tasks = [(f"{c:02d}", r) for c in range(1, 21) for r in range(1, D.N_RUNS + 1)]

    def _job(args):
        cls, run = args
        raw  = D.load_raw(cls, run)
        meta = D.parse_metadata(cls, run)
        # BLIND detection (attacker view: no metadata)
        _, s_blind, e_blind, _ = D.find_wave(raw, 0, len(raw))
        # ratio anchor (defender ground-truth for detection eval + baseline)
        anc = D.ratio_pos(meta["avg_time"])
        # ratio-window detection (what train uses) for a localization reference
        _, s_anch, e_anch, _ = D.find_wave(raw, anc + D.SEARCH_LO, anc + D.SEARCH_HI)
        return {
            "class": int(cls), "run": run,
            "total_tokens": meta["input_tokens"] + meta["output_tokens"],
            "avg_time": meta["avg_time"],
            "start_blind": s_blind, "end_blind": e_blind,
            "start_anchor": s_anch, "end_anchor": e_anch,
            "ratio_pos": anc,
        }

    recs = []
    with ThreadPoolExecutor(max_workers=D.N_WORKERS) as ex:
        for i, r in enumerate(ex.map(_job, tasks)):
            recs.append(r)
            if (i + 1) % 200 == 0:
                print(f"  built {i+1}/{len(tasks)}")
    df = pd.DataFrame(recs)
    df.to_csv(cache, index=False)
    return df


def _metrics(y_true, y_pred):
    return {
        "R2":   r2_score(y_true, y_pred),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100),
    }


# ──────────────────────────────────────────────────────────────────────────
# BLOCK 1 — estimation rigor: k-fold CV + baselines
# ──────────────────────────────────────────────────────────────────────────
def block1_estimation(df):
    print("\n" + "=" * 70)
    print("BLOCK 1 — ESTIMATION RIGOR (k-fold CV, baselines)")
    print("=" * 70)

    X = df[["start_blind"]].values.astype(float)
    y = df["total_tokens"].values.astype(float)

    # Group k-fold by CLASS would leak (same tokens); we want generalization
    # across repeated traces, so standard k-fold over all 1420 streams.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

    model_m, mean_m, ratio_m = [], [], []
    all_true, all_pred = [], []
    for tr, te in kf.split(X):
        # our model: start_blind -> tokens
        m = LinearRegression().fit(X[tr], y[tr])
        p = m.predict(X[te])
        model_m.append(_metrics(y[te], p))
        all_true.append(y[te]); all_pred.append(p)

        # baseline 1: predict the training mean for everyone
        mean_pred = np.full_like(y[te], y[tr].mean())
        mean_m.append(_metrics(y[te], mean_pred))

        # baseline 2: ratio-anchor position -> tokens (NO wave detection)
        Xr = df[["ratio_pos"]].values.astype(float)
        mr = LinearRegression().fit(Xr[tr], y[tr])
        pr = mr.predict(Xr[te])
        ratio_m.append(_metrics(y[te], pr))

    def summarize(name, ms):
        keys = ["R2", "MAE", "RMSE", "MAPE"]
        out = {k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms])) for k in keys}
        print(f"\n  {name}")
        for k in keys:
            mu, sd = out[k]
            unit = "%" if k == "MAPE" else ""
            print(f"    {k:5s} = {mu:8.4f} ± {sd:.4f}{unit}")
        return out

    print(f"\n  {N_FOLDS}-fold cross-validation (mean ± std over folds):")
    res_model = summarize("OUR MODEL  (wave-start → tokens)", model_m)
    res_ratio = summarize("BASELINE   (ratio-anchor pos → tokens, no detection)", ratio_m)
    res_mean  = summarize("BASELINE   (predict mean tokens)", mean_m)

    # residual-vs-true plot (exposes high-token degeneracy)
    yt = np.concatenate(all_true); yp = np.concatenate(all_pred)
    resid = yp - yt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="k", lw=1)
    ax.scatter(yt, resid, s=18, alpha=0.4, color="royalblue")
    # binned mean abs error
    bins = np.linspace(yt.min(), yt.max(), 12)
    bi = np.digitize(yt, bins)
    bx = [yt[bi == b].mean() for b in range(1, len(bins)) if (bi == b).any()]
    by = [np.abs(resid[bi == b]).mean() for b in range(1, len(bins)) if (bi == b).any()]
    ax.plot(bx, by, "r-o", lw=1.5, label="binned MAE")
    ax.set_xlabel("true total_tokens"); ax.set_ylabel("residual (est − true)")
    ax.set_title("Residual vs true token count (CV pooled)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "residuals.pdf", dpi=150); plt.close(fig)

    return res_model, res_ratio, res_mean


# ──────────────────────────────────────────────────────────────────────────
# BLOCK 2 — detection quality (independent of regression)
# ──────────────────────────────────────────────────────────────────────────
def block2_detection(df, fail_thresh=20_000):
    print("\n" + "=" * 70)
    print("BLOCK 2 — DETECTION QUALITY")
    print("=" * 70)

    # localization error: blind detection vs ratio-window detection (reference)
    loc_err = np.abs(df["start_blind"] - df["start_anchor"])
    print(f"\n  Blind-vs-anchored localization error (samples):")
    for q in [50, 75, 90, 95]:
        print(f"    p{q:<2d} = {np.percentile(loc_err, q):>9.0f}")
    print(f"    mean = {loc_err.mean():>8.0f}   max = {loc_err.max():>8.0f}")

    # detection FAILURE: blind detection lands far from the ratio anchor region
    fail = loc_err > fail_thresh
    print(f"\n  Detection-failure rate (|blind − anchored| > {fail_thresh:,} samples):")
    print(f"    {fail.mean()*100:.2f}%  ({fail.sum()}/{len(df)} streams)")
    # per class
    pc = df.assign(fail=fail).groupby("class")["fail"].mean() * 100
    bad = pc[pc > 5]
    if len(bad):
        print("    classes with >5% failure:")
        for c, v in bad.items():
            print(f"      class {c:02d}: {v:.1f}%")
    else:
        print("    no class exceeds 5% failure")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(loc_err, bins=60, color="seagreen", alpha=0.8)
    ax.axvline(fail_thresh, color="red", ls="--", label=f"failure thresh ({fail_thresh:,})")
    ax.set_xlabel("|blind start − anchored start|  (samples)")
    ax.set_ylabel("streams"); ax.set_title("Detection localization error")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "localization_error.pdf", dpi=150); plt.close(fig)

    return loc_err, fail


# ──────────────────────────────────────────────────────────────────────────
# BLOCK 3 — attack realism
# ──────────────────────────────────────────────────────────────────────────
def block3_attack(df):
    print("\n" + "=" * 70)
    print("BLOCK 3 — ATTACK REALISM")
    print("=" * 70)

    X = df[["start_blind"]].values.astype(float)
    y = df["total_tokens"].values.astype(float)

    # fit on a train split, evaluate attack properties on held-out
    from sklearn.model_selection import train_test_split
    itr, ite = train_test_split(np.arange(len(df)), test_size=0.3,
                                random_state=0, stratify=df["class"])
    model = LinearRegression().fit(X[itr], y[itr])
    test = df.iloc[ite].copy()
    test["est"] = model.predict(X[ite])
    test["abs_err"] = np.abs(test["est"] - test["total_tokens"])

    # ── per-class error table ────────────────────────────────────────────────
    print("\n  Per-class attack accuracy (held-out):")
    print(f"  {'cls':>3} {'true_tok':>8} {'mean_est':>8} {'MAE':>6} {'±tok 90%':>9}")
    pc = test.groupby("class").agg(
        true_tok=("total_tokens", "first"),
        mean_est=("est", "mean"),
        mae=("abs_err", "mean"),
        p90=("abs_err", lambda v: np.percentile(v, 90)),
    )
    for c, r in pc.iterrows():
        print(f"  {c:>3} {int(r['true_tok']):>8} {r['mean_est']:>8.1f} "
              f"{r['mae']:>6.1f} {r['p90']:>9.1f}")

    # ── token resolution: smallest distinguishable gap ───────────────────────
    # Two classes are "distinguishable" if their estimate distributions barely
    # overlap. Use 1.96σ separation (≈95% non-overlap) as the criterion.
    print("\n  Token resolution (smallest reliably-separated gap):")
    stats = test.groupby("class").agg(tok=("total_tokens", "first"),
                                      mu=("est", "mean"), sd=("est", "std"))
    stats = stats.sort_values("tok")
    gaps = []
    arr = stats.reset_index().to_dict("records")
    for a, b in zip(arr, arr[1:]):
        tok_gap = b["tok"] - a["tok"]
        sep = abs(b["mu"] - a["mu"]) / (1.96 * (a["sd"] + b["sd"]) / 2 + 1e-9)
        gaps.append((a["tok"], b["tok"], tok_gap, sep))
    distinguishable = [g for g in gaps if g[3] >= 1.0]
    if distinguishable:
        min_gap = min(g[2] for g in distinguishable)
        print(f"    smallest reliably-distinguishable adjacent token gap: {min_gap} tokens")
    print("    adjacent-class separability (sep>=1.0 ≈ 95% non-overlap):")
    for a_t, b_t, g, sep in gaps:
        mark = "OK " if sep >= 1.0 else "   "
        print(f"      {mark}{int(a_t):>3} → {int(b_t):>3} tok (Δ{int(g):>3}): sep={sep:.2f}")

    # ── traces-needed curve: average N repeats of the SAME class ─────────────
    print("\n  Traces-needed: accuracy when averaging N repeated traces…")
    rng = np.random.default_rng(0)
    Ns = [n for n in [1, 2, 3, 5, 10, 20] if n <= D.N_RUNS]
    curve = []
    for N in Ns:
        errs = []
        for c in range(1, 21):
            sub = df[df["class"] == c]
            if len(sub) < N:
                continue
            true_tok = sub["total_tokens"].iloc[0]
            for _ in range(200):              # bootstrap
                pick = sub.sample(N, replace=True, random_state=rng.integers(1e9))
                start_avg = pick["start_blind"].mean()
                est = float(model.predict([[start_avg]])[0])
                errs.append(abs(est - true_tok))
        curve.append((N, np.mean(errs)))
        print(f"    N={N:>2} traces:  MAE = {np.mean(errs):.2f} tokens")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([c[0] for c in curve], [c[1] for c in curve], "o-", color="purple")
    ax.set_xlabel("number of repeated traces averaged")
    ax.set_ylabel("MAE (tokens)")
    ax.set_title("Attack accuracy vs traces collected")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "traces_needed.pdf", dpi=150); plt.close(fig)

    # per-class error bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(pc.index.astype(str), pc["mae"], color="steelblue", alpha=0.8)
    ax.set_xlabel("class"); ax.set_ylabel("MAE (tokens)")
    ax.set_title("Per-class attack error (held-out)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "per_class_error.pdf", dpi=150); plt.close(fig)

    return pc, gaps, curve


# ──────────────────────────────────────────────────────────────────────────
# BONUS — parameter sensitivity sweep (robustness, anti-overfit evidence)
# ──────────────────────────────────────────────────────────────────────────
def block_sensitivity():
    print("\n" + "=" * 70)
    print("SENSITIVITY — MAE vs detector thresholds (subset of streams)")
    print("=" * 70)
    # re-detect a subset under varying OSC_THR / SUSTAIN_FRAC
    n_sub = min(15, D.N_RUNS)
    subset = [(f"{c:02d}", r) for c in range(1, 21) for r in range(1, n_sub + 1)]
    raws = {(c, r): D.load_raw(c, r) for c, r in subset}
    metas = {(c, r): D.parse_metadata(c, r) for c, r in subset}
    y = np.array([metas[k]["input_tokens"] + metas[k]["output_tokens"] for k in raws])

    base_osc, base_sus = D.OSC_THR, D.SUSTAIN_FRAC
    print("\n  OSC_THR sweep:")
    for val in [4, 5, 6, 7, 8, 10]:
        D.OSC_THR = val
        starts = np.array([D.find_wave(raws[k], 0, len(raws[k]))[1] for k in raws])
        r, _ = pearsonr(starts, y)
        print(f"    OSC_THR={val:>4}:  start↔tokens r = {r:.4f}")
    D.OSC_THR = base_osc

    print("\n  SUSTAIN_FRAC sweep:")
    for val in [0.25, 0.3, 0.4, 0.5, 0.6]:
        D.SUSTAIN_FRAC = val
        starts = np.array([D.find_wave(raws[k], 0, len(raws[k]))[1] for k in raws])
        r, _ = pearsonr(starts, y)
        print(f"    SUSTAIN_FRAC={val}:  start↔tokens r = {r:.4f}")
    D.SUSTAIN_FRAC = base_sus


def main():
    print("Building feature table (blind detection on all streams)…")
    df = build_table()
    print(f"  {len(df)} streams.")

    block1_estimation(df)
    block2_detection(df)
    block3_attack(df)
    block_sensitivity()

    print("\n" + "=" * 70)
    print(f"Plots → {OUT}/")
    print("  residuals.pdf, localization_error.pdf, traces_needed.pdf,")
    print("  per_class_error.pdf")
    print("=" * 70)


if __name__ == "__main__":
    main()