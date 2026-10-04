#!/usr/bin/env python3
"""
LLM token estimator — RAW-SAMPLE wave detection.

This is the detection + regression pipeline for the prompt-length side channel.
`evaluate.py` imports it (as `D`) and reuses its loaders, detector, and config.

==========================================================================
Data source
==========================================================================
Consumes the traces produced by the collection step (auto_collection.py):
    data/counter/<cls>/<cls>_<run>.txt   raw access-counter stream  (load_raw)
    data/metadata/<cls>/<cls>_<run>.txt  input/output tokens, avg_time (parse_metadata)
Parsed streams are cached as .npy under data/_npy_cache/; figures/outputs go to
results/. These directories are created/populated when you run the pipeline —
only the code is shipped, not the data.

==========================================================================
Why raw samples (not an averaged envelope)
==========================================================================
The end-of-activity WAVE is a narrow feature: ~a few hundred raw samples
wide, amplitude oscillating ~30-60 on a baseline ~20, with a tall spike.
Averaging the 737k-sample stream into a few hundred envelope bins smears
the wave into the noise floor (each bin averages ~1200 samples). So we work
on RAW samples and look at the data the way a human does:
    window = 1600 samples wide, amplitude viewed over 0-120.

==========================================================================
Recording model
==========================================================================
Fixed TOTAL_REC_TIME = 47.3 s window, ~N_LINES samples. Inference lasts
avg_time s, so the end-of-activity wave sits near
    ratio_pos = (avg_time / 47.3) * N_LINES.

==========================================================================
Phases
==========================================================================
TRAIN (ratio supervision):
  Search the window ratio_pos+SEARCH_LO .. ratio_pos+SEARCH_HI. Slide a
  WIN_W-wide raw
  window; score "wave-ness" (sustained elevation + oscillation, not a lone
  spike, not flat). Best position = the wave. Record its END sample-ID.
  Learn: avg raw wave SHAPE (template) + (wave-end, wave-start) -> tokens.

TEST (raw data only, no avg_time/ratio):
  Slide the wave detector over the WHOLE stream, pick the most wave-like
  location by the learned score/shape, read its end sample-ID, estimate.
"""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")

# ── config ─────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data" / "counter"      # raw access-counter streams
META_DIR = BASE / "data" / "metadata"     # per-run token/timing metadata
OUT_DIR  = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = BASE / "data" / "_npy_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "per_class_train").mkdir(exist_ok=True)
(OUT_DIR / "per_class_test").mkdir(exist_ok=True)

CLASSES        = [f"{i:02d}" for i in range(1, 21)]
N_RUNS         = 71            # runs per class expected under data/ (collect >= this)
N_LINES        = 737_200
TOTAL_REC_TIME = 47.3

# raw-window viewing parameters (matching how a human reads the plot)
WIN_W       = 1600         # window width in raw samples
AMP_VIEW    = 120          # amplitude view ceiling (for normalisation/plots)
WAVE_THR    = 28.0         # a sample is "elevated" above this

# oscillation-profile wave extraction (the active detector, find_wave):
OSC_WIN      = 60          # rolling-std window (samples) for local oscillation
OSC_THR      = 6.0         # oscillation above this = part of a wave (plateau std~2-3)
MERGE_GAP    = 800         # join active runs <= this apart (the two bumps of one wave)
MIN_WAVE_LEN = 80          # ignore oscillation regions shorter than this
MIN_RUN_WIDTH = 15         # drop std-blips narrower than this (lone spikes)
GAP_FRAC     = 0.5         # merge across a gap only if gap oscillation >= OSC_THR*this
SUSTAIN_FRAC = 0.4         # a real wave: >= this fraction of its span oscillates
SEARCH_LO   = 30_000       # TRAIN: search window starts at ratio_pos + this
SEARCH_HI   = 100_000      # TRAIN: search window ends   at ratio_pos + this
N_WORKERS   = 8
PER_CLASS_N = 3

# ── legacy / unused knobs ────────────────────────────────────────────────────
# Retained for reference; not used by the active oscillation-profile detector.
BASELINE     = 20.0        # nominal baseline counter value
REL_THR      = 0.5         # relative threshold (older scorer)
MIN_ELEVATED = 20          # backward-scan: min samples above WAVE_THR in a window
MIN_OSC      = 9.0         # backward-scan: min std in the elevated band
MIN_FRAC     = 0.03        # min elevated fraction
STEP         = 200         # sliding-window coarse step
FINE_STEP    = 25          # sliding-window fine step


# ── I/O ─────────────────────────────────────────────────────────────────────
def _si(v, d=0):
    try:    return int(v)
    except: return d

def _sf(v, d=0.0):
    try:    return float(v)
    except: return d

def parse_metadata(cls, run):
    path = META_DIR / cls / f"{cls}_{run}.txt"
    m = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                m[k.strip()] = v.strip()
    return {
        "input_tokens":  _si(m.get("input_tokens")),
        "output_tokens": _si(m.get("output_tokens")),
        "avg_time":      _sf(m.get("avg_time")),
    }

def load_raw(cls, run):
    """Load the FULL raw counter stream as a float32 array (one value/line).

    Cached: first run parses the text file (fast pandas C parser) and saves a
    binary .npy alongside it; every later run loads the .npy in <1 ms instead
    of re-parsing 737k lines (~114 ms). ~280x faster on repeat runs.
    """
    path  = DATA_DIR / cls / f"{cls}_{run}.txt"
    cache = CACHE_DIR / f"{cls}_{run}.npy"
    if cache.exists():
        try:
            return np.load(cache, mmap_mode=None)
        except Exception:
            pass  # corrupt cache -> reparse
    # fast first-parse with pandas' C engine
    arr = pd.read_csv(path, header=None, usecols=[0],
                      dtype=np.float32, sep=r"\s+").values.ravel()
    try:
        np.save(cache, arr)
    except Exception:
        pass
    return arr

def ratio_pos(avg_time):
    return int(np.clip(avg_time / TOTAL_REC_TIME, 0, 1) * (N_LINES - 1))


# ── WAVE SCORING on a raw window ─────────────────────────────────────────────
def wave_score(seg):
    """How wave-like is a raw segment? The wave = sustained elevation with
    oscillation, NOT a single tall spike and NOT flat baseline.

    score = (fraction of samples elevated above WAVE_THR)
            * (oscillation energy of the elevated part)
    A lone spike has tiny elevated-fraction -> low score.
    Flat baseline has ~zero elevated-fraction -> low score.
    The wave has many elevated samples AND high local variance -> high score.

    UNUSED scalar version, kept for reference / single-window use. The hot path
    uses the vectorized find_wave() below.
    """
    elevated = seg > WAVE_THR
    frac = elevated.mean()
    if frac < 1e-3:
        return 0.0, 0
    osc = seg[elevated].std()
    capped = np.clip(seg, 0, 70)
    sustained = uniform_filter1d((capped > WAVE_THR).astype(np.float32), 50)
    score = frac * osc * sustained.max()
    end_idx = np.where(elevated)[0].max()
    return float(score), int(end_idx)


def _sliding_sum(a, w):
    """Sum of every length-w window of a, via cumulative sum.
    Returns array of length len(a)-w+1."""
    c = np.cumsum(np.concatenate(([0.0], a)))
    return c[w:] - c[:-w]


def _rolling_std(a, w):
    """Rolling std over window w, centred, same length as a (edges shrink).
    Uses sliding sums of x and x^2 — fast, no Python loop."""
    n = len(a)
    if n < w:
        return np.zeros(n)
    s  = _sliding_sum(a,      w)
    s2 = _sliding_sum(a * a,  w)
    mean = s / w
    var  = np.clip(s2 / w - mean ** 2, 0, None)
    std  = np.sqrt(var)                      # length n-w+1
    # centre it back to length n
    out = np.zeros(n)
    off = w // 2
    out[off:off + len(std)] = std
    return out


def find_wave(raw, lo, hi):
    """EXTRACT THE WAVE via a local-oscillation profile (not a fixed window).

    The wave = a region of sustained OSCILLATION (two fluctuating bumps close
    together). A flat plateau is elevated but NOT oscillating, so it has low
    rolling-std and is excluded. Steps:

      1. oscillation profile = rolling std over OSC_WIN samples.
      2. active = profile > OSC_THR  (where real fluctuation lives).
      3. merge active runs separated by gaps <= MERGE_GAP  (joins the two
         bumps of one wave into a single region).
      4. keep regions whose extent >= MIN_WAVE_LEN (drop tiny blips).
      5. take the LAST such region (rightmost = end of kernel activity).
      6. start_pos = onset of that region (start of the first bump),
         end_pos   = end of the region.

    Returns (osc_strength, start_pos, end_pos, region_start) in absolute IDs.
    """
    lo = max(0, lo); hi = min(len(raw), hi)
    seg = raw[lo:hi].astype(np.float64)
    n = len(seg)
    if n < OSC_WIN + 2:
        return 0.0, lo, min(lo + WIN_W, hi), lo

    prof = _rolling_std(seg, OSC_WIN)        # local oscillation
    active = prof > OSC_THR

    if not active.any():
        c = int(np.argmax(prof))
        return float(prof.max()), lo + c, lo + c, lo + c

    edges = np.diff(active.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0])
    if active[0]:  starts = [0] + starts
    if active[-1]: ends   = ends + [n - 1]
    runs = list(zip(starts, ends))

    # ── DEFENSE 1: drop NARROW runs — a lone spike makes a thin std-blip
    # (a few samples wide); a real oscillating bump is broader. This removes
    # the edge-spikes of a flat plateau before they can seed a region. ───────
    runs = [(s, e) for s, e in runs if (e - s) >= MIN_RUN_WIDTH]
    if not runs:
        c = int(np.argmax(prof))
        return float(prof.max()), lo + c, lo + c, lo + c

    # ── DEFENSE 2: merge two runs only if the GAP between them is itself
    # somewhat oscillating (mean profile in the gap above OSC_THR*GAP_FRAC).
    # A flat plateau body between two spikes is NOT oscillating, so the spikes
    # will NOT be merged across it. The two genuine bumps of one wave DO have
    # elevated oscillation between them, so they merge. ──────────────────────
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        gap_lo, gap_hi = merged[-1][1], s
        gap = prof[gap_lo:gap_hi]
        gap_osc = gap.mean() if gap.size else 0.0
        if (s - merged[-1][1] <= MERGE_GAP) and (gap_osc >= OSC_THR * GAP_FRAC):
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # ── DEFENSE 3: a real wave is SUSTAINED — most of its span oscillates.
    # Require the region's active fraction (points above OSC_THR) to exceed
    # SUSTAIN_FRAC. A spike-bounded flat plateau has a low active fraction
    # (oscillation only at the edges) and is rejected. ───────────────────────
    def active_frac(s, e):
        if e <= s:
            return 0.0
        return float((prof[s:e + 1] > OSC_THR).mean())

    waves = [(s, e) for s, e in merged
             if (e - s) >= MIN_WAVE_LEN and active_frac(s, e) >= SUSTAIN_FRAC]
    if not waves:
        # nothing qualifies as a sustained wave: take the most-oscillating run
        waves = [max(merged, key=lambda r: prof[r[0]:r[1] + 1].mean()
                     if r[1] > r[0] else 0.0)]

    # LAST region = end of kernel activity
    ws, we = waves[-1]
    osc_strength = float(prof[ws:we + 1].max()) if we >= ws else float(prof.max())

    start_pos = lo + int(ws)
    end_pos   = lo + int(we)
    return osc_strength, start_pos, end_pos, start_pos


# ── parallel raw loader (returns downsample for plotting + key slices) ────────
def _load_one(args):
    cls, run = args
    raw  = load_raw(cls, run)
    meta = parse_metadata(cls, run)
    rec = {
        "class":        int(cls),
        "run":          run,
        "avg_time":     meta["avg_time"],
        "total_tokens": meta["input_tokens"] + meta["output_tokens"],
    }
    return raw, rec


def _save_window(path, raw, wstart, start_pos, end_pos, title):
    """Save the EXACT detection window as the detector saw it:
    WIN_W samples wide, amplitude axis fixed 0..AMP_VIEW (1600 x 120 view).
    This is the literal pattern patch that was matched."""
    w0 = max(0, wstart)
    w1 = min(len(raw), wstart + WIN_W)
    seg = raw[w0:w1]
    x = np.arange(w0, w1)
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(x, seg, lw=0.7, color="blue")
    ax.set_xlim(w0, w0 + WIN_W)
    ax.set_ylim(0, AMP_VIEW)
    # mark the precise elevation endpoints within the window
    ax.axvline(start_pos, color="red", ls="--", lw=1.0)
    ax.axvline(end_pos,   color="red", ls="--", lw=1.0)
    ax.axvspan(start_pos, end_pos, color="red", alpha=0.10)
    ax.set_xlabel("sample-point ID"); ax.set_ylabel("counter value")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ── plotting (raw, no filtering — as the human views it) ─────────────────────
def _save_panel(path, raw, start_pos, end_pos, anchor_pos, title, show_anchor,
                pad=8000):
    lo = max(0, start_pos - pad); hi = min(len(raw), end_pos + pad)
    x = np.arange(lo, hi)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(x, raw[lo:hi], lw=0.5, color="blue")
    ax.set_ylim(0, AMP_VIEW)
    if show_anchor and anchor_pos is not None and lo <= anchor_pos <= hi:
        ax.axvline(anchor_pos, color="darkorange", lw=1.8, label="ratio anchor")
    elif show_anchor and anchor_pos is not None:
        ax.axvline(np.clip(anchor_pos, lo, hi), color="darkorange", lw=1.8,
                   ls=":", label="ratio anchor (off-view)")
    ax.axvspan(start_pos, end_pos, color="red", alpha=0.12, label="detected wave")
    ax.axvline(end_pos, color="red", ls="--", lw=1.3, label="wave end")
    ax.set_xlabel("sample-point ID"); ax.set_ylabel("counter value")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print(f"Loading {len(CLASSES)} classes × {N_RUNS} runs RAW ({N_WORKERS} workers)…")
    print(f"Window {WIN_W} samples × {AMP_VIEW} amp   train search ratio+{SEARCH_LO}..+{SEARCH_HI}")
    print("=" * 65)

    tasks = [(cls, run) for cls in CLASSES for run in range(1, N_RUNS + 1)]

    # ── decide split + token labels up front (metadata only — cheap) ─────────
    meta_records = []
    for cls, run in tasks:
        meta = parse_metadata(cls, run)
        meta_records.append({
            "class": int(cls), "run": run, "avg_time": meta["avg_time"],
            "total_tokens": meta["input_tokens"] + meta["output_tokens"],
        })
    df  = pd.DataFrame(meta_records)
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=0.20,
                                      random_state=42, stratify=df["class"])
    tr_set = set(idx_tr.tolist())
    te_set = set(idx_te.tolist())

    # which rows need their raw array kept for figures (~120 total)
    keep_for_fig = set()
    for cls in range(1, 21):
        cls_rows = df.index[df["class"] == cls].tolist()
        keep_for_fig.update([j for j in cls_rows if j in tr_set][:PER_CLASS_N])
        keep_for_fig.update([j for j in cls_rows if j in te_set][:PER_CLASS_N])

    # ── single streaming pass: load → detect → record → discard ──────────────
    print("Streaming pass: load each file, detect wave, keep only figure rows…")
    tr_feats, tr_y, train_detail, anchor_gap = [], [], {}, []
    tr_starts, tr_ends, tr_spans = [], [], []
    test_rows, test_detail = [], {}
    fig_raw = {}                                    # row -> raw array (figures only)

    def _job(j):
        cls = f"{int(df.iloc[j]['class']):02d}"
        run = int(df.iloc[j]["run"])
        raw = load_raw(cls, run)
        if j in tr_set:
            anc = ratio_pos(df.iloc[j]["avg_time"])
            _, s, e, ws = find_wave(raw, anc + SEARCH_LO, anc + SEARCH_HI)
            payload = ("train", s, e, anc, ws)
        else:
            _, s, e, ws = find_wave(raw, 0, len(raw))
            payload = ("test", s, e, None, ws)
        keep = raw if j in keep_for_fig else None
        return j, payload, keep

    done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for j, payload, keep in ex.map(_job, idx):
            kind, s, e, anc, ws = payload
            if kind == "train":
                anchor_gap.append(abs((s + e) // 2 - anc))
                tr_feats.append([s])
                tr_starts.append(s); tr_ends.append(e); tr_spans.append(e - s)
                tr_y.append(df.iloc[j]["total_tokens"])
                train_detail[j] = (s, e, anc, ws)
            else:
                test_detail[j] = (s, e, ws)
            if keep is not None:
                fig_raw[j] = keep
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(idx)} …")
    print(f"  {len(idx)}/{len(idx)} done.   "
          f"(kept {len(fig_raw)} raw arrays in memory, ~{len(fig_raw)*N_LINES*4/1e6:.0f} MB)")

    # ── TRAIN estimator ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TRAIN: wave in ratio+30k..+100k window → fit START → tokens")
    print("=" * 65)
    Xtr = np.array(tr_feats, float); ytr = np.array(tr_y, float)
    est = LinearRegression().fit(Xtr, ytr)
    print(f"  Mean |wave-center − anchor| : {np.mean(anchor_gap):.0f} samples "
          f"({np.mean(anchor_gap)/N_LINES*100:.1f}% of stream)")
    print(f"  Estimator R² (train, START) : {est.score(Xtr, ytr):.4f}")
    rs, _   = pearsonr(np.array(tr_starts), ytr)
    re_, _  = pearsonr(np.array(tr_ends),   ytr)
    rsp, _  = pearsonr(np.array(tr_spans),  ytr)
    print(f"  Train wave-START ↔ tokens   : r = {rs:.4f}   (feature used)")
    print(f"  Train wave-end   ↔ tokens   : r = {re_:.4f}")
    print(f"  Train wave-span  ↔ tokens   : r = {rsp:.4f}")

    # ── TEST results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TEST: wave in whole stream (no ratio) → estimate tokens")
    print("=" * 65)
    rows = []
    for j in idx_te:
        s, e, ws = test_detail[j]
        est_tok = float(est.predict([[s]])[0])
        rows.append({"row": j, "class": int(df.iloc[j]["class"]),
                     "start_pos": s, "end_pos": e,
                     "total_tokens": int(df.iloc[j]["total_tokens"]),
                     "est_tok": est_tok})
    res = pd.DataFrame(rows)
    r2  = r2_score(res["total_tokens"], res["est_tok"])
    mae = mean_absolute_error(res["total_tokens"], res["est_tok"])
    rts, _  = pearsonr(res["start_pos"], res["total_tokens"])
    rte, _  = pearsonr(res["end_pos"],   res["total_tokens"])
    rtsp, _ = pearsonr(res["end_pos"] - res["start_pos"], res["total_tokens"])
    print(f"  Held-out : R² = {r2:.4f}  MAE = {mae:.1f} tokens")
    print(f"  Test wave-START ↔ tokens : r = {rts:.4f}   (feature used)")
    print(f"  Test wave-end   ↔ tokens : r = {rte:.4f}")
    print(f"  Test wave-span  ↔ tokens : r = {rtsp:.4f}")

    print("\n  Sample held-out streams:")
    print(f"  {'cls':>3} {'start_pt':>9} {'end_pt':>9} {'true':>5} {'est':>7} {'err':>6}")
    for _, r in res.head(15).iterrows():
        print(f"  {int(r['class']):>3} {int(r['start_pos']):>9} {int(r['end_pos']):>9} "
              f"{int(r['total_tokens']):>5} {r['est_tok']:>7.1f} "
              f"{r['est_tok']-r['total_tokens']:>+6.1f}")

    # ── per-class figures ────────────────────────────────────────────────────
    print("\nSaving per-class figures…")
    for cls in range(1, 21):
        cls_rows = df.index[df["class"] == cls].tolist()
        tr_rows = [j for j in cls_rows if j in train_detail][:PER_CLASS_N]
        te_rows = [j for j in cls_rows if j in test_detail][:PER_CLASS_N]
        for k, j in enumerate(tr_rows, 1):
            if j not in fig_raw:
                continue
            s, e, anc, ws = train_detail[j]
            tok = int(df.iloc[j]["total_tokens"])
            _save_panel(OUT_DIR / "per_class_train" / f"cls{cls:02d}_train_{k}.png",
                        fig_raw[j], s, e, anc,
                        f"TRAIN class {cls:02d} run {int(df.iloc[j]['run'])} "
                        f"true {tok} tok  —  wave found near ratio anchor (orange)",
                        show_anchor=True)
            _save_window(OUT_DIR / "per_class_train" / f"cls{cls:02d}_train_{k}_window.png",
                         fig_raw[j], ws, s, e,
                         f"TRAIN class {cls:02d} run {int(df.iloc[j]['run'])} "
                         f"true {tok} tok  —  detection window ({WIN_W}×{AMP_VIEW})")
        for k, j in enumerate(te_rows, 1):
            if j not in fig_raw:
                continue
            s, e, ws = test_detail[j]
            tok = int(df.iloc[j]["total_tokens"])
            est_tok = float(est.predict([[s]])[0])
            _save_panel(OUT_DIR / "per_class_test" / f"cls{cls:02d}_test_{k}.png",
                        fig_raw[j], s, e, None,
                        f"TEST class {cls:02d} run {int(df.iloc[j]['run'])} "
                        f"true {tok} / est {est_tok:.0f} tok  —  found from data only",
                        show_anchor=False)
            _save_window(OUT_DIR / "per_class_test" / f"cls{cls:02d}_test_{k}_window.png",
                         fig_raw[j], ws, s, e,
                         f"TEST class {cls:02d} run {int(df.iloc[j]['run'])} "
                         f"true {tok} / est {est_tok:.0f} tok  —  detection window ({WIN_W}×{AMP_VIEW})")

    _plot_scatter(res)
    print(f"\nDone. results/per_class_train/, results/per_class_test/, token_estimate.png")
    print("=" * 65)


def _plot_scatter(res):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(res["total_tokens"], res["est_tok"], s=40, alpha=0.6,
               color="royalblue", zorder=3)
    vals = np.r_[res["total_tokens"].values, res["est_tok"].values]
    lim = [vals.min() - 5, vals.max() + 5]
    ax.plot(lim, lim, "k--", lw=1.2, label="perfect")
    r2 = r2_score(res["total_tokens"], res["est_tok"])
    mae = mean_absolute_error(res["total_tokens"], res["est_tok"])
    ax.set_xlabel("true total_tokens"); ax.set_ylabel("estimated total_tokens")
    ax.set_title(f"Token estimate (raw wave detection)\nR² = {r2:.4f}  MAE = {mae:.1f}")
    ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout(); fig.savefig(OUT_DIR / "token_estimate.pdf", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
