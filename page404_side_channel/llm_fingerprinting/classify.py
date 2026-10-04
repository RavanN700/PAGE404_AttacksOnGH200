#!/usr/bin/env python3
"""
LLM-architecture fingerprinting — time-series classification.

Classifies which model produced a migration-delay trace. Operates on the
variable-length counter streams collected by auto_collection.py: one class
subfolder per model, one trace file per run.

Models: RandomForest, HistGradientBoosting, XGBoost (baseline + TUNED),
        LightGBM, soft-voting ensemble, optional 1-D CNN.
Plus: optional class exclude/merge, feature selection, randomized XGB search,
      early stopping, a feature-extraction parameter search, and full
      5-fold cross-validation for every model.

Folder layout:
    ROOT/<model_name>/*.txt   (one number per line per file)
e.g. ROOT/flan_t5_large/flan_t5_large_1.txt — exactly what the collector writes
under data/counter/.

Run:
    pip install numpy scipy scikit-learn matplotlib xgboost lightgbm
    pip install tensorflow          # optional, for the CNN
    python classify.py
"""

import os
import glob
import itertools
import warnings
import numpy as np
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle as sk_shuffle
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, RandomizedSearchCV)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# silence the harmless LightGBM "X does not have valid feature names" spam
warnings.filterwarnings("ignore", message=".*valid feature names.*")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT = "./data/counter/"  # collector output: one subfolder per model (<-- your path)
TARGET_LEN = 256
N_FFT      = 64
N_SEG      = 8
FILE_GLOB  = "*.txt"
RANDOM_STATE = 42
RUN_PARAM_SEARCH = True       # Phase 0: search TARGET_LEN/N_FFT/N_SEG first

XGB_USE_GPU  = True       # GH200: train XGBoost on the GPU (xgboost >= 2.0)
LGBM_USE_GPU = False      # LightGBM GPU needs a special build; CPU is fine here

# XGBoost tuning controls
RUN_XGB_SEARCH    = True
XGB_SEARCH_ITERS  = 20    # was 40; fewer candidates = much faster search
XGB_SEARCH_CV     = 3     # was 4; 3-fold is plenty for ranking params
XGB_SEARCH_ON_GPU = False # search on CPU: faster here (small data, no transfer)
USE_FEATURE_SELECT = True

# cross-validation controls
CV_INCLUDE_ENSEMBLE = True    # CV the soft-voting ensemble (refits bases x5)
CV_INCLUDE_CNN      = True    # CV the 1-D CNN (5 full retrains; slow)

# confusion-matrix output
CM_OUTFILE = "confusion_matrix.pdf"   # vector PDF with grid lines
CM_SAVE_DATA = True                   # also dump CSV + npz so you can replot

# --- class handling ---
# Class subfolders to skip entirely, e.g. ["gemma_1b_it"]. Empty = use all.
EXCLUDE_CLASSES = []
# Optional label merging: {"merged_name": ["folderA", "folderB"]}.
MERGE_GROUPS = {
}


def pretty_label(folder_name):
    """Display label for a class folder. The folder name (model name) is
    already human-readable, so this is just a safe string cast."""
    return str(folder_name)


# ---------------------------------------------------------------------------
# 1. Feature extraction
# ---------------------------------------------------------------------------
def resample_series(series, target_len):
    series = np.asarray(series, dtype=np.float64)
    if len(series) == target_len:
        return series
    if len(series) < 2:
        return np.full(target_len, series[0] if len(series) else 0.0)
    x_old = np.linspace(0.0, 1.0, len(series))
    x_new = np.linspace(0.0, 1.0, target_len)
    return interp1d(x_old, series, kind="linear")(x_new)


def summary_stats(series):
    s = np.asarray(series, dtype=np.float64)
    if len(s) < 2:
        s = np.append(s, s[-1] if len(s) else 0.0)
    diffs = np.diff(s)
    t = np.arange(len(s))
    slope = np.polyfit(t, s, 1)[0] if len(s) > 1 else 0.0
    cv = s.std() / (abs(s.mean()) + 1e-9)
    return np.array([
        len(series),
        s.mean(), s.std(),
        s.min(), s.max(), np.ptp(s),
        cv,
        np.median(s),
        np.percentile(s, 25), np.percentile(s, 75),
        np.mean(np.abs(diffs)),
        np.std(diffs),
        slope,
        float(np.sum(s)),
    ], dtype=np.float64)


def fft_features(series, n=N_FFT):
    s = np.asarray(series, dtype=np.float64)
    s = s - s.mean()
    mag = np.abs(np.fft.rfft(s))
    if len(mag) >= n:
        mag = mag[:n]
    else:
        mag = np.pad(mag, (0, n - len(mag)))
    return np.log1p(mag)


def segment_stats(series, n_seg=N_SEG):
    s = np.asarray(series, dtype=np.float64)
    if len(s) < n_seg:
        s = np.pad(s, (0, n_seg - len(s)))
    feats = []
    for seg in np.array_split(s, n_seg):
        feats += [seg.mean(), seg.std(), seg.max(), seg.min()]
    return np.array(feats, dtype=np.float64)


def features_from_series(series):
    return np.concatenate([
        resample_series(series, TARGET_LEN),
        summary_stats(series),
        fft_features(series, N_FFT),
        segment_stats(series, N_SEG),
    ])


def features_from_series_p(series, target_len, n_fft, n_seg):
    return np.concatenate([
        resample_series(series, target_len),
        summary_stats(series),
        fft_features(series, n_fft),
        segment_stats(series, n_seg),
    ])


# ---------------------------------------------------------------------------
# 2. Load
# ---------------------------------------------------------------------------
def load_dataset(root):
    class_dirs = sorted(d for d in os.listdir(root)
                        if os.path.isdir(os.path.join(root, d)))
    class_dirs = [c for c in class_dirs if c not in EXCLUDE_CLASSES]
    if not class_dirs:
        raise RuntimeError(f"No (non-excluded) class subfolders found in {root}")

    folder_to_label = {}
    for c in class_dirs:
        merged = next((g for g, members in MERGE_GROUPS.items()
                       if c in members), None)
        folder_to_label[c] = merged if merged else c

    label_keys = sorted(set(folder_to_label.values()))
    label_map = {name: i for i, name in enumerate(label_keys)}
    label_names = [pretty_label(k) for k in label_keys]

    X, y, raw_lengths = [], [], []
    for c in class_dirs:
        for f in glob.glob(os.path.join(root, c, FILE_GLOB)):
            series = np.atleast_1d(np.loadtxt(f))
            X.append(features_from_series(series))
            y.append(label_map[folder_to_label[c]])
            raw_lengths.append(len(series))

    X = np.nan_to_num(np.array(X, dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y, dtype=np.int64)
    n_extra = X.shape[1] - TARGET_LEN
    print(f"Loaded {len(X)} samples across {len(label_names)} classes")
    if EXCLUDE_CLASSES: print(f"Excluded: {EXCLUDE_CLASSES}")
    if MERGE_GROUPS:    print(f"Merged:   {MERGE_GROUPS}")
    print(f"Classes:  {label_names}")
    print(f"Feature shape: {X.shape}  (={TARGET_LEN} shape + {n_extra} engineered)")
    print(f"Series length: min={min(raw_lengths)} max={max(raw_lengths)} "
          f"mean={np.mean(raw_lengths):.0f}")
    return X, y, label_names


# ---------------------------------------------------------------------------
# 3. Reporting helpers
# ---------------------------------------------------------------------------
def report(name, y_test, pred, class_names):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Held-out accuracy: {accuracy_score(y_test, pred):.4f}\n")
    print(classification_report(y_test, pred, target_names=class_names,
                                zero_division=0))


def evaluate(name, clf, X_train, X_test, y_train, y_test, class_names):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    report(name, y_test, pred, class_names)
    return clf, pred


def cross_validate(name, clf, X, y, cv=None, n_jobs=-1):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=RANDOM_STATE)
    sc = cross_val_score(clf, X, y, cv=cv, n_jobs=n_jobs)
    print(f"{name}: 5-fold CV = {sc.mean():.4f} +/- {sc.std():.4f}")
    return sc.mean(), sc.std()


def save_confusion(y_test, pred, class_names, title, outfile,
                   save_data=CM_SAVE_DATA):
    cm = confusion_matrix(y_test, pred, labels=range(len(class_names)))
    if save_data:
        stem = os.path.splitext(outfile)[0]
        header = "," + ",".join(class_names)
        with open(stem + ".csv", "w") as fh:
            fh.write("true\\pred" + header + "\n")
            for name, row in zip(class_names, cm):
                fh.write(name + "," + ",".join(str(int(v)) for v in row) + "\n")
        np.savez(stem + ".npz", cm=cm,
                 class_names=np.array(class_names, dtype=object),
                 title=title)
        print(f"Saved confusion data  -> {stem}.csv  and  {stem}.npz")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not installed; no PDF.")
        return
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.5), max(8, n * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    cmax = cm.max() if cm.size else 0
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        fontsize=6,
                        color="white" if cm[i, j] > cmax / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    print(f"Saved confusion matrix -> {outfile}")


# ---------------------------------------------------------------------------
# 4. XGBoost (baseline, tuned, feature selection)
# ---------------------------------------------------------------------------
def _xgb_common():
    p = dict(objective="multi:softprob", n_jobs=-1, random_state=RANDOM_STATE)
    if XGB_USE_GPU:
        p.update(device="cuda", tree_method="hist")
    else:
        p.update(tree_method="hist")
    return p


def make_xgb_baseline():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[skip] xgboost not installed; pip install xgboost")
        return None
    return XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         **_xgb_common())


def make_xgb_baseline_cpu():
    """CPU clone of the baseline, used ONLY for cross-validation so that
    parallel folds don't contend on a single CUDA device."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         objective="multi:softprob", tree_method="hist",
                         n_jobs=1, random_state=RANDOM_STATE)


def feature_select_for_xgb(X_train, X_test, y_train):
    base = make_xgb_baseline()
    if base is None:
        return X_train, X_test, None
    base.fit(X_train, y_train)
    sel = SelectFromModel(base, threshold="median", prefit=True)
    Xtr_s = sel.transform(X_train)
    Xte_s = sel.transform(X_test)
    print(f"Feature selection: {X_train.shape[1]} -> {Xtr_s.shape[1]} features")
    return Xtr_s, Xte_s, sel


# ---------------------------------------------------------------------------
# 4a. Feature-extraction parameter search
# ---------------------------------------------------------------------------
def load_raw_series(root):
    class_dirs = sorted(d for d in os.listdir(root)
                        if os.path.isdir(os.path.join(root, d)))
    class_dirs = [c for c in class_dirs if c not in EXCLUDE_CLASSES]
    if not class_dirs:
        raise RuntimeError(f"No (non-excluded) class subfolders found in {root}")
    folder_to_label = {}
    for c in class_dirs:
        merged = next((g for g, members in MERGE_GROUPS.items()
                       if c in members), None)
        folder_to_label[c] = merged if merged else c
    label_keys = sorted(set(folder_to_label.values()))
    label_map = {name: i for i, name in enumerate(label_keys)}
    label_names = [pretty_label(k) for k in label_keys]
    series_list, y = [], []
    for c in class_dirs:
        for f in glob.glob(os.path.join(root, c, FILE_GLOB)):
            s = np.atleast_1d(np.loadtxt(f))
            series_list.append(np.asarray(s, dtype=np.float64))
            y.append(label_map[folder_to_label[c]])
    y = np.array(y, dtype=np.int64)
    print(f"[search] cached {len(series_list)} raw series, "
          f"{len(label_names)} classes")
    return series_list, y, label_names


def build_X(series_list, target_len, n_fft, n_seg):
    X = np.array([features_from_series_p(s, target_len, n_fft, n_seg)
                  for s in series_list], dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


TARGET_LEN_GRID = [64, 128, 192, 256, 384]
N_FFT_GRID      = [32, 64, 128]
N_SEG_GRID      = [4, 8, 16]


def make_search_model():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
            reg_lambda=1.0, objective="multiclass", n_jobs=-1,
            random_state=RANDOM_STATE, verbose=-1)
    except ImportError:
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
            l2_regularization=1.0, random_state=RANDOM_STATE)


def search_feature_params(root, cv_splits=5, grids=None, verbose=True):
    if grids is None:
        grids = (TARGET_LEN_GRID, N_FFT_GRID, N_SEG_GRID)
    tl_grid, nf_grid, ns_grid = grids
    series_list, y, _ = load_raw_series(root)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    combos = list(itertools.product(tl_grid, nf_grid, ns_grid))
    print(f"\n##### FEATURE-PARAM SEARCH: {len(combos)} combos "
          f"x {cv_splits}-fold = {len(combos) * cv_splits} fits #####")
    results = []
    for i, (tl, nf, ns) in enumerate(combos, 1):
        X = build_X(series_list, tl, nf, ns)
        model = make_search_model()
        sc = cross_val_score(model, X, y, cv=skf, n_jobs=-1)
        res = dict(target_len=tl, n_fft=nf, n_seg=ns,
                   cv_mean=float(sc.mean()), cv_std=float(sc.std()),
                   n_features=X.shape[1])
        results.append(res)
        if verbose:
            print(f"[{i:>3}/{len(combos)}] "
                  f"TL={tl:<4} N_FFT={nf:<4} N_SEG={ns:<3} "
                  f"-> CV={res['cv_mean']:.4f} +/- {res['cv_std']:.4f} "
                  f"({res['n_features']} feats)")
    results.sort(key=lambda r: (-r["cv_mean"], r["n_features"]))
    print("\n----- TOP 5 FEATURE CONFIGS -----")
    for r in results[:5]:
        print(f"  TL={r['target_len']:<4} N_FFT={r['n_fft']:<4} "
              f"N_SEG={r['n_seg']:<3}  CV={r['cv_mean']:.4f} "
              f"+/- {r['cv_std']:.4f}  ({r['n_features']} feats)")
    best = results[0]
    print(f"\nBest feature config: TARGET_LEN={best['target_len']}, "
          f"N_FFT={best['n_fft']}, N_SEG={best['n_seg']}  "
          f"(CV={best['cv_mean']:.4f} +/- {best['cv_std']:.4f})")
    return best, results


def tune_xgb(X_train, y_train):
    """Returns (fitted_best_model, best_params_dict) or (None, None)."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None, None
    param_dist = {
        "n_estimators":     [400, 600, 800, 1200],
        "max_depth":        [3, 4, 5, 6, 8],
        "learning_rate":    [0.02, 0.03, 0.05, 0.1],
        "subsample":        [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma":            [0, 0.1, 0.5, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
        "reg_alpha":        [0, 0.1, 1.0],
    }
    search_params = dict(objective="multi:softprob", tree_method="hist",
                         n_jobs=1, random_state=RANDOM_STATE)
    if XGB_SEARCH_ON_GPU:
        search_params.update(device="cuda")
    base = XGBClassifier(**search_params)
    search_jobs = 1 if XGB_SEARCH_ON_GPU else -1
    search = RandomizedSearchCV(
        base, param_dist, n_iter=XGB_SEARCH_ITERS, cv=XGB_SEARCH_CV,
        scoring="accuracy", n_jobs=search_jobs, verbose=1,
        random_state=RANDOM_STATE)
    print(f"\nRunning XGBoost randomized search "
          f"({XGB_SEARCH_ITERS} candidates x {XGB_SEARCH_CV}-fold = "
          f"{XGB_SEARCH_ITERS * XGB_SEARCH_CV} fits, "
          f"{'GPU' if XGB_SEARCH_ON_GPU else 'CPU'})...")
    search.fit(X_train, y_train)
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    best = XGBClassifier(**{**_xgb_common(), **search.best_params_})
    best.set_params(n_estimators=3000, early_stopping_rounds=50,
                    eval_metric="mlogloss")
    Xtr, Xval, ytr, yval = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train,
        random_state=RANDOM_STATE)
    best.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    try:
        print(f"Early stopping chose {best.best_iteration} trees")
    except Exception:
        pass
    return best, dict(search.best_params_)


# ---------------------------------------------------------------------------
# 4b. LightGBM
# ---------------------------------------------------------------------------
def make_lgbm():
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("[skip] lightgbm not installed; pip install lightgbm")
        return None
    params = dict(
        n_estimators=800, learning_rate=0.05, num_leaves=63, max_depth=-1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
        objective="multiclass", n_jobs=-1, random_state=RANDOM_STATE,
        verbose=-1)
    if LGBM_USE_GPU:
        params.update(device="gpu")
    return LGBMClassifier(**params)


# ---------------------------------------------------------------------------
# 5. CNN  (+ sklearn wrapper for cross-validation)
# ---------------------------------------------------------------------------
class CNNClassifier(BaseEstimator, ClassifierMixin):
    """Thin sklearn wrapper around the 1-D CNN so it can be cross-validated."""
    def __init__(self, target_len=None, epochs=120, batch_size=32,
                 random_state=RANDOM_STATE, verbose=0):
        self.target_len = target_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def _prep(self, a):
        tl = self.target_len or a.shape[1]
        a = a[:, :tl]
        m = a.mean(axis=1, keepdims=True)
        s = a.std(axis=1, keepdims=True) + 1e-8
        return ((a - m) / s)[..., np.newaxis]

    def fit(self, X, y):
        import tensorflow as tf
        from tensorflow.keras import layers, models
        tf.keras.utils.set_random_seed(self.random_state)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        Xtr = self._prep(np.asarray(X))
        Xtr2, Xval, ytr2, yval = train_test_split(
            Xtr, y, test_size=0.2, stratify=y,
            random_state=self.random_state)
        model = models.Sequential([
            layers.Input(shape=Xtr.shape[1:]),
            layers.Conv1D(32, 7, activation="relu", padding="same"),
            layers.BatchNormalization(), layers.MaxPooling1D(3),
            layers.Conv1D(64, 5, activation="relu", padding="same"),
            layers.BatchNormalization(), layers.MaxPooling1D(3),
            layers.Conv1D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(), layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 factor=0.5, patience=6,
                                                 min_lr=1e-5),
        ]
        model.fit(Xtr2, ytr2, validation_data=(Xval, yval),
                  epochs=self.epochs, batch_size=self.batch_size,
                  callbacks=cbs, verbose=self.verbose)
        self.model_ = model
        return self

    def predict(self, X):
        Xte = self._prep(np.asarray(X))
        return self.model_.predict(Xte, verbose=0).argmax(axis=1)


def make_cnn_cv():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return None
    return CNNClassifier(target_len=TARGET_LEN, epochs=120, verbose=0)


def run_cnn(X_train, X_test, y_train, y_test, class_names):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print("\n[skip] TensorFlow not installed; skipping CNN.")
        return None
    print("\n" + "=" * 60)
    print("1-D CNN (long resampled signal)")
    print("=" * 60)

    def prep(a):
        a = a[:, :TARGET_LEN]
        m = a.mean(axis=1, keepdims=True)
        s = a.std(axis=1, keepdims=True) + 1e-8
        return ((a - m) / s)[..., np.newaxis]

    Xtr, Xte = prep(X_train), prep(X_test)
    n_classes = len(class_names)
    Xtr2, Xval, ytr2, yval = train_test_split(
        Xtr, y_train, test_size=0.2, stratify=y_train,
        random_state=RANDOM_STATE)
    Xtr2, ytr2 = sk_shuffle(Xtr2, ytr2, random_state=RANDOM_STATE)
    model = models.Sequential([
        layers.Input(shape=Xtr.shape[1:]),
        layers.Conv1D(32, 7, activation="relu", padding="same"),
        layers.BatchNormalization(), layers.MaxPooling1D(3),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.BatchNormalization(), layers.MaxPooling1D(3),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(), layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=6, min_lr=1e-5),
    ]
    model.fit(Xtr2, ytr2, validation_data=(Xval, yval),
              epochs=120, batch_size=32, callbacks=cbs, verbose=2)
    loss, acc = model.evaluate(Xte, y_test, verbose=0)
    print(f"\nCNN test accuracy: {acc:.4f}")
    pred = model.predict(Xte, verbose=0).argmax(axis=1)
    print(classification_report(y_test, pred, target_names=class_names,
                                zero_division=0))
    return model


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    if RUN_PARAM_SEARCH:
        best_cfg, _ = search_feature_params(ROOT)
        global TARGET_LEN, N_FFT, N_SEG
        TARGET_LEN = best_cfg["target_len"]
        N_FFT      = best_cfg["n_fft"]
        N_SEG      = best_cfg["n_seg"]
        print(f"\n>>> Using searched params: TARGET_LEN={TARGET_LEN}, "
              f"N_FFT={N_FFT}, N_SEG={N_SEG}\n")

    X, y, class_names = load_dataset(ROOT)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)}   Test: {len(X_test)}")

    rf = RandomForestClassifier(n_estimators=600, random_state=RANDOM_STATE,
                                n_jobs=-1)
    gb = HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=RANDOM_STATE)

    # ================= PHASE 1: BASELINES =================
    print("\n##### PHASE 1: BASELINES #####")
    evaluate("RANDOM FOREST", rf, X_train, X_test, y_train, y_test, class_names)
    _, gb_pred = evaluate("HIST GRADIENT BOOSTING", gb,
                          X_train, X_test, y_train, y_test, class_names)
    best_pred, best_name = gb_pred, "HistGradientBoosting"

    xgb_base = make_xgb_baseline()
    if xgb_base is not None:
        _, xgb_pred = evaluate("XGBOOST (baseline)", xgb_base,
                               X_train, X_test, y_train, y_test, class_names)
        if accuracy_score(y_test, xgb_pred) >= accuracy_score(y_test, best_pred):
            best_pred, best_name = xgb_pred, "XGBoost-baseline"

    lgbm = make_lgbm()
    if lgbm is not None:
        _, lgbm_pred = evaluate("LIGHTGBM", lgbm,
                                X_train, X_test, y_train, y_test, class_names)
        if accuracy_score(y_test, lgbm_pred) >= accuracy_score(y_test, best_pred):
            best_pred, best_name = lgbm_pred, "LightGBM"

    estimators = [("rf", rf), ("gb", gb)]
    if xgb_base is not None:
        estimators.append(("xgb", make_xgb_baseline()))
    if lgbm is not None:
        estimators.append(("lgbm", make_lgbm()))
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    _, ens_pred = evaluate("SOFT-VOTING ENSEMBLE", ensemble,
                           X_train, X_test, y_train, y_test, class_names)
    if accuracy_score(y_test, ens_pred) >= accuracy_score(y_test, best_pred):
        best_pred, best_name = ens_pred, "Ensemble"

    save_confusion(y_test, best_pred, class_names, best_name, CM_OUTFILE)

    # ================= PHASE 2: XGBOOST TUNING =================
    best_xgb_params = None
    if xgb_base is not None and RUN_XGB_SEARCH:
        print("\n##### PHASE 2: XGBOOST TUNING (slow) #####")
        Xtr_xgb, Xte_xgb = X_train, X_test
        if USE_FEATURE_SELECT:
            Xtr_xgb, Xte_xgb, _ = feature_select_for_xgb(
                X_train, X_test, y_train)
        best_xgb, best_xgb_params = tune_xgb(Xtr_xgb, y_train)
        if best_xgb is not None:
            pred = best_xgb.predict(Xte_xgb)
            report("XGBOOST (tuned + selected)", y_test, pred, class_names)
            if accuracy_score(y_test, pred) >= accuracy_score(y_test, best_pred):
                best_pred, best_name = pred, "XGBoost-tuned"
                save_confusion(y_test, best_pred, class_names, best_name,
                               CM_OUTFILE)

    print(f"\nOverall best model: {best_name} "
          f"({accuracy_score(y_test, best_pred):.4f})")

    # ================= CROSS-VALIDATION (every model) =================
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-fold)")
    print("=" * 60)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cross_validate("Random Forest        ", rf, X, y, cv=skf)
    cross_validate("HistGradientBoosting ", gb, X, y, cv=skf)

    xgb_cv = make_xgb_baseline_cpu()
    if xgb_cv is not None:
        cross_validate("XGBoost (baseline)   ", xgb_cv, X, y, cv=skf, n_jobs=1)

    if lgbm is not None:
        cross_validate("LightGBM             ", make_lgbm(), X, y, cv=skf)

    if best_xgb_params is not None:
        xgb_tuned_cv = make_xgb_baseline_cpu()
        xgb_tuned_cv.set_params(**best_xgb_params)
        cross_validate("XGBoost (tuned*)     ", xgb_tuned_cv, X, y,
                       cv=skf, n_jobs=1)

    if CV_INCLUDE_ENSEMBLE:
        cross_validate("Soft-voting Ensemble ", ensemble, X, y, cv=skf)

    if CV_INCLUDE_CNN:
        cnn_cv = make_cnn_cv()
        if cnn_cv is not None:
            cross_validate("1-D CNN              ", cnn_cv, X, y,
                           cv=skf, n_jobs=1)

    # ================= PHASE 3: CNN (held-out) =================
    run_cnn(X_train, X_test, y_train, y_test, class_names)


if __name__ == "__main__":
    main()