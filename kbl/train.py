import os
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from . import dataset
from .features import FEATURE_VERSION, clip_peak, extract


def models():
    return {
        # SVC(probability=True) is deprecated in sklearn 1.9 so calibrate it instead
        "svm": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(C=10, gamma="scale"), cv=3, ensemble=False)),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "forest": RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1),
    }


def featurize(clips, sr, augment):
    # returns [original features, augmented copy 1, copy 2, ...]
    rng = np.random.default_rng(0)
    ms = sr // 1000
    variants = [{}]
    if augment:
        variants += [
            dict(shift=-2 * ms, gain_db=-2.0),
            dict(shift=2 * ms, gain_db=2.0),
            dict(shift=ms, noise=0.05),
            dict(shift=-ms, gain_db=1.0, noise=0.03),
        ]
    return [np.stack([extract(c, sr, rng=rng, **v) for c in clips]) for v in variants]


def run(data_dir=C.DATA_DIR, model_path=C.MODEL_PATH, min_samples=5, augment=True, only=None):
    t0 = time.time()
    clips, labels, sr = dataset.load(data_dir)
    if not clips:
        print(f"no samples in {data_dir}, run 'python main.py record' first")
        return

    cnt = Counter(labels)
    keep = {k for k, n in cnt.items() if n >= min_samples}
    dropped = sorted(set(cnt) - keep)
    if dropped:
        print(f"ignoring keys with < {min_samples} samples: {', '.join(dropped)}")
    if only is not None:
        keep &= set(only)
    sel = [i for i, l in enumerate(labels) if l in keep]
    clips = [clips[i] for i in sel]
    y = np.array([labels[i] for i in sel])
    if len(set(y)) < 2:
        print("need at least 2 keys with enough samples")
        return

    print(f"{len(y)} clips, {len(set(y))} labels. extracting features...")
    feats = featurize(clips, sr, augment)
    X0 = feats[0]
    print(f"  {X0.shape[1]} features per clip ({time.time() - t0:.1f}s)")

    # augmented copies only go in the train side of each fold, never the test side
    n_splits = int(min(5, Counter(y).most_common()[-1][1]))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    is_key = y != C.RELEASE_LABEL
    print(f"\n{n_splits}-fold cross validation (accuracy on key presses):")
    results = {}
    for name, model in models().items():
        pred = np.empty(len(y), dtype=object)
        for tr, te in skf.split(X0, y):
            model.fit(np.concatenate([f[tr] for f in feats]), np.concatenate([y[tr]] * len(feats)))
            pred[te] = model.predict(X0[te])
        acc = float(np.mean(pred[is_key] == y[is_key])) if is_key.any() else 0.0
        results[name] = (acc, pred)
        print(f"  {name:<7} {acc:6.1%}")

    best = max(results, key=lambda k: results[k][0])
    acc, pred = results[best]
    print(f"\nbest: {best} ({acc:.1%}), random guessing would be {1 / max(1, len(set(y[is_key]))):.1%}")
    report(y, pred)

    # retrain the winner on everything
    model = models()[best]
    model.fit(np.concatenate(feats), np.concatenate([y] * len(feats)))
    peaks = np.array([clip_peak(c, sr) for c, k in zip(clips, is_key) if k])
    bundle = {
        "model": model, "model_name": best, "sr": sr, "feature_version": FEATURE_VERSION,
        "classes": list(model.classes_), "cv_accuracy": acc, "n_samples": len(y),
        "peak_p10": float(np.percentile(peaks, 10)) if len(peaks) else None,
        "peak_median": float(np.median(peaks)) if len(peaks) else None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"\nsaved {model_path} ({time.time() - t0:.1f}s)")


def report(y, pred):
    labels = sorted(set(y))
    width = max(len(l) for l in labels)
    print("\nper key:")
    for l in labels:
        m = y == l
        wrong = Counter(p for p in pred[m] if p != l).most_common(2)
        extra = ("   mixed up with " + ", ".join(f"{k} x{n}" for k, n in wrong)) if wrong else ""
        print(f"  {l:<{width}}  {np.mean(pred[m] == l):6.1%}  (n={m.sum()}){extra}")
    pairs = Counter((a, b) for a, b in zip(y, pred) if a != b).most_common(5)
    if pairs:
        print("\nmost confused (actual -> guessed):")
        for (a, b), n in pairs:
            print(f"  {a} -> {b}: {n}")


def load_model(model_path=C.MODEL_PATH):
    if not os.path.exists(model_path):
        raise SystemExit(f"no model at {model_path}, run 'python main.py train' first")
    bundle = joblib.load(model_path)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise SystemExit("model is from an older feature version, retrain it")
    return bundle
