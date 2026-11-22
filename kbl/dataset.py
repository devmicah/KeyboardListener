# clips live in data/samples/<key>/<timestamp>.wav
import os
from collections import Counter
from datetime import datetime

import numpy as np
from scipy.io import wavfile

from . import config as C


def save_clip(data_dir, label, clip, sr):
    d = os.path.join(data_dir, label)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".wav")
    wavfile.write(path, sr, np.asarray(clip, dtype=np.float32))
    return path


def counts(data_dir):
    c = Counter()
    if not os.path.isdir(data_dir):
        return c
    for label in os.listdir(data_dir):
        d = os.path.join(data_dir, label)
        if os.path.isdir(d):
            c[label] = sum(1 for f in os.listdir(d) if f.endswith(".wav"))
    return c


def load(data_dir):
    clips, labels, sr = [], [], None
    if not os.path.isdir(data_dir):
        return clips, labels, C.SR
    for label in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, label)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".wav"):
                continue
            file_sr, x = wavfile.read(os.path.join(d, f))
            if sr is None:
                sr = file_sr
            elif file_sr != sr:
                print(f"  skipping {label}/{f}, sample rate {file_sr} != {sr}")
                continue
            if x.dtype.kind in "iu":
                x = x.astype(np.float32) / np.iinfo(x.dtype).max
            clips.append(x.astype(np.float32))
            labels.append(label)
    return clips, labels, sr or C.SR


def print_counts(data_dir, wanted=None):
    c = counts(data_dir)
    keys = sorted(set(c) | set(wanted or []))
    if not keys:
        print("  no samples yet")
        return
    width = max(len(k) for k in keys)
    for k in keys:
        n = c.get(k, 0)
        print(f"  {k:<{width}}  {n:4d}  {'#' * min(n, 60)}")
    print(f"  total: {sum(c.values())} clips, {len(c)} labels")
