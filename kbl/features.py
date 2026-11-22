from functools import lru_cache

import numpy as np
from scipy.fft import dct, rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt

from . import config as C

FEATURE_VERSION = 1  # bump this if extract() changes so old models get rejected
N_FFT = 512
FRAME_HOP = 256
N_MELS = 40
N_MFCC = 20
N_SEGMENTS = 8
TOUCH_SAMPLES = 1024  # first ~23ms, when the finger actually hits the key
N_TOUCH_BANDS = 48
N_ENV = 24


@lru_cache(maxsize=None)
def highpass_sos(sr):
    return butter(4, C.HIGHPASS_HZ, btype="highpass", fs=sr, output="sos")


def envelope(x, hop=C.HOP):
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    f = np.asarray(x[:n * hop], dtype=np.float64).reshape(n, hop)
    return np.sqrt((f * f).mean(axis=1) + 1e-14)


def find_onset(region, sr, lo, hi, max_back=0.03):
    # find the loudest point in region[lo:hi], then walk back to where the click starts
    # returns (offset, peak, floor) or None
    y = sosfiltfilt(highpass_sos(sr), np.asarray(region, dtype=np.float64))
    env = envelope(y)
    a, b = max(0, lo // C.HOP), min(len(env), hi // C.HOP)
    if b <= a:
        return None
    p = a + int(np.argmax(env[a:b]))
    peak = float(env[p])
    floor = float(np.percentile(env, 15))
    thr = floor + 0.2 * (peak - floor)
    i, limit = p, max(0, p - int(max_back * sr / C.HOP))
    while i > limit and env[i - 1] > thr:
        i -= 1
    return i * C.HOP, peak, floor


@lru_cache(maxsize=None)
def mel_filterbank(sr, n_fft=N_FFT, n_mels=N_MELS, fmin=100.0):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    hz = mel2hz(np.linspace(hz2mel(fmin), hz2mel(sr / 2), n_mels + 2))
    bins = rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for m in range(n_mels):
        l, c, r = hz[m], hz[m + 1], hz[m + 2]
        fb[m] = np.maximum(0, np.minimum((bins - l) / (c - l), (r - bins) / (r - c)))
    return fb


@lru_cache(maxsize=None)
def touch_bands(sr):
    freqs = rfftfreq(TOUCH_SAMPLES, 1.0 / sr)
    edges = np.geomspace(200, min(20000, sr / 2), N_TOUCH_BANDS + 1)
    return [np.where((freqs >= edges[k]) & (freqs < edges[k + 1]))[0] for k in range(N_TOUCH_BANDS)]


def extract(clip, sr=C.SR, shift=0, gain_db=0.0, noise=0.0, rng=None):
    # clip -> 294 features. shift/gain/noise are only for augmentation in training
    L = int(round(C.CLIP_LEN * sr))
    start = int(round((C.SAVE_PRE - C.CLIP_PRE) * sr)) + int(shift)
    x = np.asarray(clip[max(0, start):start + L], dtype=np.float64)
    if len(x) < L:
        x = np.pad(x, (0, L - len(x)))
    x = x * 10.0 ** (gain_db / 20.0)
    if noise > 0:
        rng = rng or np.random.default_rng()
        x = x + rng.normal(0, noise * np.sqrt(np.mean(x * x) + 1e-14), len(x))

    y = sosfiltfilt(highpass_sos(sr), x)
    rms = np.sqrt(np.mean(y * y)) + 1e-9
    loudness = [20 * np.log10(rms), 20 * np.log10(np.max(np.abs(y)) + 1e-9)]
    y = y / rms  # normalize so the rest doesn't care how hard the key was hit

    # mfccs
    n_frames = 1 + (L - N_FFT) // FRAME_HOP
    idx = np.arange(N_FFT)[None, :] + FRAME_HOP * np.arange(n_frames)[:, None]
    spec = np.abs(rfft(y[idx] * np.hanning(N_FFT), axis=1)) ** 2
    logmel = np.log(mel_filterbank(sr) @ spec.T + 1e-8)
    mfcc = dct(logmel, type=2, axis=0, norm="ortho")[:N_MFCC]
    # average into 8 chunks so we keep the order (touch -> bottom out -> decay)
    chunks = np.array_split(np.arange(n_frames), N_SEGMENTS)
    mfcc_chunks = np.concatenate([mfcc[:, c].mean(axis=1) for c in chunks])

    # spectrum of just the first hit
    t = y[:TOUCH_SAMPLES] * np.hanning(TOUCH_SAMPLES)
    p = np.abs(rfft(t)) ** 2
    touch = np.log(np.array([p[b].mean() if len(b) else 0.0 for b in touch_bands(sr)]) + 1e-8)
    touch -= touch.mean()

    # loudness shape over time
    env = np.log(np.array([np.sqrt(np.mean(s * s) + 1e-12) for s in np.array_split(y, N_ENV)]))

    return np.concatenate([
        mfcc_chunks, logmel.mean(axis=1), mfcc.std(axis=1), touch, env, loudness
    ]).astype(np.float32)


def clip_peak(clip, sr=C.SR):
    y = sosfiltfilt(highpass_sos(sr), np.asarray(clip, dtype=np.float64))
    env = envelope(y)
    return float(env.max()) if len(env) else 0.0
