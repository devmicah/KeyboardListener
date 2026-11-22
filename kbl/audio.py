import threading
import time

import numpy as np
import sounddevice as sd

from . import config as C


class AudioRing:
    """keeps the last 30s of mic audio, indexed by total samples received"""

    def __init__(self, sr=C.SR, seconds=30, device=None, blocksize=256):
        self.sr = sr
        self.size = int(sr * seconds)
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.total = 0
        self.lock = threading.Lock()
        self.last_cb = None
        self.overflows = 0
        self.stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32", device=device,
                                     blocksize=blocksize, latency="low", callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        if status.input_overflow:
            self.overflows += 1
        x = indata[:, 0]
        n = len(x)
        with self.lock:
            i = self.total % self.size
            if i + n <= self.size:
                self.buf[i:i + n] = x
            else:
                k = self.size - i
                self.buf[i:] = x[:k]
                self.buf[:n - k] = x[k:]
            self.total += n
            self.last_cb = time.perf_counter()

    def start(self):
        self.stream.start()
        t0 = time.time()
        while self.total == 0 and time.time() - t0 < 3:
            time.sleep(0.01)
        if self.total == 0:
            raise RuntimeError("no audio from the mic, check the device (python main.py devices)")
        return self

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def now_index(self):
        # rough guess of which sample is "right now"
        # newest sample in the buffer is about `latency` old when the callback runs
        with self.lock:
            if self.last_cb is None:
                return self.total
            dt = time.perf_counter() - self.last_cb
            return self.total + int((dt + self.stream.latency) * self.sr)

    def get(self, start, end):
        # returns None if that range isn't in the buffer (yet / anymore)
        start, end = int(start), int(end)
        with self.lock:
            if end > self.total or start < self.total - self.size or start >= end:
                return None
            i, j = start % self.size, end % self.size
            if i < j:
                return self.buf[i:j].copy()
            return np.concatenate([self.buf[i:], self.buf[:j]])

    def wait_for(self, index, timeout=2.0):
        t0 = time.time()
        while self.total < index:
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.005)
        return True
