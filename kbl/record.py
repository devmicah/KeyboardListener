# training mode - type keys, save a labeled clip for each press
import queue
import threading
import time

from . import config as C
from . import dataset
from .audio import AudioRing
from .features import find_onset
from .keys import key_label


def run(data_dir=C.DATA_DIR, device=None, wanted=None, target=None, release=True):
    from pynput.keyboard import Key, Listener

    sr = C.SR
    print("current samples:")
    dataset.print_counts(data_dir, wanted)
    print("\nrecording. press keys one at a time, not too fast (~3 a second max)")
    print("mix it up a bit, different fingers / how hard you press. ESC to stop\n")

    ring = AudioRing(sr=sr, device=device)
    events = queue.Queue()
    stop = threading.Event()
    press_lock = threading.Lock()
    press_times = []   # sample index of every press
    last_press = {}    # label -> index of its last press
    held = set()
    counts = dataset.counts(data_dir)
    saved = skipped = 0

    def on_press(key):
        if key == Key.esc:
            stop.set()
            return False
        label = key_label(key)
        if label is None or label in held:  # held = windows auto repeat
            return
        held.add(label)
        idx = ring.now_index()
        with press_lock:
            press_times.append(idx)
        last_press[label] = idx
        if wanted is None or label in wanted:
            events.put(("press", label, idx, None))

    def on_release(key):
        label = key_label(key)
        if label is None:
            return
        held.discard(label)
        if release and (wanted is None or label in wanted) and label in last_press:
            events.put(("release", label, ring.now_index(), last_press[label]))

    def near_other_press(idx, window, exclude=None):
        with press_lock:
            return any(abs(p - idx) < window for p in press_times if p != exclude)

    def process(kind, label, idx, press_idx):
        nonlocal saved, skipped
        pre = int((C.SEARCH_BEFORE + C.SAVE_PRE) * sr)
        post = int((C.SEARCH_AFTER + C.SAVE_POST) * sr) + C.HOP
        if not ring.wait_for(idx + post, timeout=2):
            return
        start = idx - pre
        region = ring.get(start, idx + post)
        if region is None:
            return

        if kind == "press":
            if near_other_press(idx, int(C.MIN_GAP * sr), exclude=idx):
                skipped += 1
                print(f"  skip   {label:<10} too close to another press, slow down a bit")
                return
            lo = int(C.SAVE_PRE * sr)
            hi = lo + int((C.SEARCH_BEFORE + C.SEARCH_AFTER) * sr)
        else:
            # only keep releases that are clearly away from any press
            if idx - press_idx < int(0.15 * sr) or near_other_press(idx, int(0.15 * sr)):
                return
            lo, hi = pre - int(0.04 * sr), pre + int(0.08 * sr)

        found = find_onset(region, sr, lo, hi)
        if found is None:
            return
        off, peak, floor = found
        if peak < floor * 4:
            if kind == "press":
                skipped += 1
                print(f"  skip   {label:<10} didn't hear a click (mic too far / too quiet?)")
            return

        onset = start + off
        clip = ring.get(onset - int(C.SAVE_PRE * sr), onset + int(C.SAVE_POST * sr))
        if clip is None:
            return
        name = C.RELEASE_LABEL if kind == "release" else label
        dataset.save_clip(data_dir, name, clip, sr)
        counts[name] += 1
        if kind == "release":
            return

        saved += 1
        line = f"  saved  {label:<10} ({counts[label]:3d})  level {peak:.4f}  snr {peak / floor:5.1f}x"
        if wanted:
            need = sorted((counts.get(k, 0), k) for k in wanted)
            if target:
                todo = [k for n, k in need if n < target]
                line += ("   next: " + " ".join(todo[:6])) if todo else "   all keys at target!"
            else:
                line += f"   lowest: {need[0][1]} ({need[0][0]})"
        print(line)

    def worker():
        while not (stop.is_set() and events.empty()):
            try:
                ev = events.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                process(*ev)
            except Exception as e:
                print(f"  error on {ev[1]}: {e}")

    with ring:
        th = threading.Thread(target=worker, daemon=True)
        th.start()
        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()
        try:
            while not stop.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop.set()
        listener.stop()
        th.join(timeout=3)

    flush_stdin()
    print(f"\ndone. saved {saved} presses ({skipped} skipped)")
    if ring.overflows:
        print(f"heads up: {ring.overflows} audio overflows, some clips might be messed up")
    dataset.print_counts(data_dir, wanted)


def flush_stdin():
    # otherwise everything you typed while recording gets dumped into the terminal
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    except ImportError:
        try:
            import sys, termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass
