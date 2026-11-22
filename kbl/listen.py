# test mode - no keyboard, just the mic
import time

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from . import config as C
from .audio import AudioRing
from .features import envelope, extract, find_onset, highpass_sos
from .keys import MODIFIERS, label_text, type_label
from .train import load_model


def run(model_path=C.MODEL_PATH, device=None, ratio=6.0, threshold=None,
        min_conf=0.25, type_out=False, debug=False, autocap=True):
    bundle = load_model(model_path)
    model, sr = bundle["model"], bundle["sr"]
    classes = list(model.classes_)
    print(f"model: {bundle['model_name']}, {len(classes)} labels, "
          f"cv accuracy {bundle['cv_accuracy']:.1%} (trained {bundle['trained_at']})")

    controller = None
    if type_out:
        from pynput.keyboard import Controller
        controller = Controller()
        print("type mode on, guesses get typed into whatever window is focused")

    hop = C.HOP
    sos = highpass_sos(sr)
    ring = AudioRing(sr=sr, device=device)
    with ring:
        print("measuring background noise, stay quiet for 2 sec...")
        time.sleep(2.0)
        end = ring.total - ring.total % hop
        calib = ring.get(end - int(1.5 * sr), end)
        floor = float(np.median(envelope(sosfilt(sos, calib)[int(0.2 * sr):])))
        thr = threshold if threshold else floor * ratio
        msg = f"  noise floor {floor:.5f}, trigger at {thr:.5f}"
        if bundle.get("peak_p10"):
            msg += f"  (training clicks ~{bundle['peak_median']:.4f}, quiet ones {bundle['peak_p10']:.4f})"
        print(msg)
        if bundle.get("peak_p10") and thr > bundle["peak_p10"]:
            print("  trigger is higher than a lot of the training clicks, try a lower --ratio")
        print("\nlistening, start typing. ctrl+c to stop\n")

        zi = sosfilt_zi(sos) * 0.0
        pos = end
        armed, refractory_until = True, 0
        pending = []
        text = ""
        try:
            while True:
                total = ring.total
                n = ((total - pos) // hop) * hop
                if n > 0:
                    x = ring.get(pos, pos + n)
                    if x is None:  # fell behind, just skip ahead
                        pos = total - total % hop
                        continue
                    yf, zi = sosfilt(sos, x, zi=zi)
                    for k, e in enumerate(envelope(yf)):
                        fi = pos + k * hop
                        if armed and fi >= refractory_until and e > thr:
                            pending.append(fi)
                            armed = False
                            refractory_until = fi + int(C.REFRACTORY * sr)
                        elif not armed and fi >= refractory_until and e < thr * 0.6:
                            armed = True
                        if threshold is None and e < floor * 2:  # slowly follow the background noise
                            floor = 0.999 * floor + 0.001 * e
                            thr = floor * ratio
                    pos += n

                # wait until we have the whole clip before guessing
                for t in list(pending):
                    need = t + int((C.LIVE_AFTER + C.SAVE_POST) * sr) + hop
                    if total < need:
                        continue
                    pending.remove(t)
                    text = guess(ring, model, classes, sr, t, need, min_conf, controller, text, debug, autocap)
                time.sleep(0.005)
        except KeyboardInterrupt:
            pass
    print(f"\nstopped. you typed:\n{text}")


def sentence_start(text):
    if text == "" or text.endswith("\n"):
        return True
    return text.endswith(" ") and text.rstrip(" ").endswith(".")


def guess(ring, model, classes, sr, t, need, min_conf, controller, text, debug, autocap):
    start = t - int((C.LIVE_BEFORE + C.SAVE_PRE) * sr)
    region = ring.get(start, need)
    if region is None:
        return text
    lo = int(C.SAVE_PRE * sr)
    hi = lo + int((C.LIVE_BEFORE + C.LIVE_AFTER) * sr)
    found = find_onset(region, sr, lo, hi)
    if found is None:
        return text
    off, peak, _ = found
    onset = start + off
    clip = ring.get(onset - int(C.SAVE_PRE * sr), onset + int(C.SAVE_POST * sr))
    if clip is None:
        return text

    proba = model.predict_proba(extract(clip, sr)[None, :])[0]
    order = np.argsort(proba)[::-1]
    best, conf = classes[order[0]], float(proba[order[0]])
    alts = ", ".join(f"{classes[i]} {proba[i]:.0%}" for i in order[1:4])

    if best == C.RELEASE_LABEL:
        if debug:
            print(f"  (release, {conf:.0%}, level {peak:.4f})")
        return text
    if conf < min_conf:
        print(f"  {'?':>10}  {conf:4.0%}   maybe: {best}, {alts}")
        return text

    # no shift key, so just capitalize the start of sentences
    out = label_text(best) or ""
    if autocap and out.isalpha() and sentence_start(text):
        out = out.upper()
    if best == "backspace":
        text = text[:-1]
    elif best == "enter":
        text += "\n"
    else:
        text += out

    if controller is not None and best not in MODIFIERS:
        if out.isupper():
            controller.tap(out)
        else:
            type_label(controller, best)

    lvl = f"  level {peak:.4f}" if debug else ""
    print(f"  {best:>10}  {conf:4.0%}   alt: {alts:<32}{lvl} | {text.split(chr(10))[-1][-40:]}")
    return text
