import argparse
import time

from kbl import config as C


def cmd_devices(args):
    import sounddevice as sd
    print(sd.query_devices())
    print("\nuse --device N to pick a mic")


def cmd_meter(args):
    import numpy as np
    from scipy.signal import sosfilt
    from kbl.audio import AudioRing
    from kbl.features import envelope, highpass_sos

    sos = highpass_sos(C.SR)
    print("mic level for 20 sec (ctrl+c to stop), tap some keys and you should see spikes")
    with AudioRing(device=args.device) as ring:
        t_end = time.time() + 20
        try:
            while time.time() < t_end:
                time.sleep(0.1)
                x = ring.get(ring.total - int(0.1 * C.SR), ring.total)
                if x is None:
                    continue
                peak = float(envelope(sosfilt(sos, x)).max())
                bar = "#" * int(min(60, max(0, 20 * np.log10(peak + 1e-9) + 80)))
                print(f"\r{peak:8.5f} {bar:<60}", end="", flush=True)
        except KeyboardInterrupt:
            pass
    print()


def cmd_record(args):
    from kbl import record
    from kbl.keys import parse_key_set
    record.run(args.data, args.device, parse_key_set(args.keys), args.target, not args.no_release)


def cmd_stats(args):
    from kbl import dataset
    dataset.print_counts(args.data)


def cmd_train(args):
    from kbl import train
    from kbl.keys import parse_key_set
    only = parse_key_set(args.keys)
    if only is not None and not args.no_release:
        only = list(only) + [C.RELEASE_LABEL]
    train.run(args.data, args.model, args.min_samples, not args.no_augment, only)


def cmd_test(args):
    from kbl import listen
    listen.run(args.model, args.device, args.ratio, args.threshold, args.min_confidence,
               args.type, args.debug, not args.no_autocap)


def device_arg(s):
    return int(s) if s.isdigit() else s


def main():
    p = argparse.ArgumentParser(description="guess which key you pressed from the sound")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("devices", help="list mics")
    sp.set_defaults(fn=cmd_devices)

    sp = sub.add_parser("meter", help="live mic level")
    sp.add_argument("--device", type=device_arg)
    sp.set_defaults(fn=cmd_meter)

    sp = sub.add_parser("record", help="training mode, saves a clip for every key press")
    sp.add_argument("--data", default=C.DATA_DIR)
    sp.add_argument("--device", type=device_arg)
    sp.add_argument("--keys", default="sentences",
                    help="sentences (default) | letters | digits | alnum | typing | full | any | a,s,d,f")
    sp.add_argument("--target", type=int, help="samples you want per key")
    sp.add_argument("--no-release", action="store_true", help="don't save key release sounds")
    sp.set_defaults(fn=cmd_record)

    sp = sub.add_parser("stats", help="how many samples each key has")
    sp.add_argument("--data", default=C.DATA_DIR)
    sp.set_defaults(fn=cmd_stats)

    sp = sub.add_parser("train", help="train on the recorded clips")
    sp.add_argument("--data", default=C.DATA_DIR)
    sp.add_argument("--model", default=C.MODEL_PATH)
    sp.add_argument("--keys", help="only train on these keys")
    sp.add_argument("--min-samples", type=int, default=5)
    sp.add_argument("--no-augment", action="store_true")
    sp.add_argument("--no-release", action="store_true")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("test", help="test mode, guesses keys from sound only")
    sp.add_argument("--model", default=C.MODEL_PATH)
    sp.add_argument("--device", type=device_arg)
    sp.add_argument("--ratio", type=float, default=6.0, help="trigger at noise floor x this")
    sp.add_argument("--threshold", type=float, help="fixed trigger level instead of --ratio")
    sp.add_argument("--min-confidence", type=float, default=0.25)
    sp.add_argument("--type", action="store_true", help="actually type the guesses")
    sp.add_argument("--debug", action="store_true")
    sp.add_argument("--no-autocap", action="store_true")
    sp.set_defaults(fn=cmd_test)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
