"""Render one instrument or event to a mono wav for listening.

usage: preview_instrument.py <name> [--seconds 4] [--param k=v ...] [--out x.wav]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from montagelib.synth import EVENTS, INSTRUMENTS, SR  # noqa: E402


def _value(v: str):
    return [float(x) for x in v.split(",")] if "," in v else float(v)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("name")
    ap.add_argument("--seconds", type=float, default=4.0, help="instrument length (events: ignored)")
    ap.add_argument("--param", action="append", default=[], metavar="k=v",
                    help="instrument parameter, lists as comma-separated numbers")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.name not in INSTRUMENTS and a.name not in EVENTS:
        print(f"unknown name {a.name!r}\ninstruments: {' '.join(INSTRUMENTS)}\n"
              f"events: {' '.join(EVENTS)}", file=sys.stderr)
        return 2
    try:
        params = {k: _value(v) for k, v in (s.split("=", 1) for s in a.param)}
    except ValueError:
        print(f"bad --param {a.param!r}: expected k=number or k=n1,n2,...", file=sys.stderr)
        return 2
    try:
        x = (INSTRUMENTS[a.name](a.seconds, **params) if a.name in INSTRUMENTS
             else EVENTS[a.name](**params))
    except TypeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x * (0.9 / peak)
    out = pathlib.Path(a.out or f"{a.name}.wav")
    wavfile.write(str(out), SR, (x * 32767).astype(np.int16))
    print(out, f"{len(x) / SR:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
