#!/usr/bin/env python3
"""
Raspberry Pi GPIO benchmarking script (two markers: preprocess vs inference).

Features:
- Uses BCM numbering and toggles two GPIOs:
  * pin A: HIGH during preprocess(x) (e.g., STFT/mel)
  * pin B: HIGH during run_inference(features)
- Measures time with time.perf_counter_ns() for each phase.
- Imports functions dynamically via importlib.
- Iterates inputs from a folder and passes the path to preprocess.
- Prints stats and writes a CSV with timestamps and durations.

Example:
  python3 rpi_benchmark_two_markers.py \
    --module my_model \
    --pre preprocess \
    --infer run_inference \
    --inputs /path/a/input_wav/ \
    --pin-a 23 --pin-b 24 \
    --loops 100 --warmup 10 \
    --csv rpi_zero2w.csv
"""

import argparse
import importlib
import os
import statistics as stats
import sys
import time
from typing import List


def list_inputs(input_path: str) -> List[str]:
    if input_path is None:
        return []
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
        files = [f for f in files if os.path.isfile(f)]
        files.sort()
        return files
    if os.path.isfile(input_path):
        return [input_path]
    return []


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi GPIO benchmarking (two markers)")
    parser.add_argument("--module", required=True, help="Python module to import (e.g., my_model)")
    parser.add_argument("--pre", required=True, help="Preprocess function name (e.g., preprocess)")
    parser.add_argument("--infer", required=True, help="Inference function name (e.g., run_inference)")
    parser.add_argument("--inputs", required=False, default=None, help="Path to input folder or file")
    parser.add_argument("--pin-a", type=int, default=23, help="BCM GPIO pin for preprocess marker (default: 23)")
    parser.add_argument("--pin-b", type=int, default=24, help="BCM GPIO pin for inference marker (default: 24)")
    parser.add_argument("--loops", type=int, default=100, help="Number of measured iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (not recorded)")
    parser.add_argument("--csv", required=True, help="Output CSV filepath")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-iteration printing")
    args = parser.parse_args()

    try:
        import RPi.GPIO as GPIO
    except Exception as e:
        print("Error: RPi.GPIO not available. Install with 'pip3 install RPi.GPIO' on Raspberry Pi.")
        print(f"Details: {e}")
        sys.exit(1)

    # Dynamic import of the module and target functions
    mod = importlib.import_module(args.module)
    if not hasattr(mod, args.pre):
        print(f"Error: preprocess function '{args.pre}' not found in module '{args.module}'")
        sys.exit(1)
    if not hasattr(mod, args.infer):
        print(f"Error: inference function '{args.infer}' not found in module '{args.module}'")
        sys.exit(1)
    pre_fn = getattr(mod, args.pre)
    infer_fn = getattr(mod, args.infer)

    inputs = list_inputs(args.inputs)
    if not inputs:
        print("Warning: no inputs found; will call preprocess(None) then run_inference(features).")

    # Setup GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(args.pin_a, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(args.pin_b, GPIO.OUT, initial=GPIO.LOW)

    pin_a = args.pin_a
    pin_b = args.pin_b

    pre_durations_ns = []
    inf_durations_ns = []

    def get_input(i):
        if not inputs:
            return None
        return inputs[i % len(inputs)]

    # Warmup
    for i in range(max(0, args.warmup)):
        x = get_input(i)
        try:
            features = pre_fn(x)
        except TypeError:
            features = pre_fn()
        try:
            infer_fn(features)
        except TypeError:
            infer_fn()

    # Run measured loops
    rows = []
    for i in range(args.loops):
        x = get_input(i)

        # Preprocess phase
        t_pre_start = time.perf_counter_ns()
        try:
            GPIO.output(pin_a, GPIO.HIGH)
            features = pre_fn(x)
        except TypeError:
            GPIO.output(pin_a, GPIO.HIGH)
            features = pre_fn()
        finally:
            GPIO.output(pin_a, GPIO.LOW)
        t_pre_end = time.perf_counter_ns()
        pre_dur = t_pre_end - t_pre_start
        pre_durations_ns.append(pre_dur)

        # Inference phase
        t_inf_start = time.perf_counter_ns()
        try:
            GPIO.output(pin_b, GPIO.HIGH)
            infer_fn(features)
        except TypeError:
            GPIO.output(pin_b, GPIO.HIGH)
            infer_fn()
        finally:
            GPIO.output(pin_b, GPIO.LOW)
        t_inf_end = time.perf_counter_ns()
        inf_dur = t_inf_end - t_inf_start
        inf_durations_ns.append(inf_dur)

        total_dur = (t_inf_end - t_pre_start)
        if not args.quiet:
            print(f"[{i+1:04d}] pre={pre_dur/1e6:.3f} ms, infer={inf_dur/1e6:.3f} ms, total={total_dur/1e6:.3f} ms")
        rows.append((i + 1, x if isinstance(x, str) else "",
                     t_pre_start, t_pre_end, pre_dur,
                     t_inf_start, t_inf_end, inf_dur,
                     total_dur, pin_a, pin_b))

    # Stats
    def smean(a):
        return stats.mean(a) if a else 0
    def smedian(a):
        return stats.median(a) if a else 0
    def sstd(a):
        return stats.pstdev(a) if len(a) > 1 else 0

    print("\nResults (ms):")
    print(f"  preprocess: mean={smean(pre_durations_ns)/1e6:.3f}, median={smedian(pre_durations_ns)/1e6:.3f}, std={sstd(pre_durations_ns)/1e6:.3f}")
    print(f"  inference:  mean={smean(inf_durations_ns)/1e6:.3f}, median={smedian(inf_durations_ns)/1e6:.3f}, std={sstd(inf_durations_ns)/1e6:.3f}")

    # Write CSV
    header = (
        "index,input," \
        "t_pre_start_ns,t_pre_end_ns,pre_duration_ns," \
        "t_inf_start_ns,t_inf_end_ns,inf_duration_ns," \
        "total_duration_ns,marker_pin_a,marker_pin_b\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
    with open(args.csv, "w", encoding="utf-8") as f:
        f.write(header)
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

    # Cleanup
    GPIO.cleanup()


if __name__ == "__main__":
    main()
