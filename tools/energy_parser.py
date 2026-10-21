#!/usr/bin/env python3
"""
Oscilloscope CSV energy parser for GPIO‑delimited windows.

Given a CSV containing time and shunt voltage columns (and optionally a GPIO
marker column), compute:
- Energy per inference E_inf (J)
- Average current I_avg (A)
- Average power P_avg (W)

Modes:
1) Auto (default): if a marker column is present, detect HIGH windows and
   integrate over them. If multiple pulses are present, report per‑pulse stats.
   If a single long window is present, require --count N to divide energy/time.
2) Manual window: provide --t-start and --t-end to integrate over [t_start, t_end].

Assumptions:
- Time axis in seconds (float)
- Shunt voltage in volts
- Marker is logic‑level voltage (use --threshold to set HIGH threshold)

Example (auto, with marker):
  python3 tools/energy_parser.py --csv scope_dump.csv \
    --time-col time --vshunt-col Math --marker-col CH1 \
    --rshunt 1.0 --vdd 3.3 --count 100

Example (manual window):
  python3 tools/energy_parser.py --csv scope_dump.csv \
    --time-col time --vshunt-col Math --t-start 0.250 --t-end 0.850 \
    --rshunt 0.5 --vdd 3.3 --count 100
"""

import argparse
import csv
import math
from typing import List, Tuple, Optional


def load_csv(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")
    return rows, reader.fieldnames or []


def to_float_list(rows, key: str) -> List[float]:
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except Exception:
            vals.append(float("nan"))
    return vals


def find_windows(time_s: List[float], marker_v: List[float], thr: float) -> List[Tuple[float, float]]:
    windows = []
    high = False
    t0: Optional[float] = None
    for i in range(len(time_s)):
        v = marker_v[i]
        t = time_s[i]
        now_high = v >= thr
        if now_high and not high:
            high = True
            t0 = t
        elif not now_high and high:
            high = False
            if t0 is not None:
                windows.append((t0, t))
                t0 = None
    if high and t0 is not None:
        windows.append((t0, time_s[-1]))
    return windows


def integrate_trapz(x: List[float], y: List[float], t0: float, t1: float) -> float:
    # Integrate y(x) over [t0, t1] using trapezoidal rule on the subset interval
    if t1 <= t0:
        return 0.0
    area = 0.0
    # Walk through segments and clip to window
    for i in range(1, len(x)):
        xa, xb = x[i - 1], x[i]
        ya, yb = y[i - 1], y[i]
        if xb <= t0 or xa >= t1:
            continue
        # Clip segment
        xa_c = max(xa, t0)
        xb_c = min(xb, t1)
        if xb_c <= xa_c:
            continue
        # Linear interpolation for clipped endpoints
        def lerp(xa, xb, ya, yb, xq):
            if xb == xa:
                return ya
            w = (xq - xa) / (xb - xa)
            return ya + w * (yb - ya)
        ya_c = ya if xa_c == xa else lerp(xa, xb, ya, yb, xa_c)
        yb_c = yb if xb_c == xb else lerp(xa, xb, ya, yb, xb_c)
        area += 0.5 * (ya_c + yb_c) * (xb_c - xa_c)
    return area


def main():
    p = argparse.ArgumentParser(description="Parse scope CSV to energy and latency statistics")
    p.add_argument("--csv", required=True, help="CSV exported by scope")
    p.add_argument("--time-col", default="time", help="Time column name (s)")
    p.add_argument("--vshunt-col", default="vshunt", help="Shunt voltage column name (V)")
    p.add_argument("--marker-col", default=None, help="Marker voltage column name (optional)")
    p.add_argument("--threshold", type=float, default=1.0, help="Marker HIGH threshold (V)")
    p.add_argument("--rshunt", type=float, required=True, help="Shunt resistance (ohm)")
    p.add_argument("--vdd", type=float, required=True, help="Supply voltage Vdd (V)")
    p.add_argument("--count", type=int, default=None, help="N inferences in the window (if a single long window)")
    p.add_argument("--t-start", type=float, default=None, help="Manual window start time (s)")
    p.add_argument("--t-end", type=float, default=None, help="Manual window end time (s)")
    args = p.parse_args()

    rows, headers = load_csv(args.csv)
    if args.time_col not in headers:
        raise SystemExit(f"Time column '{args.time_col}' not found. Headers: {headers}")
    if args.vshunt_col not in headers:
        raise SystemExit(f"Shunt column '{args.vshunt_col}' not found. Headers: {headers}")

    t = to_float_list(rows, args.time_col)
    vsh = to_float_list(rows, args.vshunt_col)

    windows: List[Tuple[float, float]] = []
    if args.t_start is not None and args.t_end is not None:
        windows = [(args.t_start, args.t_end)]
    elif args.marker_col and args.marker_col in headers:
        marker = to_float_list(rows, args.marker_col)
        windows = find_windows(t, marker, args.threshold)
        if not windows:
            raise SystemExit("No HIGH windows found from marker. Adjust --threshold or column names.")
    else:
        raise SystemExit("Provide either --marker-col or both --t-start and --t-end.")

    energies = []
    durations = []
    for (t0, t1) in windows:
        # Current = Vshunt / R; Energy = Vdd * integral(I dt)
        area_v = integrate_trapz(t, vsh, t0, t1)  # volt-second
        charge = area_v / args.rshunt  # coulombs
        energy = args.vdd * charge  # joules
        energies.append(energy)
        durations.append(t1 - t0)

    if len(windows) == 1 and (args.count is None or args.count <= 0):
        print("Note: Single window detected and --count not provided. Reporting window totals.")

    # Compute per-inference values
    if len(windows) > 1:
        # Multiple pulses: treat as per-inference already
        e_inf_list = energies
        t_inf_list = durations
    else:
        # Single window: divide by count if provided; else treat as 1 inference
        N = args.count or 1
        e_inf_list = [energies[0] / N]
        t_inf_list = [durations[0] / N]

    # Aggregates
    def mean(a):
        return sum(a) / len(a) if a else math.nan
    def std(a):
        if len(a) < 2:
            return 0.0
        m = mean(a)
        return math.sqrt(sum((x - m) ** 2 for x in a) / len(a))

    e_mean = mean(e_inf_list)
    e_std = std(e_inf_list)
    t_mean = mean(t_inf_list)
    t_std = std(t_inf_list)

    # Average current/power over the integrated windows (overall)
    total_energy = sum(energies)
    total_time = sum(durations)
    i_avg = (total_energy / args.vdd) / total_time if total_time > 0 else math.nan
    p_avg = total_energy / total_time if total_time > 0 else math.nan

    print("Results:")
    print(f"  Windows: {len(windows)}")
    print(f"  E_inf mean: {e_mean:.6e} J  (std: {e_std:.6e})")
    print(f"  T_inf mean: {t_mean:.6e} s  (std: {t_std:.6e})")
    print(f"  I_avg over window(s): {i_avg:.6e} A")
    print(f"  P_avg over window(s): {p_avg:.6e} W")


if __name__ == "__main__":
    main()

