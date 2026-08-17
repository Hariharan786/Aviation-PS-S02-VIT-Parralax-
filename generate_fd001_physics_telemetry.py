
import random
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from model.bearing_vibration import build_bearing_summary, synthesize_cycle_waveform, add_bearing_status

COLUMNS = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30", "Nf", "Nc", "epr",
    "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
    "Nf_dmd", "PCNfR_dmd", "W31", "W32"
]

LSTM_FEATURES = [
    "setting_1", "setting_2", "setting_3",
    "T24", "T30", "T50", "P30", "Ps30", "Nc", "NRc", "phi", "epr"
]

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def scale(x, src_lo, src_hi, dst_lo, dst_hi):
    if src_hi == src_lo:
        return (dst_lo + dst_hi) / 2.0
    return dst_lo + ((x - src_lo) / (src_hi - src_lo)) * (dst_hi - dst_lo)

def physics_state(cycle, cycles, rng, degradation):
    """Physics-inspired flight/engine state."""
    phase = (cycle - 1) / max(1, cycles - 1)

    altitude_ft = clamp(
        22000 + 11000 * math.sin(2 * math.pi * phase)
        + rng.uniform(-1000, 1000),
        10000, 35000
    )

    mach = clamp(
        0.60 + 0.16 * math.sin(2 * math.pi * phase + 0.4)
        + rng.uniform(-0.015, 0.015),
        0.4, 0.8
    )

    throttle = clamp(
        82 + 15 * math.sin(2 * math.pi * phase - 0.5)
        + rng.uniform(-2, 2),
        65, 100
    )

    gamma = 1.4
    gamma_exp = (gamma - 1) / gamma
    compressor_efficiency = 0.85

    t_ambient_R = 518.67 - 3.56 * (altitude_ft / 1000)
    p_ambient_psia = 14.7 * ((1 - 6.875e-6 * altitude_ft) ** 5.256)

    ram = 1 + 0.2 * mach**2
    t_inlet_R = t_ambient_R * ram
    p_inlet_psia = p_ambient_psia * ram**3.5

    core_rpm = 7000 + 30 * throttle
    fan_rpm = 1500 + 10 * throttle

    pressure_ratio = 10 + 15 * (throttle / 100)
    effective_pr = pressure_ratio * (1 - 0.10 * degradation)

    p_hpc_out = p_inlet_psia * effective_pr
    t_hpc_ideal = t_inlet_R * effective_pr**gamma_exp
    t_hpc_out = t_inlet_R + (t_hpc_ideal - t_inlet_R) / compressor_efficiency

    fuel_ratio = 200 + 3.5 * effective_pr * (core_rpm / 10000)
    t_comb = t_hpc_out + 800 + 12 * throttle + 35 * degradation
    t_lpt = 0.4 * t_comb

    return {
        "altitude": altitude_ft,
        "mach": mach,
        "throttle": throttle,
        "t_inlet": t_inlet_R,
        "p_inlet": p_inlet_psia,
        "p_hpc": p_hpc_out,
        "t_hpc": t_hpc_out,
        "t_comb": t_comb,
        "t_lpt": t_lpt,
        "core_rpm": core_rpm,
        "fan_rpm": fan_rpm,
        "fuel_ratio": fuel_ratio,
        "degradation": degradation,
    }

def physics_to_fd001_settings(state):
    # These are C-MAPSS-compatible encodings of the physical flight inputs.
    return {
        "setting_1": scale(state["altitude"], 10000, 35000, -0.0087, 0.0087),
        "setting_2": scale(state["mach"], 0.4, 0.8, -0.0006, 0.0006),
        "setting_3": 100.0,
    }

def generate(
    train_file="train_FD001.txt",
    output_file="FD001_physics_live_test.txt",
    num_engines=10,
    cycles_per_engine=30,
    seed=42,
    generate_bearing=True,
    bearing_sample_rate_hz=2048,
    bearing_duration_s=0.5,
):
    """
    Generate telemetry that the existing AeroGuard FD001 frontend can ingest.

    The output contains NO RUL/health label.
    Each engine has 30 cycles because the current FD001 LSTM uses a
    30-cycle sequence.

    The physics equations drive the operating settings and the major
    temperature/pressure/speed trends. A real FD001 row is used as the
    statistical baseline for secondary C-MAPSS channels so the generated
    vector remains close to the training distribution.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    train_path = Path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_file} not found. Put this script beside train_FD001.txt."
        )

    train = pd.read_csv(
        train_path, sep=r"\s+", header=None,
        names=COLUMNS, engine="python"
    )

    sensor_cols = COLUMNS[5:]
    ranges = {
        c: (float(train[c].min()), float(train[c].max()))
        for c in sensor_cols
    }
    means = train[sensor_cols].mean()
    stds = train[sensor_cols].std().replace(0, 1e-6)

    rows = []

    for engine_id in range(1, num_engines + 1):
        # Hidden degradation severity. It is NOT written into the file.
        severity = rng.uniform(0.0, 1.0)

        for cycle in range(1, cycles_per_engine + 1):
            progress = (cycle - 1) / max(1, cycles_per_engine - 1)
            degradation = severity * progress**2

            state = physics_state(
                cycle, cycles_per_engine, rng, degradation
            )

            settings = physics_to_fd001_settings(state)

            # Pick a real FD001 row close to the physics-derived operating
            # setting. This preserves realistic multivariate relationships.
            target_s1 = settings["setting_1"]
            target_s2 = settings["setting_2"]

            distance = (
                ((train["setting_1"] - target_s1) / 0.0087) ** 2
                + ((train["setting_2"] - target_s2) / 0.0006) ** 2
            )

            # Randomly sample among the nearest healthy-like operating points.
            nearest_idx = np.argsort(distance.to_numpy())[:100]
            base = train.iloc[int(rng.choice(nearest_idx))].copy()

            row = {c: float(base[c]) for c in COLUMNS[2:]}
            row["engine_id"] = engine_id
            row["cycle"] = cycle

            # Preserve the physics-derived operating conditions.
            row["setting_1"] = settings["setting_1"]
            row["setting_2"] = settings["setting_2"]
            row["setting_3"] = 100.0

            # Physics-driven normalized deviations. The multipliers are
            # intentionally moderate so the result remains close to C-MAPSS.
            def perturb(sensor, signal, source_lo, source_hi,
                        sensitivity=0.25, sign=1.0):
                normalized = (
                    (signal - source_lo) /
                    max(source_hi - source_lo, 1e-9)
                )
                centered = normalized - 0.5
                value = (
                    row[sensor]
                    + sign * centered * sensitivity * stds[sensor]
                    + np_rng.normal(0, 0.025 * stds[sensor])
                )
                return clamp(value, ranges[sensor][0], ranges[sensor][1])

            row["T24"] = perturb(
                "T24", state["t_inlet"], 400, 540, 0.35
            )
            row["T30"] = perturb(
                "T30", state["t_hpc"], 900, 1500, 0.40,
                sign=1.0
            )
            row["T50"] = perturb(
                "T50", state["t_comb"], 2300, 3300, 0.45,
                sign=1.0
            )
            row["P30"] = perturb(
                "P30", state["p_hpc"], 50, 100, 0.40,
                sign=1.0
            )
            row["Ps30"] = perturb(
                "Ps30", state["p_inlet"], 3, 15, 0.35
            )
            row["Nc"] = perturb(
                "Nc", state["core_rpm"], 9000, 10000, 0.30
            )
            row["NRc"] = perturb(
                "NRc", state["core_rpm"], 8000, 9000, 0.30
            )
            row["phi"] = perturb(
                "phi", state["fuel_ratio"], 200, 230, 0.30
            )

            # Small degradation signature in the main health sensors.
            row["T30"] = clamp(
                row["T30"] + degradation * 0.30 * stds["T30"],
                *ranges["T30"]
            )
            row["T50"] = clamp(
                row["T50"] + degradation * 0.35 * stds["T50"],
                *ranges["T50"]
            )
            row["P30"] = clamp(
                row["P30"] + degradation * 0.25 * stds["P30"],
                *ranges["P30"]
            )
            row["Ps30"] = clamp(
                row["Ps30"] + degradation * 0.20 * stds["Ps30"],
                *ranges["Ps30"]
            )
            row["Nc"] = clamp(
                row["Nc"] - degradation * 0.25 * stds["Nc"],
                *ranges["Nc"]
            )
            row["NRc"] = clamp(
                row["NRc"] - degradation * 0.25 * stds["NRc"],
                *ranges["NRc"]
            )

            # Keep the remaining C-MAPSS channels statistically anchored
            # to the sampled training row, with tiny telemetry noise.
            for sensor in sensor_cols:
                if sensor in {
                    "T24", "T30", "T50", "P30", "Ps30",
                    "Nc", "NRc", "phi"
                }:
                    continue

                row[sensor] = clamp(
                    row[sensor]
                    + np_rng.normal(0, 0.015 * stds[sensor]),
                    ranges[sensor][0],
                    ranges[sensor][1]
                )

            rows.append(row)

    out = pd.DataFrame(rows, columns=COLUMNS)

    with open(output_file, "w", newline="") as f:
        for _, row in out.iterrows():
            values = []
            for c in COLUMNS:
                if c in {"engine_id", "cycle"}:
                    values.append(str(int(row[c])))
                else:
                    values.append(f"{row[c]:.6f}")
            f.write(" ".join(values) + "\n")

    # Optional CSV for human inspection; not used by the C-MAPSS TXT loader.
    csv_file = str(Path(output_file).with_suffix(".csv"))
    out.to_csv(csv_file, index=False)

    if generate_bearing:
        bearing_summary = add_bearing_status(build_bearing_summary(out, bearing_sample_rate_hz, bearing_duration_s, seed))
        bearing_summary_file = str(Path(output_file).with_name(Path(output_file).stem + "_bearing_summary.csv"))
        bearing_summary.to_csv(bearing_summary_file, index=False)

        # Keep the default artifact compact: write a representative waveform for the
        # latest cycle of the first engine. Use --all-bearings via the helper module
        # if a full raw waveform export is desired.
        first = out.sort_values(["engine_id", "cycle"]).iloc[-1]
        wave, _ = synthesize_cycle_waveform(first, bearing_sample_rate_hz, bearing_duration_s, seed)
        waveform_file = str(Path(output_file).with_name(Path(output_file).stem + "_bearing_waveform.csv"))
        wave.to_csv(waveform_file, index=False)
        print(f"Bearing summary: {bearing_summary_file}")
        print(f"Representative bearing waveform: {waveform_file}")

    print(f"Generated {len(out)} telemetry rows.")
    print(f"Engines: {num_engines}")
    print(f"Cycles/engine: {cycles_per_engine}")
    print(f"TXT: {output_file}")
    print(f"CSV: {csv_file}")


if __name__ == "__main__":
    generate()
