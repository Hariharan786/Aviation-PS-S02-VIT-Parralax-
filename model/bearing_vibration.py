"""Physics-informed bearing vibration synthesis and analysis.

The C-MAPSS data is cycle-level telemetry, not raw accelerometer data. This module
bridges that gap by deriving operating conditions (Mach, airspeed, dynamic pressure,
shaft speed) from each cycle and synthesising an accelerometer-like waveform using
bearing characteristic frequencies plus broadband/structural noise.

It is intentionally a *physics-informed synthetic signal*, not a replacement for
certified measured vibration data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BearingGeometry:
    rolling_elements: int = 9
    ball_diameter_mm: float = 10.0
    pitch_diameter_mm: float = 50.0
    contact_angle_deg: float = 0.0


DEFAULT_GEOMETRY = BearingGeometry()


def _safe_scale(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


def derive_operating_state(row: pd.Series) -> Dict[str, float]:
    """Derive physical operating variables from one C-MAPSS cycle."""
    s1 = float(row.get("setting_1", 0.0))
    s2 = float(row.get("setting_2", 0.0))
    nc = float(row.get("Nc", 9060.0))
    nrc = float(row.get("NRc", 8140.0))
    t30 = float(row.get("T30", 1300.0))
    p30 = float(row.get("P30", 100.0))

    altitude_ft = float(np.clip(25000.0 + (s1 / 0.0087) * 15000.0, 10000.0, 35000.0))
    mach = float(np.clip(0.60 + (s2 / 0.0006) * 0.20, 0.40, 0.80))
    altitude_m = altitude_ft * 0.3048

    # ISA-like atmosphere approximation, consistent with the existing engine model.
    temp_k = max(216.65, 288.15 - 0.0065 * altitude_m)
    pressure_pa = 101325.0 * (max(1e-4, 1.0 - 2.25577e-5 * altitude_m) ** 5.25588)
    a_mps = math.sqrt(1.4 * 287.05 * temp_k)
    airspeed_mps = mach * a_mps
    rho = pressure_pa / (287.05 * temp_k)
    dynamic_pressure_pa = 0.5 * rho * airspeed_mps ** 2

    # Nc/NRc are corrected speeds in C-MAPSS, so use their relative movement to
    # estimate an equivalent mechanical shaft speed rather than treating Nc as rpm.
    core_rpm = float(np.clip(7000.0 + 30.0 * (nc - 9020.0), 7000.0, 13800.0))
    fan_rpm = float(np.clip(1500.0 + 10.0 * (nrc - 8100.0), 1500.0, 3440.0))

    throttle_proxy = float(np.clip(
        65.0 + 35.0 * (0.55 * _safe_scale(nc, 9020.0, 9245.0)
                        + 0.45 * _safe_scale(t30, 1100.0, 1600.0)),
        65.0, 100.0,
    ))
    load = float(np.clip(
        0.45 * _safe_scale(p30, 80.0, 120.0)
        + 0.35 * _safe_scale(t30, 1100.0, 1600.0)
        + 0.20 * _safe_scale(dynamic_pressure_pa, 5000.0, 20000.0),
        0.0, 1.0,
    ))

    return {
        "altitude_ft": altitude_ft,
        "mach": mach,
        "temperature_k": temp_k,
        "pressure_pa": pressure_pa,
        "airspeed_mps": airspeed_mps,
        "dynamic_pressure_pa": dynamic_pressure_pa,
        "core_rpm": core_rpm,
        "fan_rpm": fan_rpm,
        "throttle_proxy_pct": throttle_proxy,
        "load_factor": load,
    }


def estimate_cycle_timing(row: pd.Series, previous_row: pd.Series | None = None) -> Dict[str, float]:
    """Estimate variable flight time represented by one engine cycle.

    C-MAPSS does not contain wall-clock timestamps or route distance. We therefore
    estimate the duration of each mission segment from Mach, altitude, throttle,
    and the change in altitude between adjacent cycles. The estimate is deliberately
    variable: higher true airspeed shortens a segment, climb/descent adds a vertical
    path component, and high-throttle segments use a larger mission-distance proxy.
    """
    state = derive_operating_state(row)
    true_airspeed = max(state["airspeed_mps"], 95.0)
    throttle = state["throttle_proxy_pct"]
    altitude_ft = state["altitude_ft"]

    # Nominal segment distance varies with power setting; not a fixed time step.
    horizontal_distance_nm = 1.10 + 0.55 * _safe_scale(throttle, 65.0, 100.0)
    # Higher altitude expands the cruise segment slightly; low altitude tends to
    # represent climb/approach slices in this synthetic mission profile.
    horizontal_distance_nm *= 0.92 + 0.16 * _safe_scale(altitude_ft, 10000.0, 35000.0)
    horizontal_distance_m = horizontal_distance_nm * 1852.0

    vertical_distance_m = 0.0
    climb_rate_mps = 0.0
    if previous_row is not None:
        prev_state = derive_operating_state(previous_row)
        delta_alt_m = state["altitude_ft"] * 0.3048 - prev_state["altitude_ft"] * 0.3048
        vertical_distance_m = abs(delta_alt_m)
        # Approximate vertical speed over the current segment; bounded to plausible
        # transport-aircraft climb/descent rates for a simulation.
        climb_rate_mps = float(np.clip(
            2.0 + 0.06 * (state["throttle_proxy_pct"] - 65.0), 2.0, 22.0
        ))

    horizontal_time_s = horizontal_distance_m / true_airspeed
    vertical_time_s = vertical_distance_m / max(climb_rate_mps, 1.0)
    # Couple throttle and Mach to the segment estimate; this remains a flight-time
    # model rather than the UI refresh interval.
    time_s = horizontal_time_s + 0.35 * vertical_time_s
    time_s *= 1.0 + 0.08 * abs(state["mach"] - 0.6) / 0.2
    time_s = float(np.clip(time_s, 3.0, 30.0))
    ground_distance_nm = float((true_airspeed * time_s) / 1852.0)
    return {
        "cycle_duration_s": time_s,
        "flight_distance_nm": ground_distance_nm,
        "horizontal_distance_nm": horizontal_distance_nm,
        "vertical_distance_m": vertical_distance_m,
        "climb_rate_mps": climb_rate_mps,
        "flight_time_s": time_s,
        "true_airspeed_mps": true_airspeed,
    }

def bearing_frequencies(shaft_hz: float, geometry: BearingGeometry = DEFAULT_GEOMETRY) -> Dict[str, float]:
    """Return standard bearing characteristic frequencies in Hz."""
    n = geometry.rolling_elements
    bd = geometry.ball_diameter_mm
    pd = geometry.pitch_diameter_mm
    ca = math.radians(geometry.contact_angle_deg)
    ratio = (bd / pd) * math.cos(ca)
    return {
        "shaft_1x_hz": shaft_hz,
        "ftf_hz": 0.5 * shaft_hz * (1.0 - ratio),
        "bpfo_hz": 0.5 * n * shaft_hz * (1.0 - ratio),
        "bpfi_hz": 0.5 * n * shaft_hz * (1.0 + ratio),
        "bsf_hz": (pd / (2.0 * bd)) * shaft_hz * (1.0 - ratio ** 2),
    }


def _degradation_proxy(row: pd.Series, group: pd.DataFrame | None = None) -> float:
    """Estimate bearing-severity from cycle telemetry without using RUL labels."""
    nc = float(row.get("Nc", 9060.0))
    t30 = float(row.get("T30", 1300.0))
    p30 = float(row.get("P30", 100.0))
    base = 0.35 * _safe_scale(t30, 1150.0, 1550.0) + 0.25 * _safe_scale(p30, 80.0, 120.0)
    speed_stress = 0.20 * _safe_scale(nc, 9040.0, 9200.0)
    mach_stress = 0.20 * _safe_scale(float(row.get("setting_2", 0.0)), -0.0002, 0.0005)
    severity = float(np.clip(base + speed_stress + mach_stress, 0.0, 1.0))
    if group is not None and len(group) > 1:
        # Add a monotonic-with-cycle component so the synthetic vibration follows
        # the same degradation direction as the existing physics engine.
        c = float(row["cycle"])
        cmax = max(float(group["cycle"].max()), c + 1.0)
        severity = float(np.clip(0.55 * severity + 0.45 * (c / cmax), 0.0, 1.0))
    return severity


def synthesize_cycle_waveform(
    row: pd.Series,
    sample_rate_hz: int = 2048,
    duration_s: float | None = 0.5,
    seed: int = 42,
    geometry: BearingGeometry = DEFAULT_GEOMETRY,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Convert one cycle into a deterministic, physics-informed vibration waveform."""
    rng = np.random.default_rng(seed + int(row.get("engine_id", 0)) * 100003 + int(row.get("cycle", 0)))
    state = derive_operating_state(row)
    timing = estimate_cycle_timing(row)
    actual_duration_s = float(duration_s) if duration_s is not None else timing["cycle_duration_s"]
    shaft_hz = state["core_rpm"] / 60.0
    freqs = bearing_frequencies(shaft_hz, geometry)
    severity = _degradation_proxy(row)

    n = max(256, int(sample_rate_hz * actual_duration_s))
    t = np.arange(n, dtype=float) / sample_rate_hz
    x = np.zeros(n, dtype=float)

    # Operating-condition components: shaft order + fan/order sidebands.
    x += (0.08 + 0.12 * state["load_factor"]) * np.sin(2 * np.pi * freqs["shaft_1x_hz"] * t)
    x += 0.045 * np.sin(2 * np.pi * 2.0 * freqs["shaft_1x_hz"] * t + 0.4)
    x += 0.025 * np.sin(2 * np.pi * freqs["ftf_hz"] * t)

    # Fault energy grows with severity. Impulses are resonantly excited near 3 kHz.
    fault_mix = {
        "bpfo": 0.015 + 0.18 * severity,
        "bpfi": 0.010 + 0.14 * severity,
        "bsf": 0.008 + 0.10 * severity,
    }
    resonance_hz = 2800.0
    for name, amp in fault_mix.items():
        f = freqs[name + "_hz"]
        periodic = np.sin(2 * np.pi * f * t)
        carrier = np.sin(2 * np.pi * resonance_hz * t)
        x += amp * periodic * carrier
        # Sparse impact train at the characteristic frequency.
        phase = rng.random()
        impact_phase = (f * t + phase) % 1.0
        impact = np.exp(-((impact_phase) / 0.018) ** 2)
        x += amp * 0.65 * impact * np.sin(2 * np.pi * resonance_hz * t)

    # Broadband noise increases slightly with aerodynamic/mechanical load and fault severity.
    noise_std = 0.012 + 0.025 * state["load_factor"] + 0.035 * severity
    x += rng.normal(0.0, noise_std, size=n)

    # Mild deterministic amplitude modulation from Mach/dynamic pressure.
    modulation = 1.0 + 0.12 * np.sin(2 * math.pi * 0.8 * t) + 0.10 * (state["mach"] - 0.6)
    x *= modulation

    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    crest = float(peak / max(rms, 1e-12))
    kurt = float(pd.Series(x).kurtosis())

    wave = pd.DataFrame({
        "engine_id": int(row["engine_id"]),
        "cycle": int(row["cycle"]),
        "time_s": t,
        "vibration_g": x,
    })
    meta = {
        **state,
        **timing,
        **freqs,
        "bearing_severity": severity,
        "rms_g": rms,
        "peak_g": peak,
        "crest_factor": crest,
        "kurtosis": kurt,
        "sample_rate_hz": float(sample_rate_hz),
        "duration_s": float(actual_duration_s),
    }
    return wave, meta


def summarize_cycle(row: pd.Series, sample_rate_hz: int = 2048, duration_s: float | None = 0.5, seed: int = 42, previous_row: pd.Series | None = None) -> Dict[str, float]:
    _, meta = synthesize_cycle_waveform(row, sample_rate_hz, duration_s, seed)
    timing = estimate_cycle_timing(row, previous_row)
    return {"engine_id": int(row["engine_id"]), "cycle": int(row["cycle"]), **meta, **timing}


def build_bearing_summary(telemetry: pd.DataFrame, sample_rate_hz: int = 2048, duration_s: float | None = 0.5, seed: int = 42) -> pd.DataFrame:
    """Generate one vibration-analysis record per engine/cycle, including variable flight timing."""
    records = []
    ordered = telemetry.sort_values(["engine_id", "cycle"]).copy()
    for eid, group in ordered.groupby("engine_id", sort=False):
        previous = None
        cumulative = 0.0
        for _, row in group.iterrows():
            rec = summarize_cycle(row, sample_rate_hz, duration_s, seed, previous_row=previous)
            cumulative += float(rec["cycle_duration_s"])
            rec["flight_time_s_cumulative"] = cumulative
            rec["flight_time_min_cumulative"] = cumulative / 60.0
            records.append(rec)
            previous = row
    return pd.DataFrame(records)


def bearing_status(row: pd.Series) -> str:
    """Classify vibration condition using relative severity and waveform indicators."""
    severity = float(row.get("bearing_severity", 0.0))
    crest = float(row.get("crest_factor", 0.0))
    kurt = float(row.get("kurtosis", 0.0))
    if severity >= 0.88 or crest >= 7.0 or kurt >= 8.0:
        return "CRITICAL"
    if severity >= 0.72 or crest >= 5.0 or kurt >= 5.0:
        return "WARNING"
    if severity >= 0.55 or crest >= 3.8 or kurt >= 3.0:
        return "WATCH"
    return "NORMAL"


def add_bearing_status(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["bearing_status"] = out.apply(bearing_status, axis=1)
    out["bearing_risk_score"] = np.clip(
        100.0 * (0.65 * out["bearing_severity"] + 0.20 * np.clip((out["crest_factor"] - 3.0) / 5.0, 0, 1) + 0.15 * np.clip((out["kurtosis"] - 3.0) / 8.0, 0, 1)),
        0, 100,
    )
    return out


def vibration_spectrum(waveform: pd.DataFrame, sample_rate_hz: int = 2048, max_hz: float = 1200.0) -> pd.DataFrame:
    """Compute a one-sided amplitude spectrum for the generated accelerometer signal."""
    x = waveform["vibration_g"].to_numpy(dtype=float)
    x = x - np.mean(x)
    window = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * window)) * (2.0 / max(np.sum(window), 1e-12))
    freq = np.fft.rfftfreq(len(x), d=1.0 / sample_rate_hz)
    mask = freq <= max_hz
    return pd.DataFrame({"frequency_hz": freq[mask], "amplitude_g": spec[mask]})
