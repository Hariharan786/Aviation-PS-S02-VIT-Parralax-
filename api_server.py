"""
AeroGuard Real-Time Telemetry API Server
-----------------------------------------
Run with:
    python api_server.py

Endpoints:
  POST /generate        – start generating telemetry (streaming to internal buffer)
  POST /reset           – clear all accumulated telemetry and stop streaming
  GET  /status          – return stream state and row count
  GET  /telemetry       – return all accumulated telemetry as C-MAPSS TXT
  GET  /stream          – Server-Sent Events: push one new row every <interval> seconds
"""

from __future__ import annotations

import io
import sys
import time
import asyncio
import threading
from collections import deque
from pathlib import Path
from typing import Generator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ---------------------------------------------------------------------------
# Project root = directory containing this script
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from generate_fd001_physics_telemetry import (
    COLUMNS,
    physics_state,
    physics_to_fd001_settings,
    clamp,
    scale,
)
import random, math
import numpy as np
import pandas as pd

from model.bearing_vibration import build_bearing_summary, add_bearing_status, synthesize_cycle_waveform

# ---------------------------------------------------------------------------
# Shared state (thread-safe via a lock)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_rows: deque[dict] = deque()          # all generated rows so far
_streaming = False                    # True while the background thread is active
_total_cycles_generated = 0

# Pre-loaded training stats (loaded once when /generate is first called)
_train_stats: dict | None = None

# ---------------------------------------------------------------------------
# Training data loader (for baseline statistics)
# ---------------------------------------------------------------------------

def _load_train_stats(train_file: Path) -> dict:
    """Load FD001 training stats (ranges, means, stds) once."""
    train = pd.read_csv(
        train_file, sep=r"\s+", header=None, names=COLUMNS, engine="python"
    )
    sensor_cols = COLUMNS[5:]
    ranges = {c: (float(train[c].min()), float(train[c].max())) for c in sensor_cols}
    means  = train[sensor_cols].mean()
    stds   = train[sensor_cols].std().replace(0, 1e-6)
    # Also keep the full training DataFrame for nearest-neighbour sampling
    return {"ranges": ranges, "means": means, "stds": stds, "train": train}


# ---------------------------------------------------------------------------
# Row generator (yields one row at a time)
# ---------------------------------------------------------------------------

def _iter_rows(
    num_engines: int,
    cycles_per_engine: int,
    seed: int,
    stats: dict,
) -> Generator[dict, None, None]:
    """Yield one telemetry row (dict) at a time."""
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    train      = stats["train"]
    ranges     = stats["ranges"]
    stds       = stats["stds"]
    sensor_cols = COLUMNS[5:]

    for engine_id in range(1, num_engines + 1):
        severity = rng.uniform(0.0, 1.0)
        for cycle in range(1, cycles_per_engine + 1):
            progress    = (cycle - 1) / max(1, cycles_per_engine - 1)
            degradation = severity * progress ** 2

            state    = physics_state(cycle, cycles_per_engine, rng, degradation)
            settings = physics_to_fd001_settings(state)

            target_s1 = settings["setting_1"]
            target_s2 = settings["setting_2"]
            distance  = (
                ((train["setting_1"] - target_s1) / 0.0087) ** 2
                + ((train["setting_2"] - target_s2) / 0.0006) ** 2
            )
            nearest_idx = np.argsort(distance.to_numpy())[:100]
            base = train.iloc[int(rng.choice(nearest_idx))].copy()

            row: dict = {c: float(base[c]) for c in COLUMNS[2:]}
            row["engine_id"] = engine_id
            row["cycle"]     = cycle

            # Physics-derived operating conditions
            row["setting_1"] = settings["setting_1"]
            row["setting_2"] = settings["setting_2"]
            row["setting_3"] = 100.0

            def perturb(sensor, signal, source_lo, source_hi, sensitivity=0.25, sign=1.0):
                normalized = (signal - source_lo) / max(source_hi - source_lo, 1e-9)
                centered   = normalized - 0.5
                value      = (
                    row[sensor]
                    + sign * centered * sensitivity * float(stds[sensor])
                    + np_rng.normal(0, 0.025 * float(stds[sensor]))
                )
                return clamp(value, ranges[sensor][0], ranges[sensor][1])

            row["T24"]  = perturb("T24",  state["t_inlet"],  400,  540, 0.35)
            row["T30"]  = perturb("T30",  state["t_hpc"],    900, 1500, 0.40)
            row["T50"]  = perturb("T50",  state["t_comb"],  2300, 3300, 0.45)
            row["P30"]  = perturb("P30",  state["p_hpc"],     50,  100, 0.40)
            row["Ps30"] = perturb("Ps30", state["p_inlet"],    3,   15, 0.35)
            row["Nc"]   = perturb("Nc",   state["core_rpm"], 9000,10000, 0.30)
            row["NRc"]  = perturb("NRc",  state["core_rpm"], 8000, 9000, 0.30)
            row["phi"]  = perturb("phi",  state["fuel_ratio"], 200, 230, 0.30)

            # Degradation signature
            row["T30"]  = clamp(row["T30"]  + degradation * 0.30 * float(stds["T30"]),  *ranges["T30"])
            row["T50"]  = clamp(row["T50"]  + degradation * 0.35 * float(stds["T50"]),  *ranges["T50"])
            row["P30"]  = clamp(row["P30"]  + degradation * 0.25 * float(stds["P30"]),  *ranges["P30"])
            row["Ps30"] = clamp(row["Ps30"] + degradation * 0.20 * float(stds["Ps30"]), *ranges["Ps30"])
            row["Nc"]   = clamp(row["Nc"]   - degradation * 0.25 * float(stds["Nc"]),   *ranges["Nc"])
            row["NRc"]  = clamp(row["NRc"]  - degradation * 0.25 * float(stds["NRc"]),  *ranges["NRc"])

            for sensor in sensor_cols:
                if sensor in {"T24","T30","T50","P30","Ps30","Nc","NRc","phi"}:
                    continue
                row[sensor] = clamp(
                    row[sensor] + np_rng.normal(0, 0.015 * float(stds[sensor])),
                    ranges[sensor][0], ranges[sensor][1],
                )

            yield row


# ---------------------------------------------------------------------------
# Background streaming thread
# ---------------------------------------------------------------------------

def _stream_worker(
    num_engines: int,
    cycles_per_engine: int,
    seed: int,
    interval: float,
):
    global _streaming, _total_cycles_generated
    stats = _train_stats

    for row in _iter_rows(num_engines, cycles_per_engine, seed, stats):
        with _lock:
            if not _streaming:
                break
            _rows.append(row)
            _total_cycles_generated += 1
        time.sleep(interval)

    with _lock:
        _streaming = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AeroGuard Telemetry API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate")
def start_generate(
    num_engines: int = 10,
    cycles_per_engine: int = 50,
    seed: int = 42,
    interval: float = 0.5,
):
    """Start streaming telemetry into the buffer (non-blocking)."""
    global _streaming, _train_stats, _total_cycles_generated

    train_file = ROOT / "train_FD001.txt"
    if not train_file.exists():
        return JSONResponse(
            {"error": f"train_FD001.txt not found at {ROOT}"}, status_code=500
        )

    with _lock:
        if _streaming:
            return {"status": "already_running", "rows": len(_rows)}
        _streaming = True
        if _train_stats is None:
            _train_stats = _load_train_stats(train_file)

    t = threading.Thread(
        target=_stream_worker,
        args=(num_engines, cycles_per_engine, seed, interval),
        daemon=True,
    )
    t.start()
    return {"status": "started", "num_engines": num_engines, "cycles_per_engine": cycles_per_engine}


@app.post("/reset")
def reset():
    """Stop streaming and clear all accumulated telemetry."""
    global _streaming, _total_cycles_generated
    with _lock:
        _streaming = False
        _rows.clear()
        _total_cycles_generated = 0
    return {"status": "reset"}


@app.get("/status")
def get_status():
    with _lock:
        return {
            "streaming": _streaming,
            "rows_accumulated": len(_rows),
            "total_cycles_generated": _total_cycles_generated,
            "bearing_analysis_available": len(_rows) > 0,
        }


@app.get("/telemetry")
def get_telemetry():
    """Return all accumulated telemetry rows as a C-MAPSS-format TXT."""
    with _lock:
        snapshot = list(_rows)

    buf = io.StringIO()
    for row in snapshot:
        parts = []
        for c in COLUMNS:
            if c in ("engine_id", "cycle"):
                parts.append(str(int(row[c])))
            else:
                parts.append(f"{row[c]:.6f}")
        buf.write(" ".join(parts) + "\n")

    txt = buf.getvalue()
    return StreamingResponse(
        io.BytesIO(txt.encode()),
        media_type="text/plain",
        headers={"Content-Disposition": "inline; filename=live_telemetry.txt"},
    )


@app.get("/bearing_summary")
def get_bearing_summary():
    """Return cycle-level bearing vibration features derived from live telemetry."""
    with _lock:
        snapshot = list(_rows)
    if not snapshot:
        return JSONResponse({"rows": [], "message": "No telemetry generated yet."})
    df = pd.DataFrame(snapshot)
    summary = add_bearing_status(build_bearing_summary(df, sample_rate_hz=2048, duration_s=0.5, seed=42))
    return summary.to_dict(orient="records")


@app.get("/bearing_waveform")
def get_bearing_waveform(engine_id: int = 1, cycle: int | None = None):
    """Return a waveform for one live engine/cycle as JSON."""
    with _lock:
        snapshot = list(_rows)
    if not snapshot:
        return JSONResponse({"rows": [], "message": "No telemetry generated yet."})
    df = pd.DataFrame(snapshot)
    df = df[df["engine_id"].astype(int) == int(engine_id)].sort_values("cycle")
    if df.empty:
        return JSONResponse({"rows": [], "message": f"Engine {engine_id} not found."}, status_code=404)
    row = df.iloc[-1] if cycle is None else df[df["cycle"].astype(int) == int(cycle)].iloc[0]
    wave, meta = synthesize_cycle_waveform(row, 2048, 0.5, 42)
    return {"meta": meta, "waveform": wave.to_dict(orient="records")}


@app.get("/stream")
async def sse_stream():
    """Server-Sent Events: push accumulated row count every second."""
    async def event_generator():
        last_sent = 0
        while True:
            with _lock:
                current = len(_rows)
                streaming = _streaming
            if current != last_sent:
                last_sent = current
                yield f"data: {{\"rows\": {current}, \"streaming\": {str(streaming).lower()}}}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
