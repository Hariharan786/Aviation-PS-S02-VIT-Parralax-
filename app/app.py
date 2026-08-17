import sys, json, tempfile, os, joblib, time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Try to import httpx for the Live Telemetry mode (graceful fallback)
# ---------------------------------------------------------------------------
try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from hybrid_predictor import predict_file as predict_hybrid
from condition_aware_anomaly import load, score
from bearing_vibration import build_bearing_summary, add_bearing_status, synthesize_cycle_waveform, vibration_spectrum

st.set_page_config(page_title="AeroGuard PS-S02", page_icon="✈️", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS Injection for Fonts and Purplish Hues
# ---------------------------------------------------------------------------
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900&family=Rubik:wght@400;500;700&display=swap');

/* Apply Rubik to all general text */
html, body, [class*="css"], p, div, span, label, li {
    font-family: 'Rubik', sans-serif !important;
}

/* Apply Lato to all headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Lato', sans-serif !important;
}

/* Override primary button colors to a purplish shade */
div.stButton > button[kind="primary"] {
    background-color: #6b21a8 !important; 
    border-color: #6b21a8 !important;
    color: white !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #581c87 !important;
    border-color: #581c87 !important;
}

/* General link and accent color overrides */
a {
    color: #6b21a8 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
# ---------------------------------------------------------------------------

st.title("✈️ AeroGuard — Aircraft Engine Health & Predictive Maintenance")
st.caption("PS-S02 | LSTM + Random Forest ensemble RUL + anomaly detection + sensor diagnostics")

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    dataset = st.selectbox("Dataset", DATASETS)
    st.markdown("### Data source")
    mode = st.radio(
        "Input mode",
        ["📁 Upload File", "🔴 Live Telemetry"],
        index=0,
    )

    if mode == "🔴 Live Telemetry":
        st.markdown("---")
        st.markdown("#### Live generation settings")
        live_num_engines     = st.slider("Engines", 2, 20, 10)
        live_cycles          = st.slider("Cycles / engine", 10, 100, 40)
        live_seed            = st.number_input("Random seed", value=42, step=1)
        live_interval        = st.slider("Generation interval (s)", 0.1, 3.0, 0.5, step=0.1)
        live_refresh         = st.slider("UI refresh every (s)", 3, 30, 8)
        st.markdown("---")

        col_start, col_stop = st.columns(2)
        with col_start:
            btn_start = st.button("▶ Start", type="primary", use_container_width=True)
        with col_stop:
            btn_stop  = st.button("⏹ Stop",  use_container_width=True)

        # Track streaming state in session
        if "live_streaming" not in st.session_state:
            st.session_state.live_streaming = False

        if btn_start and _HTTPX_OK:
            try:
                r = httpx.post(
                    f"{API_BASE}/generate",
                    params={
                        "num_engines":      live_num_engines,
                        "cycles_per_engine": live_cycles,
                        "seed":             int(live_seed),
                        "interval":         live_interval,
                    },
                    timeout=5,
                )
                st.session_state.live_streaming = True
                st.toast("✅ Telemetry stream started!")
            except Exception as e:
                st.error(f"Cannot reach API server: {e}\n\nMake sure `python api_server.py` is running.")

        if btn_start and not _HTTPX_OK:
            st.error("`httpx` not installed. Run:\n```\npip install httpx\n```")

        if btn_stop and _HTTPX_OK:
            try:
                httpx.post(f"{API_BASE}/reset", timeout=5)
                st.session_state.live_streaming = False
                st.toast("⏹ Stream stopped and buffer cleared.")
            except Exception:
                pass

    else:
        uploaded = st.file_uploader(
            f"Upload unseen {dataset} telemetry",
            type=["txt", "csv"],
            help="C-MAPSS telemetry only; CSV may include the 26-column header.",
        )
    st.markdown("### Monitoring pipeline")
    st.write("Telemetry → sensor normalisation → LSTM + Random Forest → anomaly detection → deterioration → maintenance")

# ---------------------------------------------------------------------------
# Model readiness check
# ---------------------------------------------------------------------------
lstm_artifact   = ROOT / "models" / dataset / "lstm_artifact.joblib"
rf_artifact     = ROOT / "models" / dataset / "rf_artifact.joblib"
anomaly_artifact = ROOT / "models" / dataset / "anomaly_artifact.joblib"
if not all(p.exists() for p in [lstm_artifact, rf_artifact, anomaly_artifact]):
    st.error(f"{dataset} is missing one or more model artifacts.")
    st.code("python train_all.py")
    st.stop()

metrics_path     = ROOT / "models" / dataset / "metrics.json"
rf_metrics_path  = ROOT / "models" / dataset / "rf_metrics.json"
anom_metrics_path = ROOT / "models" / dataset / "anomaly_metrics.json"

st.subheader(f"{dataset} model readiness")
c1, c2, c3, c4, c5 = st.columns(5)
if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    c1.metric("LSTM MAE", f"{m['regression']['MAE_cycles']:.2f}")
    c2.metric("LSTM R²",  f"{m['regression']['R2']:.3f}")
if rf_metrics_path.exists():
    rm = json.loads(rf_metrics_path.read_text())
    c3.metric("RF MAE", f"{rm['MAE_cycles']:.2f}")
    c4.metric("RF R²",  f"{rm['R2']:.3f}")
if anom_metrics_path.exists():
    am = json.loads(anom_metrics_path.read_text())
    c5.metric("Anomaly F1", f"{am['f1']:.3f}")

# ---------------------------------------------------------------------------
# COLUMNS definition used by the normaliser
# ---------------------------------------------------------------------------
COLUMNS = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30", "Nf", "Nc", "epr",
    "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
    "Nf_dmd", "PCNfR_dmd", "W31", "W32",
]

def normalize_uploaded_telemetry(uploaded_file):
    raw = uploaded_file.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as f:
        f.write(raw); src = f.name
    try:
        with open(src, "r", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(4096)
        first  = sample.splitlines()[0] if sample.splitlines() else ""
        header = any(x in first.lower() for x in ["engine_id", "setting_1", "cycle"])
        if "," in first:
            d = pd.read_csv(src, sep=",",      header=0 if header else None, names=None if header else COLUMNS, engine="python")
        elif "\t" in first:
            d = pd.read_csv(src, sep="\t",     header=0 if header else None, names=None if header else COLUMNS, engine="python")
        else:
            d = pd.read_csv(src, sep=r"\s+",   header=None, names=COLUMNS, engine="python")
        if d.shape[1] != 26:
            raise ValueError(f"Telemetry must contain exactly 26 C-MAPSS columns; received {d.shape[1]}.")
        d.columns = COLUMNS
        for c in COLUMNS:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        if d[COLUMNS].isna().any().any():
            raise ValueError("Telemetry contains missing or non-numeric values.")
        d.engine_id = d.engine_id.astype(int)
        d.cycle     = d.cycle.astype(int)
        out = tempfile.NamedTemporaryFile(delete=False, suffix="_normalized.txt", mode="w", newline="")
        for _, r in d.iterrows():
            out.write(" ".join(
                str(int(r[c])) if c in ["engine_id", "cycle"] else f"{float(r[c]):.12g}"
                for c in COLUMNS
            ) + "\n")
        out.close()
        return out.name
    finally:
        try:
            os.unlink(src)
        except OSError:
            pass


def txt_from_raw_text(raw_text: str) -> str:
    """Write raw C-MAPSS text to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix="_live.txt", mode="w", newline="") as f:
        f.write(raw_text)
        return f.name


# ---------------------------------------------------------------------------
# Bearing vibration analysis
# ---------------------------------------------------------------------------
def render_bearing_analysis(telemetry_df, title_prefix=""):
    if telemetry_df is None or telemetry_df.empty:
        return
    try:
        bearing = add_bearing_status(build_bearing_summary(telemetry_df, sample_rate_hz=2048, duration_s=0.5, seed=42))
    except Exception as exc:
        st.warning(f"Bearing vibration analysis unavailable: {exc}")
        return

    st.subheader("🛞 Bearing Vibration Analysis")
    st.caption("Physics-informed synthetic accelerometer signal. Bearing vibration now feeds the integrated engine health/RUL decision. Mission timing is estimated separately from Mach, altitude, throttle/load and altitude change; the waveform below is a short diagnostic capture, not the whole mission timeline.")

    latest = bearing.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1).copy()
    critical = int((latest["bearing_status"] == "CRITICAL").sum())
    warning = int(latest["bearing_status"].isin(["WARNING", "WATCH"]).sum())
    avg_risk = float(latest["bearing_risk_score"].mean())
    q1, q2, q3 = st.columns(3)
    q1.metric("Bearing risk", f"{avg_risk:.1f}/100")
    q2.metric("Warning / watch", warning)
    q3.metric("Critical", critical)

    st.dataframe(latest[["engine_id", "cycle", "mach", "altitude_ft", "throttle_proxy_pct", "airspeed_mps", "core_rpm", "cycle_duration_s", "flight_time_min_cumulative", "bpfo_hz", "bpfi_hz", "bsf_hz", "rms_g", "crest_factor", "kurtosis", "bearing_risk_score", "bearing_status"]], use_container_width=True, hide_index=True)

    engines = sorted(bearing.engine_id.unique())
    selected_engine = st.selectbox("Bearing waveform engine", engines, key=f"bearing_engine_{title_prefix}")
    cycles = sorted(bearing.loc[bearing.engine_id == selected_engine, "cycle"].astype(int).unique())
    selected_cycle = st.selectbox("Bearing waveform cycle", cycles, index=len(cycles)-1, key=f"bearing_cycle_{title_prefix}")
    row = telemetry_df[(telemetry_df.engine_id.astype(int) == int(selected_engine)) & (telemetry_df.cycle.astype(int) == int(selected_cycle))].iloc[0]
    waveform, meta = synthesize_cycle_waveform(row, sample_rate_hz=2048, duration_s=0.5, seed=42)

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Mach", f"{meta['mach']:.3f}")
    w2.metric("Core speed", f"{meta['core_rpm']:.0f} rpm")
    w3.metric("RMS", f"{meta['rms_g']:.3f} g")
    w4.metric("Crest factor", f"{meta['crest_factor']:.2f}")

    st.line_chart(waveform.set_index("time_s")["vibration_g"], height=260)
    spectrum = vibration_spectrum(waveform, 2048, 1200.0)
    st.line_chart(spectrum.set_index("frequency_hz")["amplitude_g"], height=240)
    st.caption(f"Expected bearing frequencies: FTF {meta['ftf_hz']:.1f} Hz · BPFO {meta['bpfo_hz']:.1f} Hz · BPFI {meta['bpfi_hz']:.1f} Hz · BSF {meta['bsf_hz']:.1f} Hz. Elevated energy near these characteristic frequencies is used as a bearing-fault indicator.")

    st.download_button(
        "Download bearing cycle summary",
        bearing.to_csv(index=False).encode("utf-8"),
        "bearing_vibration_summary.csv",
        "text/csv",
        key=f"bearing_summary_{title_prefix}",
    )
    st.download_button(
        "Download selected bearing waveform",
        waveform.to_csv(index=False).encode("utf-8"),
        f"bearing_engine_{selected_engine}_cycle_{selected_cycle}_waveform.csv",
        "text/csv",
        key=f"bearing_wave_{title_prefix}",
    )

# ---------------------------------------------------------------------------
# ── LIVE TELEMETRY MODE ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
if mode == "🔴 Live Telemetry":
    if not _HTTPX_OK:
        st.error("`httpx` not installed. Run `pip install httpx` and restart Streamlit.")
        st.stop()

    # Status banner
    status_placeholder = st.empty()
    try:
        status_resp = httpx.get(f"{API_BASE}/status", timeout=3)
        server_status = status_resp.json()
        rows_ready = server_status.get("rows_accumulated", 0)
        is_live    = server_status.get("streaming", False)
    except Exception:
        st.warning(
            "⚠️  Cannot reach the API server at `http://localhost:8000`.\n\n"
            "Start it first:\n```\npython api_server.py\n```"
        )
        st.stop()

    if is_live:
        status_placeholder.success(f"🔴 **LIVE** — {rows_ready} rows accumulated and counting…")
    elif rows_ready > 0:
        status_placeholder.info(f"⏸ **Paused** — {rows_ready} rows in buffer (press ▶ Start to resume).")
    else:
        status_placeholder.warning("▶ Press **Start** in the sidebar to begin live telemetry generation.")

    if rows_ready < 30:
        min_rows_needed = 30
        st.info(
            f"⏳ Waiting for at least {min_rows_needed} rows before running the pipeline "
            f"(currently {rows_ready}). The UI will auto-refresh every {live_refresh}s."
        )
        # Auto-refresh and rerun while waiting
        time.sleep(live_refresh)
        st.rerun()

    # Pull the accumulated telemetry from the API
    try:
        telem_resp = httpx.get(f"{API_BASE}/telemetry", timeout=10)
        raw_txt    = telem_resp.text
    except Exception as e:
        st.error(f"Failed to fetch telemetry: {e}")
        st.stop()

    if not raw_txt.strip():
        st.warning("No telemetry data yet. Waiting for the stream…")
        time.sleep(live_refresh)
        st.rerun()

    telemetry_path = txt_from_raw_text(raw_txt)

    try:
        fleet_rul, ensemble_meta = predict_hybrid(telemetry_path, dataset, ROOT)
        raw_anom = load(telemetry_path)
        anom_art = joblib.load(anomaly_artifact)
        anom     = score(raw_anom, anom_art)
    except Exception as e:
        st.error(f"Could not process telemetry: {e}")
        st.exception(e)
        try:
            os.unlink(telemetry_path)
        except OSError:
            pass
        st.stop()

    try:
        os.unlink(telemetry_path)
    except OSError:
        pass

    latest_anom = anom.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
    latest_rul  = fleet_rul.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
    fleet = latest_rul.merge(
        latest_anom[["engine_id", "cycle", "anomaly_score", "abnormal_condition",
                     "abnormal_sensor_count", "abnormal_sensors", "trend_status"]],
        on=["engine_id", "cycle"], how="left",
    )
    # predict_hybrid already integrated bearing health, variable flight time,
    # composite health, and bearing-aware maintenance recommendation.

    def combined(row):
        return str(row.get("overall_status", row.get("ensemble_status", "HEALTHY")))

    fleet["combined_status"] = fleet.apply(combined, axis=1)
    critical = int((fleet.combined_status == "CRITICAL").sum())
    warning  = int(fleet.combined_status.str.contains("WARNING").sum())
    abnormal = int((fleet.abnormal_condition != "NORMAL").sum())

    a, b, c, d, e = st.columns(5)
    a.metric("Engines",          len(fleet))
    b.metric("Integrated Avg RUL", f"{fleet.overall_RUL_cycles.mean():.1f} cycles")
    c.metric("Abnormal",         abnormal)
    d.metric("Warning",          warning)
    e.metric("Critical",         critical)
    st.caption(
        f"Ensemble weights: LSTM {ensemble_meta['lstm_weight']:.1%} · "
        f"Random Forest {ensemble_meta['rf_weight']:.1%}. "
        f"Weights are based on inverse validation MAE.  |  "
        f"⟳ Last refreshed with **{rows_ready}** telemetry rows."
    )

    st.divider()
    st.subheader("🚦 Current Engine Health")
    st.dataframe(
        fleet[[
            "engine_id", "cycle", "overall_RUL_cycles", "ensemble_RUL_cycles", "overall_health_score",
            "bearing_risk_score", "bearing_status", "cycle_duration_s", "flight_time_min_cumulative",
            "lstm_RUL_cycles", "rf_RUL_cycles", "anomaly_score", "abnormal_condition",
            "abnormal_sensor_count", "abnormal_sensors", "trend_status", "combined_status",
            "overall_maintenance_recommendation", "maintenance_signal",
        ]],
        use_container_width=True, hide_index=True,
    )

    st.subheader("📉 Deterioration Trends")
    selected = st.selectbox("Engine trend", sorted(anom.engine_id.unique()))
    trend  = anom[anom.engine_id == selected].sort_values("cycle").copy()
    ruleng = fleet_rul[fleet_rul.engine_id == selected].sort_values("cycle")
    st.line_chart(trend.set_index("cycle")[["anomaly_score", "max_sensor_z"]])
    st.line_chart(ruleng.set_index("cycle")[["overall_RUL_cycles", "overall_health_score"]])
    st.write(
        f"Engine {selected}: anomaly **{trend.anomaly_score.iloc[-1]:.3f}**, "
        f"trend **{trend.trend_status.iloc[-1]}**, "
        f"abnormal sensors **{trend.abnormal_sensors.iloc[-1]}**"
    )

    st.subheader("📡 Sensor Performance & Abnormality")
    sp1 = ROOT / "models" / dataset / "sensor_performance.csv"
    sp2 = ROOT / "models" / dataset / "anomaly_sensor_performance.csv"
    s1  = pd.read_csv(sp1) if sp1.exists() else pd.DataFrame()
    s2  = pd.read_csv(sp2) if sp2.exists() else pd.DataFrame()
    if not s1.empty: s1 = s1.rename(columns={"spearman_r": "rul_spearman"})
    if not s2.empty: s2 = s2.rename(columns={"spearman_r": "anomaly_rul_spearman"})
    sp = (
        s1[["sensor", "rul_spearman", "absolute_spearman", "signal"]]
        .merge(s2[["sensor", "anomaly_rul_spearman"]], on="sensor", how="outer")
        if not s1.empty and not s2.empty else s1
    )
    st.dataframe(sp, use_container_width=True, hide_index=True)
    if not sp.empty and "absolute_spearman" in sp:
        st.bar_chart(sp.sort_values("absolute_spearman", ascending=False).set_index("sensor")[["absolute_spearman"]])

    st.subheader("🛠 Maintenance Recommendation")
    st.dataframe(
        fleet.sort_values(["overall_RUL_cycles", "bearing_risk_score", "anomaly_score"])[[
            "engine_id", "overall_RUL_cycles", "overall_health_score", "bearing_risk_score",
            "bearing_status", "flight_time_min_cumulative", "anomaly_score", "abnormal_sensors",
            "trend_status", "combined_status", "overall_maintenance_recommendation",
        ]],
        use_container_width=True, hide_index=True,
    )

    render_bearing_analysis(raw_anom, "live")

    def maintenance_priority(row):
        status = str(row["combined_status"]); signal = str(row.get("overall_maintenance_recommendation", row.get("maintenance_signal", "")))
        if status == "CRITICAL" or signal.startswith("URGENT"): return "URGENT"
        if "WARNING" in status or status in ["ANOMALY", "DEGRADING"] or "INSPECTION" in signal.upper(): return "ATTENTION"
        return "ROUTINE"

    # upload bearing analysis is rendered after the upload pipeline below

    def health_word(status):
        return {
            "HEALTHY":          "healthy",
            "DEGRADING":        "showing signs of degradation",
            "WARNING":          "requiring attention",
            "WARNING + ANOMALY": "showing degradation and abnormal telemetry",
            "ANOMALY":          "showing abnormal telemetry",
            "CRITICAL":         "in a critical condition",
        }.get(str(status), str(status).lower())

    def engine_verbal_summary(row):
        eid = int(row["engine_id"]); rul = float(row.get("overall_RUL_cycles", row["ensemble_RUL_cycles"]))
        health = float(row.get("overall_health_score", row["ensemble_health_score"])); anomaly = float(row["anomaly_score"])
        bearing = float(row.get("bearing_risk_score", 0.0)); bstatus = str(row.get("bearing_status", "NORMAL"))
        flight_min = float(row.get("flight_time_min_cumulative", 0.0))
        status = health_word(row["combined_status"]); trend = str(row["trend_status"]).lower()
        sensors = str(row["abnormal_sensors"])
        sensor_text = "No individual sensor is currently flagged as abnormal." if sensors in ["", "nan", "None", "[]"] else f"The main sensors flagged are {sensors}."
        priority = maintenance_priority(row)
        action = str(row.get("overall_maintenance_recommendation", "Continue routine monitoring."))
        return (
            f"Engine {eid} is {status}. Its integrated remaining useful life is about {rul:.0f} cycles, "
            f"with an overall health score of {health:.0f} percent. "
            f"Bearing vibration risk is {bearing:.0f}/100 ({bstatus}), with about {flight_min:.1f} minutes of estimated flight time accumulated in the observed mission. "
            f"The current anomaly score is {anomaly:.2f} and the deterioration trend is {trend}. "
            f"{sensor_text} Recommendation: {action}."
        )

    def fleet_verbal_summary(fleet_df, dataset_name):
        total = len(fleet_df); crit = int((fleet_df["combined_status"] == "CRITICAL").sum())
        attn  = int(fleet_df["combined_status"].isin(["WARNING","WARNING + ANOMALY","ANOMALY","DEGRADING"]).sum())
        healthy = total - crit - attn
        avg_rul = float(fleet_df.get("overall_RUL_cycles", fleet_df["ensemble_RUL_cycles"]).mean()); avg_health = float(fleet_df.get("overall_health_score", fleet_df["ensemble_health_score"]).mean())
        lines = [
            f"{dataset_name} fleet health briefing.",
            f"The system evaluated {total} engine{'s' if total != 1 else ''}.",
            f"On average, the engines have an estimated remaining useful life of about {avg_rul:.0f} cycles and an average health score of {avg_health:.0f} percent.",
        ]
        lines.append(f"{crit} engine{'s' if crit != 1 else ''} require urgent attention because the combined health and anomaly assessment is critical." if crit else "No engine is currently classified as critical.")
        lines.append(f"{attn} engine{'s' if attn != 1 else ''} require additional attention because they are degrading, showing abnormal telemetry, or carrying a warning." if attn else "No additional engines are currently flagged for elevated attention.")
        if healthy:
            lines.append(f"{healthy} engine{'s' if healthy != 1 else ''} are currently suitable for routine monitoring.")
        lines.append("These recommendations are model-based monitoring guidance and should be confirmed by qualified engineering and maintenance procedures.")
        return " ".join(lines)

    fleet_brief   = fleet_verbal_summary(fleet, dataset)
    engine_briefs = [engine_verbal_summary(row) for _, row in fleet.sort_values("engine_id").iterrows()]
    full_brief    = fleet_brief + "\n\nEngine-by-engine summary:\n" + "\n".join(f"- {x}" for x in engine_briefs)

    st.subheader("🗣️ Plain-English Engine Health Briefing")
    st.info(fleet_brief)
    with st.expander("Read the engine-by-engine maintenance briefing"):
        for brief in engine_briefs:
            st.write("• " + brief)

    st.download_button(
        "Download verbal maintenance briefing", full_brief.encode("utf-8"),
        f"{dataset}_PS_S02_verbal_maintenance_brief.txt", "text/plain",
    )
    st.download_button(
        "Download combined health/anomaly report",
        fleet.to_csv(index=False).encode(),
        f"{dataset}_PS_S02_hybrid_health_report.csv", "text/csv",
    )

    # ── Auto-refresh while live ────────────────────────────────────────────
    if is_live:
        time.sleep(live_refresh)
        st.rerun()

    st.stop()   # Do not fall through to Upload File section


# ---------------------------------------------------------------------------
# ── UPLOAD FILE MODE (original behaviour, unchanged) ────────────────────────
# ---------------------------------------------------------------------------
if uploaded is None:
    st.info("Upload telemetry to run the full PS-S02 monitoring workflow.")
    if anom_metrics_path.exists():
        am = json.loads(anom_metrics_path.read_text())
        a, b, c, d = st.columns(4)
        a.metric("Precision", f"{am['precision']:.3f}")
        b.metric("Recall",    f"{am['recall']:.3f}")
        c.metric("F1",        f"{am['f1']:.3f}")
        d.metric(
            "Detection delay",
            f"{am['mean_detection_delay_cycles']:.2f} cycles"
            if am["mean_detection_delay_cycles"] is not None else "N/A",
        )
    st.stop()

telemetry_path = normalize_uploaded_telemetry(uploaded)
try:
    fleet_rul, ensemble_meta = predict_hybrid(telemetry_path, dataset, ROOT)
    raw = load(telemetry_path)
    anom_art = joblib.load(anomaly_artifact)
    anom = score(raw, anom_art)
except Exception as e:
    st.error(f"Could not process telemetry: {e}"); st.exception(e); st.stop()

latest_anom = anom.sort_values(["engine_id","cycle"]).groupby("engine_id").tail(1)
latest_rul  = fleet_rul.sort_values(["engine_id","cycle"]).groupby("engine_id").tail(1)
fleet = latest_rul.merge(
    latest_anom[[
        "engine_id","cycle","anomaly_score","abnormal_condition",
        "abnormal_sensor_count","abnormal_sensors","trend_status"
    ]],
    on=["engine_id","cycle"], how="left",
)

def combined(row):
    return str(row.get("overall_status", row.get("ensemble_status", "HEALTHY")))

fleet["combined_status"] = fleet.apply(combined, axis=1)
critical = int((fleet.combined_status=="CRITICAL").sum())
warning  = int(fleet.combined_status.str.contains("WARNING").sum())
abnormal = int((fleet.abnormal_condition!="NORMAL").sum())

a,b,c,d,e = st.columns(5)
a.metric("Engines",          len(fleet))
b.metric("Ensemble Avg RUL", f"{fleet.ensemble_RUL_cycles.mean():.1f} cycles")
c.metric("Abnormal",         abnormal)
d.metric("Warning",          warning)
e.metric("Critical",         critical)
st.caption(f"Ensemble weights: LSTM {ensemble_meta['lstm_weight']:.1%} · Random Forest {ensemble_meta['rf_weight']:.1%}. Weights are based on inverse validation MAE.")

st.divider(); st.subheader("🚦 Current Engine Health")
st.dataframe(fleet[[
    "engine_id","cycle","overall_RUL_cycles","overall_health_score","bearing_risk_score","bearing_status",
    "cycle_duration_s","flight_time_min_cumulative","lstm_RUL_cycles","rf_RUL_cycles","anomaly_score","abnormal_condition",
    "abnormal_sensor_count","abnormal_sensors","trend_status","combined_status","overall_maintenance_recommendation"
]], use_container_width=True, hide_index=True)

st.subheader("📉 Deterioration Trends")
selected = st.selectbox("Engine trend", sorted(anom.engine_id.unique()))
trend  = anom[anom.engine_id==selected].sort_values("cycle").copy()
ruleng = fleet_rul[fleet_rul.engine_id==selected].sort_values("cycle")
st.line_chart(trend.set_index("cycle")[["anomaly_score","max_sensor_z"]])
st.line_chart(ruleng.set_index("cycle")[["overall_RUL_cycles","overall_health_score"]])
st.write(f"Engine {selected}: anomaly **{trend.anomaly_score.iloc[-1]:.3f}**, trend **{trend.trend_status.iloc[-1]}**, abnormal sensors **{trend.abnormal_sensors.iloc[-1]}**")

st.subheader("📡 Sensor Performance & Abnormality")
sp1=ROOT/"models"/dataset/"sensor_performance.csv"; sp2=ROOT/"models"/dataset/"anomaly_sensor_performance.csv"
s1=pd.read_csv(sp1) if sp1.exists() else pd.DataFrame(); s2=pd.read_csv(sp2) if sp2.exists() else pd.DataFrame()
if not s1.empty: s1=s1.rename(columns={"spearman_r":"rul_spearman"})
if not s2.empty: s2=s2.rename(columns={"spearman_r":"anomaly_rul_spearman"})
sp=s1[["sensor","rul_spearman","absolute_spearman","signal"]].merge(s2[["sensor","anomaly_rul_spearman"]],on="sensor",how="outer") if not s1.empty and not s2.empty else s1
st.dataframe(sp,use_container_width=True,hide_index=True)
if not sp.empty and "absolute_spearman" in sp: st.bar_chart(sp.sort_values("absolute_spearman",ascending=False).set_index("sensor")[["absolute_spearman"]])

st.subheader("📊 Abnormal-Condition Evaluation")
if anom_metrics_path.exists():
    am=json.loads(anom_metrics_path.read_text()); x1,x2,x3,x4=st.columns(4)
    x1.metric("Precision",f"{am['precision']:.3f}"); x2.metric("Recall",f"{am['recall']:.3f}")
    x3.metric("F1",f"{am['f1']:.3f}")
    x4.metric("Detection delay",f"{am['mean_detection_delay_cycles']:.2f} cycles" if am["mean_detection_delay_cycles"] is not None else "N/A")
    cm=pd.DataFrame(am["confusion_matrix"],index=["Actual Normal","Actual Abnormal"],columns=["Pred Normal","Pred Abnormal"])
    st.dataframe(cm,use_container_width=True)
    st.caption("Evaluation proxy: validation RUL ≥ 80 = normal; RUL ≤ 40 = abnormal; RUL 41–79 excluded.")

render_bearing_analysis(raw, "upload")

def health_word(status):
    return {"HEALTHY":"healthy","DEGRADING":"showing signs of degradation","WARNING":"requiring attention","WARNING + ANOMALY":"showing degradation and abnormal telemetry","ANOMALY":"showing abnormal telemetry","CRITICAL":"in a critical condition"}.get(str(status),str(status).lower())

def maintenance_priority(row):
    status=str(row["combined_status"]); signal=str(row.get("overall_maintenance_recommendation", row.get("maintenance_signal", "")))
    if status=="CRITICAL" or signal.startswith("URGENT"): return "URGENT"
    if "WARNING" in status or status in ["ANOMALY","DEGRADING"] or "INSPECTION" in signal.upper(): return "ATTENTION"
    return "ROUTINE"

def engine_verbal_summary(row):
    eid=int(row["engine_id"]); rul=float(row.get("overall_RUL_cycles",row["ensemble_RUL_cycles"])); health=float(row.get("overall_health_score",row["ensemble_health_score"])); anomaly=float(row["anomaly_score"])
    bearing=float(row.get("bearing_risk_score",0.0)); bstatus=str(row.get("bearing_status","NORMAL")); flight_min=float(row.get("flight_time_min_cumulative",0.0))
    status=health_word(row["combined_status"]); trend=str(row["trend_status"]).lower(); sensors=str(row["abnormal_sensors"])
    sensor_text="No individual sensor is currently flagged as abnormal." if sensors in ["","nan","None","[]"] else f"The main sensors flagged are {sensors}."
    action=str(row.get("overall_maintenance_recommendation","Continue routine monitoring."))
    return (f"Engine {eid} is {status}. Its integrated remaining useful life is about {rul:.0f} cycles, with an overall health score of {health:.0f} percent. Bearing vibration risk is {bearing:.0f}/100 ({bstatus}) and the estimated observed flight time is {flight_min:.1f} minutes. The current anomaly score is {anomaly:.2f} and the deterioration trend is {trend}. {sensor_text} Recommendation: {action}.")

def fleet_verbal_summary(fleet_df,dataset_name):
    total=len(fleet_df); crit=int((fleet_df["combined_status"]=="CRITICAL").sum()); attn=int(fleet_df["combined_status"].isin(["WARNING","WARNING + ANOMALY","ANOMALY","DEGRADING"]).sum()); healthy=total-crit-attn
    avg_rul=float(fleet_df.get("overall_RUL_cycles",fleet_df["ensemble_RUL_cycles"]).mean()); avg_health=float(fleet_df.get("overall_health_score",fleet_df["ensemble_health_score"]).mean())
    lines=[f"{dataset_name} fleet health briefing.",f"The system evaluated {total} engine{'s' if total!=1 else ''}.",f"On average, the engines have an estimated remaining useful life of about {avg_rul:.0f} cycles and an average health score of {avg_health:.0f} percent."]
    lines.append(f"{crit} engine{'s' if crit!=1 else ''} require urgent attention because the combined health and anomaly assessment is critical." if crit else "No engine is currently classified as critical.")
    lines.append(f"{attn} engine{'s' if attn!=1 else ''} require additional attention because they are degrading, showing abnormal telemetry, or carrying a warning." if attn else "No additional engines are currently flagged for elevated attention.")
    if healthy: lines.append(f"{healthy} engine{'s' if healthy!=1 else ''} are currently suitable for routine monitoring.")
    lines.append("These recommendations are model-based monitoring guidance and should be confirmed by qualified engineering and maintenance procedures.")
    return " ".join(lines)

fleet_brief=fleet_verbal_summary(fleet,dataset)
engine_briefs=[engine_verbal_summary(row) for _,row in fleet.sort_values("engine_id").iterrows()]
full_brief=fleet_brief+"\n\nEngine-by-engine summary:\n"+"\n".join(f"- {x}" for x in engine_briefs)

st.subheader("🗣️ Plain-English Engine Health Briefing")
st.info(fleet_brief)
with st.expander("Read the engine-by-engine maintenance briefing"):
    for brief in engine_briefs:
        st.write("• "+brief)

st.download_button("Download verbal maintenance briefing",full_brief.encode("utf-8"),f"{dataset}_PS_S02_verbal_maintenance_brief.txt","text/plain")
st.subheader("🛠 Maintenance Recommendation")
st.dataframe(fleet.sort_values(["overall_RUL_cycles","bearing_risk_score","anomaly_score"])[["engine_id","overall_RUL_cycles","overall_health_score","bearing_risk_score","bearing_status","flight_time_min_cumulative","anomaly_score","abnormal_sensors","trend_status","combined_status","overall_maintenance_recommendation"]],use_container_width=True,hide_index=True)
st.download_button("Download combined health/anomaly report",fleet.to_csv(index=False).encode(),f"{dataset}_PS_S02_hybrid_health_report.csv","text/csv")
