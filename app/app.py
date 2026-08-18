import sys, json, tempfile, os, joblib, time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from hybrid_predictor import predict_file as predict_hybrid
from condition_aware_anomaly import load, score
from bearing_vibration import build_bearing_summary, add_bearing_status, synthesize_cycle_waveform, vibration_spectrum

# ---------------------------------------------------------------------------
# Import the physics telemetry generator (runs in-process on Streamlit Cloud)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ROOT))
from generate_fd001_physics_telemetry import generate as _generate_telemetry

st.set_page_config(page_title="FlightPret", page_icon="", layout="wide")

SENSOR_MAP = {
    "T2": "T2- Total Temperature Fan Inlet",
    "T24": "T24- Total Temperature LPC Out",
    "T30": "T30- Total Temperature HPC Out",
    "T50": "T50- Total Temperature LPT Out",
    "P2": "P2- Total Pressure Fan Inlet",
    "P15": "P15- Total Pressure Bypass Duct",
    "P30": "P30- Total Pressure HPC Out",
    "Nf": "Nf- Physical Fan Speed",
    "Nc": "Nc- Physical Core Speed",
    "epr": "epr- Engine Pressure Ratio",
    "Ps30": "Ps30- Static Pressure HPC Out",
    "phi": "phi- Fuel Flow- Ps30 Ratio",
    "NRf": "NRf- Corrected Fan Speed",
    "NRc": "NRc- Corrected Core Speed",
    "BPR": "BPR- Bypass Ratio",
    "farB": "farB- Burner Fuel-Air Ratio",
    "htBleed": "htBleed- Bleed Enthalpy",
    "Nf_dmd": "Nf_dmd- Demanded Fan Speed",
    "PCNfR_dmd": "PCNfR_dmd- Demanded Corrected Fan Speed",
    "W31": "W31- HPT Coolant Bleed",
    "W32": "W32- LPT Coolant Bleed",
    "setting_1": "setting_1- Alt. Setting",
    "setting_2": "setting_2- Mach Setting",
    "setting_3": "setting_3- TRA Setting"
}

# ---------------------------------------------------------------------------
# Custom CSS Injection for Fonts and Purplish Hues
# ---------------------------------------------------------------------------
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900&family=Rubik:wght@400;500;700&display=swap');

html, body, p, label, li, .stMarkdown, .stText {
    font-family: 'Rubik', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Lato', sans-serif !important;
}

/* Sidebar Custom Navigation Styling */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    padding: 16px 20px !important;
    border-radius: 8px !important;
    background-color: #1e1e24 !important;
    border: 1px solid #333 !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    margin: 0 !important;
}

/* Hide the radio circle */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important; 
    margin: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background-color: #27272f !important;
    border-color: #555 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input[aria-checked="true"]) {
    background-color: #1e1e24 !important;
    border: 1px solid #a855f7 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input[aria-checked="true"]) p {
    color: #a855f7 !important;
    font-weight: 700 !important;
}

div.stButton > button[kind="primary"] {
    background-color: #6b21a8 !important; 
    border-color: #6b21a8 !important;
    color: white !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #581c87 !important;
    border-color: #581c87 !important;
}

a {
    color: #6b21a8 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("FlightPret — Aircraft Engine Health & Predictive Maintenance")
st.caption("Use the **Dataset Selection** dropdown in the sidebar to switch between C-MAPSS datasets (FD001 – FD004).")

# ---------------------------------------------------------------------------
# Global Settings
# ---------------------------------------------------------------------------
# Dataset is now dynamic in the sidebar

def no_data_warning():
    st.markdown("""
    <div style="background-color: rgba(168, 85, 247, 0.1); border: 1px solid #a855f7; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
        <p style="color: #e9d5ff; margin: 0; font-size: 16px; font-weight: 500;">
            No data available. Please upload a telemetry file in the sidebar or start Live Telemetry.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialisation (Live Telemetry buffer + Dataset)
# ---------------------------------------------------------------------------
if "live_raw_txt" not in st.session_state:
    st.session_state.live_raw_txt = ""
if "live_rows" not in st.session_state:
    st.session_state.live_rows = 0
if "live_generated" not in st.session_state:
    st.session_state.live_generated = False
if "dataset" not in st.session_state:
    st.session_state.dataset = "FD001"

# ---------------------------------------------------------------------------
# Sidebar Navigation & Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Go to", ["Overview", "Live Telemetry", "Engine Health"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### Dataset Selection")
    _selected_ds = st.selectbox(
        "Select C-MAPSS Dataset",
        ["FD001", "FD002", "FD003", "FD004"],
        index=["FD001", "FD002", "FD003", "FD004"].index(st.session_state.dataset),
        key="dataset_selectbox",
    )
    if _selected_ds != st.session_state.dataset:
        st.session_state.dataset = _selected_ds
        st.session_state.live_raw_txt = ""
        st.session_state.live_rows = 0
        st.session_state.live_generated = False
        st.cache_data.clear()
        st.rerun()
    dataset = st.session_state.dataset

    st.markdown("---")
    st.markdown("### Upload Data")
    uploaded = st.file_uploader(
        f"Upload unseen telemetry ({dataset})",
        type=["txt", "csv"],
        help="C-MAPSS telemetry only; CSV may include the 26-column header.",
    )

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

with st.sidebar:
    st.markdown("---")
    with st.expander("Model Readiness", expanded=True):
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            st.metric("LSTM R² (Deep Learning)",  f"{m['regression']['R2']:.3f}")
        if rf_metrics_path.exists():
            rm = json.loads(rf_metrics_path.read_text())
            st.metric("Random Forest R² (Ensemble)",  f"{rm['R2']:.3f}")
        if anom_metrics_path.exists():
            am = json.loads(anom_metrics_path.read_text())
            st.metric("Anomaly Detection F1 Score", f"{am['f1']:.3f}")

# ---------------------------------------------------------------------------
# COLUMNS definition used by the normaliser
# ---------------------------------------------------------------------------
COLUMNS = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "T2- Total Temperature at Fan Inlet", "T24- Total Temperature LPC Out", "T30- Total Temperature HPC Out", "T50- Total Temperature LPT Out",
    "P2- Pressure at Fan Inlet", "P15- Total Pressure in Bypass Duct", "P30- Total Pressure HPC Out", "Nf- Physical Fan Speed", "Nc- Physical Core Speed", "epr- Engine Pressure Ratio",
    "Ps30- Static Pressure HPC Out", "phi- Fuel Flow-Ps30 Ratio", "NRf- Corrected Fan Speed", "NRc- Corrected Core Speed", "BPR- Bypass Ratio", "farB- Burner Fuel-Air Ratio", "htBleed- Bleed Enthalpy",
    "Nf_dmd- Demanded Fan Speed", "PCNfR_dmd- Demanded Corrected Fan Speed", "W31- HPT Coolant Bleed", "W32- LPT Coolant Bleed",
]

@st.cache_data(show_spinner="Running AI inference pipeline...", max_entries=2)
def cached_process_telemetry(raw_bytes: bytes, data_source: str, dataset_name: str, root_str: str):
    import tempfile, os, joblib
    with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as f:
        f.write(raw_bytes)
        src = f.name
    
    out_name = None
    try:
        if data_source == "upload":
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
            out_name = out.name
            for _, r in d.iterrows():
                out.write(" ".join(
                    str(int(r[c])) if c in ["engine_id", "cycle"] else f"{float(r[c]):.12g}"
                    for c in COLUMNS
                ) + "\n")
            out.close()
        else:
            out_name = src
            
        fleet_rul, ensemble_meta = predict_hybrid(out_name, dataset_name, Path(root_str))
        raw_anom = load(out_name)
        anom_art = joblib.load(Path(root_str) / "models" / dataset_name / "anomaly_artifact.joblib")
        anom = score(raw_anom, anom_art)
        
        return fleet_rul, ensemble_meta, raw_anom, anom
    finally:
        try:
            os.unlink(src)
        except OSError:
            pass
        if out_name and out_name != src:
            try:
                os.unlink(out_name)
            except OSError:
                pass

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def health_word(status):
    return {
        "HEALTHY":          "healthy",
        "DEGRADING":        "showing signs of degradation",
        "WARNING":          "requiring attention",
        "WARNING + ANOMALY": "showing degradation and abnormal telemetry",
        "ANOMALY":          "showing abnormal telemetry",
        "CRITICAL":         "in a critical condition",
    }.get(str(status), str(status).lower())

def maintenance_priority(row):
    status = str(row["combined_status"])
    signal = str(row.get("overall_maintenance_recommendation", row.get("maintenance_signal", "")))
    if status == "CRITICAL" or signal.startswith("URGENT"): return "URGENT"
    if "WARNING" in status or status in ["ANOMALY", "DEGRADING"] or "INSPECTION" in signal.upper(): return "ATTENTION"
    return "ROUTINE"

def engine_verbal_summary(row):
    eid = int(row["engine_id"])
    rul = float(row.get("overall_RUL_cycles", row.get("ensemble_RUL_cycles", 0)))
    health = float(row.get("overall_health_score", row.get("ensemble_health_score", 0)))
    anomaly = float(row["anomaly_score"])
    bearing = float(row.get("bearing_risk_score", 0.0))
    bstatus = str(row.get("bearing_status", "NORMAL"))
    flight_min = float(row.get("flight_time_min_cumulative", 0.0))
    status = health_word(row["combined_status"])
    trend = str(row["trend_status"]).lower()
    sensors = str(row["abnormal_sensors"])
    if sensors in ["", "nan", "None", "[]"]:
        sensor_text = "No individual sensor is currently flagged as abnormal."
    else:
        import ast
        try:
            s_list = ast.literal_eval(sensors)
            mapped = [SENSOR_MAP.get(s, s) for s in s_list]
            sensor_text = f"The main sensors flagged are: {', '.join(mapped)}."
        except:
            sensor_text = f"The main sensors flagged are {sensors}."
    action = str(row.get("overall_maintenance_recommendation", "Continue routine monitoring."))
    return (
        f"Engine {eid} is {status}. Its integrated remaining useful life is about {rul:.0f} cycles, "
        f"with an overall health score of {health:.0f} percent. "
        f"Bearing vibration risk is {bearing:.0f}/100 ({bstatus}), with about {flight_min:.1f} minutes of estimated flight time accumulated in the observed mission. "
        f"The current anomaly score is {anomaly:.2f} and the deterioration trend is {trend}. "
        f"{sensor_text} Recommendation: {action}."
    )

def fleet_verbal_summary(fleet_df, dataset_name):
    total = len(fleet_df)
    crit = int((fleet_df["combined_status"] == "CRITICAL").sum())
    attn  = int(fleet_df["combined_status"].isin(["WARNING","WARNING + ANOMALY","ANOMALY","DEGRADING"]).sum())
    healthy = total - crit - attn
    avg_rul = float(fleet_df.get("overall_RUL_cycles", fleet_df.get("ensemble_RUL_cycles", pd.Series([0]))).mean())
    avg_health = float(fleet_df.get("overall_health_score", fleet_df.get("ensemble_health_score", pd.Series([0]))).mean())
    lines = [
        f"{dataset_name} fleet health briefing.",
        f"The system evaluated {total} engine{'s' if total != 1 else ''}.",
        f"On average, the engines have an estimated remaining useful life of about {avg_rul:.0f} cycles and an average health score of {avg_health:.0f} percent.",
    ]
    if crit:
        lines.append(f"{crit} engine{'s' if crit != 1 else ''} require urgent attention because the combined health and anomaly assessment is critical.")
    else:
        lines.append("No engine is currently classified as critical.")
    if attn:
        lines.append(f"{attn} engine{'s' if attn != 1 else ''} require additional attention because they are degrading, showing abnormal telemetry, or carrying a warning.")
    else:
        lines.append("No additional engines are currently flagged for elevated attention.")
    if healthy:
        lines.append(f"{healthy} engine{'s' if healthy != 1 else ''} are currently suitable for routine monitoring.")
    lines.append("These recommendations are model-based monitoring guidance and should be confirmed by qualified engineering and maintenance procedures.")
    return " ".join(lines)


def render_bearing_analysis(telemetry_df, selected_engine=None):
    if telemetry_df is None or telemetry_df.empty:
        return
    try:
        bearing = add_bearing_status(build_bearing_summary(telemetry_df, sample_rate_hz=2048, duration_s=0.5, seed=42))
    except Exception as exc:
        st.warning(f"Bearing vibration analysis unavailable: {exc}")
        return

    st.subheader("Bearing Vibration Analysis")
    st.caption("Physics-informed synthetic accelerometer signal. Bearing vibration now feeds the integrated engine health/RUL decision.")

    latest = bearing.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1).copy()
    
    if selected_engine is not None:
        latest = latest[latest.engine_id == selected_engine]
        if latest.empty:
            st.warning("No bearing data available for this engine.")
            return

        avg_risk = float(latest["bearing_risk_score"].mean())
        status = latest["bearing_status"].iloc[0]
        st.write(f"**Bearing Risk Score:** {avg_risk:.1f}/100 — **Status:** {status}")

        st.dataframe(latest[["engine_id", "cycle", "mach", "altitude_ft", "throttle_proxy_pct", "airspeed_mps", "core_rpm", "cycle_duration_s", "flight_time_min_cumulative", "bpfo_hz", "bpfi_hz", "bsf_hz", "rms_g", "crest_factor", "kurtosis", "bearing_risk_score", "bearing_status"]], use_container_width=True, hide_index=True)

        cycles = sorted(bearing.loc[bearing.engine_id == selected_engine, "cycle"].astype(int).unique())
        selected_cycle = st.selectbox("Bearing waveform cycle", cycles, index=len(cycles)-1, key=f"bearing_cycle_{selected_engine}")
        row = telemetry_df[(telemetry_df.engine_id.astype(int) == int(selected_engine)) & (telemetry_df.cycle.astype(int) == int(selected_cycle))].iloc[0]
        waveform, meta = synthesize_cycle_waveform(row, sample_rate_hz=2048, duration_s=0.5, seed=42)

        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Mach", f"{meta['mach']:.3f}")
        w2.metric("Core speed", f"{meta['core_rpm']:.0f} rpm")
        w3.metric("RMS", f"{meta['rms_g']:.3f} g")
        w4.metric("Crest factor", f"{meta['crest_factor']:.2f}")

        c_wave = alt.Chart(waveform).mark_line(color="#8b5cf6").encode(
            x=alt.X("time_s:Q", title="Time (s)"),
            y=alt.Y("vibration_g:Q", scale=alt.Scale(zero=False), title="Vibration (g)"),
            tooltip=["time_s:Q", "vibration_g:Q"]
        ).interactive().properties(height=260)
        st.altair_chart(c_wave, use_container_width=True)
        
        spectrum = vibration_spectrum(waveform, 2048, 1200.0)
        c_spec = alt.Chart(spectrum).mark_line(color="#ec4899").encode(
            x=alt.X("frequency_hz:Q", title="Frequency (Hz)"),
            y=alt.Y("amplitude_g:Q", scale=alt.Scale(zero=False), title="Amplitude (g)"),
            tooltip=["frequency_hz:Q", "amplitude_g:Q"]
        ).interactive().properties(height=240)
        st.altair_chart(c_spec, use_container_width=True)
        st.caption(f"Expected bearing frequencies: FTF {meta['ftf_hz']:.1f} Hz · BPFO {meta['bpfo_hz']:.1f} Hz · BPFI {meta['bpfi_hz']:.1f} Hz · BSF {meta['bsf_hz']:.1f} Hz. Elevated energy near these characteristic frequencies is used as a bearing-fault indicator.")

        st.download_button(
            "Download bearing cycle summary",
            bearing.to_csv(index=False).encode("utf-8"),
            f"bearing_summary_engine_{selected_engine}.csv",
            "text/csv",
            key=f"bearing_summary_{selected_engine}",
        )
        st.download_button(
            "Download selected bearing waveform",
            waveform.to_csv(index=False).encode("utf-8"),
            f"bearing_engine_{selected_engine}_cycle_{selected_cycle}_waveform.csv",
            "text/csv",
            key=f"bearing_wave_{selected_engine}",
        )

# ---------------------------------------------------------------------------
# Data Resolution Logic (Live vs Upload)
# ---------------------------------------------------------------------------
rows_ready = st.session_state.live_rows
raw_txt    = st.session_state.live_raw_txt
is_live    = st.session_state.live_generated

# Prioritize uploaded file if present, otherwise use live telemetry
data_source = None
raw_bytes = None
mode_indicator = "None"

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    data_source = "upload"
    mode_indicator = "📁 Upload File"
elif raw_txt.strip() and rows_ready >= 30:
    raw_bytes = raw_txt.encode("utf-8")
    data_source = "live"
    mode_indicator = "Live Telemetry"

fleet_rul = None
raw_anom = None
anom = None
ensemble_meta = None

if raw_bytes is not None:
    try:
        fleet_rul, ensemble_meta, raw_anom, anom = cached_process_telemetry(raw_bytes, data_source, dataset, str(ROOT))
    except Exception as e:
        st.error(f"Could not process telemetry: {e}")
        st.exception(e)

# Pre-compute fleet summary if data is available
fleet = None
if anom is not None and fleet_rul is not None:
    latest_anom = anom.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
    latest_rul  = fleet_rul.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
    fleet = latest_rul.merge(
        latest_anom[["engine_id", "cycle", "anomaly_score", "abnormal_condition",
                     "abnormal_sensor_count", "abnormal_sensors", "trend_status"]],
        on=["engine_id", "cycle"], how="left",
    )
    def combined(row):
        return str(row.get("overall_status", row.get("ensemble_status", "HEALTHY")))
    fleet["combined_status"] = fleet.apply(combined, axis=1)


# ---------------------------------------------------------------------------
# Render Specific Page
# ---------------------------------------------------------------------------

if page == "Overview":
    st.header("Overview")
    if fleet is None:
        no_data_warning()
        if anom_metrics_path.exists():
            st.subheader("Abnormal-Condition Evaluation (Historical)")
            am = json.loads(anom_metrics_path.read_text())
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Precision", f"{am['precision']:.3f}")
            x2.metric("Recall",    f"{am['recall']:.3f}")
            x3.metric("F1",        f"{am['f1']:.3f}")
            x4.metric("Detection delay", f"{am['mean_detection_delay_cycles']:.2f} cycles" if am["mean_detection_delay_cycles"] is not None else "N/A")
    else:
        critical = int((fleet.combined_status == "CRITICAL").sum())
        warning  = int(fleet.combined_status.str.contains("WARNING").sum())
        abnormal = int((fleet.abnormal_condition != "NORMAL").sum())

        st.caption(
            f"Ensemble weights: LSTM {ensemble_meta['lstm_weight']:.1%} · "
            f"Random Forest {ensemble_meta['rf_weight']:.1%}. "
            f"Weights are based on inverse validation MAE."
            + (f"  |  ⟳ Last refreshed with **{rows_ready}** telemetry rows." if data_source == "live" else "")
        )
        
        a, b, c, d, e = st.columns(5)
        a.metric("Engines",          len(fleet))
        rul_col = "overall_RUL_cycles" if "overall_RUL_cycles" in fleet.columns else "ensemble_RUL_cycles"
        b.metric("Avg RUL", f"{fleet[rul_col].mean():.1f} cycles")
        c.metric("Abnormal",         abnormal)
        d.metric("Warning",          warning)
        e.metric("Critical",         critical)

        st.divider()
        st.subheader("Current Engine Health")
        cols_to_show = [
            "engine_id", "cycle", "overall_RUL_cycles", "ensemble_RUL_cycles", "overall_health_score",
            "bearing_risk_score", "bearing_status", "cycle_duration_s", "flight_time_min_cumulative",
            "lstm_RUL_cycles", "rf_RUL_cycles", "anomaly_score", "abnormal_condition",
            "abnormal_sensor_count", "abnormal_sensors", "trend_status", "combined_status",
            "overall_maintenance_recommendation", "maintenance_signal",
        ]
        cols_to_show = [c for c in cols_to_show if c in fleet.columns]
        
        def color_status(val):
            if not isinstance(val, str):
                return ""
            val_upper = val.upper()
            if "WARNING" in val_upper or "CRITICAL" in val_upper or "URGENT" in val_upper:
                return "color: #ef4444; font-weight: bold;"
            elif "WATCH" in val_upper or "ATTENTION" in val_upper or "DEGRADING" in val_upper or "ANOMALY" in val_upper:
                return "color: #eab308; font-weight: bold;"
            elif "NORMAL" in val_upper or "HEALTHY" in val_upper or "ROUTINE" in val_upper:
                return "color: #22c55e; font-weight: bold;"
            return ""
            
        styled_fleet = fleet[cols_to_show].style.map(color_status)
        st.dataframe(styled_fleet, use_container_width=True, hide_index=True)

        st.subheader("Maintenance Recommendation")
        maint_cols = [
            "engine_id", "overall_RUL_cycles", "overall_health_score", "bearing_risk_score",
            "bearing_status", "flight_time_min_cumulative", "anomaly_score", "abnormal_sensors",
            "trend_status", "combined_status", "overall_maintenance_recommendation"
        ]
        maint_cols = [c for c in maint_cols if c in fleet.columns]
        sort_cols = [c for c in ["overall_RUL_cycles", "bearing_risk_score", "anomaly_score"] if c in fleet.columns]
        
        styled_maint = fleet.sort_values(sort_cols)[maint_cols].style.map(color_status)
        st.dataframe(
            styled_maint,
            use_container_width=True, hide_index=True,
        )

        st.subheader("Sensor Performance & Abnormality")
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
        if not sp.empty:
            sp["sensor"] = sp["sensor"].map(lambda x: SENSOR_MAP.get(x, x))
        st.dataframe(sp, use_container_width=True, hide_index=True)
        if not sp.empty and "absolute_spearman" in sp:
            st.bar_chart(sp.sort_values("absolute_spearman", ascending=False).set_index("sensor")[["absolute_spearman"]])

        if anom_metrics_path.exists() and data_source == "upload":
            st.subheader("Abnormal-Condition Evaluation")
            am = json.loads(anom_metrics_path.read_text())
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Precision", f"{am['precision']:.3f}")
            x2.metric("Recall",    f"{am['recall']:.3f}")
            x3.metric("F1",        f"{am['f1']:.3f}")
            x4.metric("Detection delay", f"{am['mean_detection_delay_cycles']:.2f} cycles" if am["mean_detection_delay_cycles"] is not None else "N/A")
            cm = pd.DataFrame(am["confusion_matrix"], index=["Actual Normal", "Actual Abnormal"], columns=["Pred Normal", "Pred Abnormal"])
            st.dataframe(cm, use_container_width=True)
            st.caption("Evaluation proxy: validation RUL ≥ 80 = normal; RUL ≤ 40 = abnormal; RUL 41–79 excluded.")

        fleet_brief   = fleet_verbal_summary(fleet, dataset)
        engine_briefs = [engine_verbal_summary(row) for _, row in fleet.sort_values("engine_id").iterrows()]
        full_brief    = fleet_brief + "\n\nEngine-by-engine summary:\n" + "\n".join(f"- {x}" for x in engine_briefs)

        st.subheader("Fleet Thermal Efficiency Trends")
        if raw_anom is not None and not raw_anom.empty:
            df_te = raw_anom.copy()
            # Ideal Brayton cycle efficiency based on Overall Pressure Ratio (OPR = P30 / P2)
            # gamma = 1.4 -> (gamma-1)/gamma = 0.2857
            df_te["thermal_efficiency"] = 1 - (df_te["P2"] / df_te["P30"])**0.2857
            
            # Apply Exponentially Weighted Moving Average (span=15) for smooth trend visualization
            df_te["smoothed_efficiency"] = df_te.groupby("engine_id")["thermal_efficiency"].transform(
                lambda x: x.ewm(span=15, adjust=False).mean()
            )
            
            chart_te = alt.Chart(df_te.reset_index()).mark_line(opacity=0.8).encode(
                x=alt.X("cycle:Q", title="Cycle"),
                y=alt.Y("smoothed_efficiency:Q", scale=alt.Scale(zero=False), title="Ideal Brayton Efficiency (smoothed)"),
                color=alt.Color("engine_id:N", legend=alt.Legend(title="Engine ID")),
                tooltip=["engine_id:N", "cycle:Q", "smoothed_efficiency:Q"]
            ).interactive()
            st.altair_chart(chart_te, use_container_width=True)
            
        st.subheader("Engine-by-Engine Maintenance Briefing")
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

elif page == "Live Telemetry":
    st.header("Live Telemetry")

    with st.expander("Live Generation Settings", expanded=(not is_live)):
        live_num_engines = st.slider("Engines", 2, 20, 10)
        live_cycles      = st.slider("Cycles / engine", 10, 100, 40)
        live_seed        = st.number_input("Random seed", value=42, step=1)

        c_start, c_stop = st.columns(2)
        with c_start:
            if st.button("▶ Start Telemetry", type="primary", use_container_width=True):
                train_file = ROOT / f"train_{dataset}.txt"
                if not train_file.exists():
                    st.error(
                        f"`train_{dataset}.txt` not found at `{ROOT}`. "
                        "Make sure the training data file is present in the repository root."
                    )
                else:
                    with st.spinner("Generating telemetry... this may take a few seconds."):
                        try:
                            tmp = tempfile.NamedTemporaryFile(
                                delete=False, suffix="_live.txt", mode="w"
                            )
                            tmp_path = tmp.name
                            tmp.close()

                            _generate_telemetry(
                                train_file=str(train_file),
                                output_file=tmp_path,
                                num_engines=int(live_num_engines),
                                cycles_per_engine=int(live_cycles),
                                seed=int(live_seed),
                                generate_bearing=False,  # skip CSV side-effects on cloud
                            )

                            with open(tmp_path, "r") as f:
                                generated_txt = f.read()

                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass

                            total_rows = live_num_engines * live_cycles
                            st.session_state.live_raw_txt  = generated_txt
                            st.session_state.live_rows     = total_rows
                            st.session_state.live_generated = True
                            # Clear the inference cache so it re-runs on the new data
                            cached_process_telemetry.clear()
                            st.toast(f"✅ {total_rows} telemetry rows generated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Generation failed: {e}")
                            st.exception(e)

        with c_stop:
            if st.button("⏹ Stop / Clear Telemetry", use_container_width=True):
                st.session_state.live_raw_txt   = ""
                st.session_state.live_rows      = 0
                st.session_state.live_generated = False
                cached_process_telemetry.clear()
                st.toast("⏹ Telemetry buffer cleared.")
                st.rerun()

    st.divider()

    status_placeholder = st.empty()
    if is_live and rows_ready > 0:
        status_placeholder.success(f"✅ {rows_ready} rows in buffer — ready for analysis.")
    elif rows_ready > 0:
        status_placeholder.info(f"⏸ {rows_ready} rows in buffer (press ▶ Start to regenerate).")
    else:
        status_placeholder.warning("▶ Press Start Telemetry above to generate live data.")

    if 0 < rows_ready < 30:
        st.info(f"Waiting for at least 30 rows before running the pipeline (currently {rows_ready}).")

    if raw_anom is not None and not raw_anom.empty:
        st.subheader("Raw Telemetry Data")
        st.dataframe(raw_anom, use_container_width=True)

        st.subheader("Live Sensor Trends")
        plot_options = ["T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi", "BPR", "htBleed", "W31", "W32"]
        sensors_to_plot = st.multiselect(
            "Select sensors to plot",
            options=plot_options,
            default=["T24", "P30", "Nc"],
            format_func=lambda x: SENSOR_MAP.get(x, x)
        )
        
        if sensors_to_plot:
            engines_to_plot = sorted(raw_anom.engine_id.unique())
            selected_engine = st.selectbox("Select Engine for Telemetry Plots", engines_to_plot, key="telemetry_engine_select")
            plot_data = raw_anom[raw_anom.engine_id == selected_engine].sort_values("cycle")
            if not plot_data.empty:
                for sensor in sensors_to_plot:
                    desc_name = SENSOR_MAP.get(sensor, sensor)
                    st.markdown(f"**{desc_name} Trend**")
                    chart = alt.Chart(plot_data.reset_index()).mark_line(color="#38bdf8").encode(
                        x=alt.X("cycle:Q", title="Cycle"),
                        y=alt.Y(f"{sensor}:Q", scale=alt.Scale(zero=False), title=desc_name),
                        tooltip=["cycle:Q", f"{sensor}:Q"]
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No data for this engine.")

elif page == "Engine Health":
    st.header("Engine Health")
    if fleet is None or anom is None:
        no_data_warning()
    else:
        engines = sorted(anom.engine_id.unique())
        selected = st.selectbox("Select Engine", engines, key="engine_health_select")
        
        st.subheader("Deterioration Trends")
        trend  = anom[anom.engine_id == selected].sort_values("cycle").copy()
        ruleng = fleet_rul[fleet_rul.engine_id == selected].sort_values("cycle")
        
        # Split anomaly_score and max_sensor_z onto separate charts to avoid crushing scale
        c_anom = alt.Chart(trend.reset_index()).mark_line(color="#38bdf8").encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("anomaly_score:Q", scale=alt.Scale(zero=False), title="Anomaly Score"),
            tooltip=["cycle:Q", "anomaly_score:Q"]
        ).interactive().properties(title="Anomaly Score Trend", height=200)

        c_zscore = alt.Chart(trend.reset_index()).mark_line(color="#f97316").encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("max_sensor_z:Q", scale=alt.Scale(zero=False), title="Max Sensor Z-Score"),
            tooltip=["cycle:Q", "max_sensor_z:Q"]
        ).interactive().properties(title="Max Sensor Z-Score Trend", height=200)

        st.altair_chart(alt.vconcat(c_anom, c_zscore).resolve_scale(y="independent"), use_container_width=True)

        # RUL + health score trend across all cycles for the selected engine
        # fleet_rul typically contains one row per engine (last-cycle prediction only).
        # When that is the case we reconstruct a synthetic per-cycle history from the
        # anom cycle list using: RUL(c) ≈ final_RUL + (max_cycle - c)
        rul_history = fleet_rul[fleet_rul.engine_id == selected].sort_values("cycle")
        rul_col    = "overall_RUL_cycles"   if "overall_RUL_cycles"   in rul_history.columns else "ensemble_RUL_cycles"
        health_col = "overall_health_score" if "overall_health_score" in rul_history.columns else "ensemble_health_score"

        if not rul_history.empty:
            if len(rul_history) <= 1:
                # Build synthetic history from anom cycles
                final_row    = rul_history.iloc[-1]
                final_rul    = float(final_row[rul_col])
                final_health = float(final_row[health_col])
                final_cycle  = int(final_row["cycle"])

                cycles_anom  = trend["cycle"].astype(int).sort_values().values
                min_cycle    = int(cycles_anom.min()) if len(cycles_anom) else 1

                rul_series    = [final_rul + (final_cycle - int(c)) for c in cycles_anom]
                # Linear health: 100 at first cycle → final_health at last cycle
                health_series = [
                    100.0 + (final_health - 100.0) * (int(c) - min_cycle) / max(final_cycle - min_cycle, 1)
                    for c in cycles_anom
                ]
                rul_history = pd.DataFrame({"cycle": cycles_anom, rul_col: rul_series, health_col: health_series})

            c_rul = alt.Chart(rul_history.reset_index()).mark_line(color="#a855f7").encode(
                x=alt.X("cycle:Q", title="Cycle"),
                y=alt.Y(f"{rul_col}:Q", scale=alt.Scale(zero=False), title="RUL (cycles)"),
                tooltip=["cycle:Q", f"{rul_col}:Q"]
            ).interactive().properties(title="Remaining Useful Life Trend", height=200)

            c_health = alt.Chart(rul_history.reset_index()).mark_line(color="#22c55e").encode(
                x=alt.X("cycle:Q", title="Cycle"),
                y=alt.Y(f"{health_col}:Q", scale=alt.Scale(zero=False), title="Health Score (%)"),
                tooltip=["cycle:Q", f"{health_col}:Q"]
            ).interactive().properties(title="Health Score Trend", height=200)

            st.altair_chart(alt.vconcat(c_rul, c_health).resolve_scale(y="independent"), use_container_width=True)
        else:
            st.info("No RUL / health history available for this engine.")
        
        st.write(
            f"Engine {selected}: anomaly **{trend.anomaly_score.iloc[-1]:.3f}**, "
            f"trend **{trend.trend_status.iloc[-1]}**, "
            f"abnormal sensors **{trend.abnormal_sensors.iloc[-1]}**"
        )
        
        render_bearing_analysis(raw_anom, selected_engine=selected)