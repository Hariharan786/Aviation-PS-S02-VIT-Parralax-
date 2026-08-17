# AeroGuard PS-S02 — Hybrid LSTM + Random Forest

This version combines the existing AeroGuard LSTM RUL model and anomaly pipeline with a Random Forest RUL model based on the uploaded `predictor.py` design.

## Hybrid architecture

Telemetry → format normalization (TXT/CSV) → condition-specific sensor scaling → EMA smoothing for RF branch →

- **LSTM RUL branch**: sequential 30-cycle model
- **Random Forest RUL branch**: 30-cycle flattened sensor window
- **Condition-aware anomaly branch**: K-Means + Isolation Forest + sensor robust-z scores

The final RUL is a validation-MAE-weighted ensemble of LSTM and Random Forest. The anomaly branch remains independent, so an engine can have acceptable RUL while still receiving an anomaly warning.

## What was borrowed from `predictor.py`

The RF branch follows the source design: K-Means operating-regime assignment, per-cluster sensor scalers, EMA smoothing, last-sequence extraction, flattening, and Random Forest prediction. The uploaded source uses these components in `CMAPSS_Predictor`. See `reference_predictor.py`.

The new implementation improves the source for this project by:
- using the dataset-specific number of regimes (FD001/FD003: 1; FD002/FD004: 6)
- accepting both TXT and CSV
- training one RF artifact per dataset
- using validation MAE to weight the LSTM/RF ensemble
- integrating the RF output into the existing anomaly/sensor/maintenance dashboard

## Dashboard

The operational-condition awareness graph has been removed as requested. Operating-condition clustering still runs internally because it is needed for condition-specific normalization and anomaly detection.

The dashboard contains:
- LSTM RUL
- Random Forest RUL
- Ensemble RUL and health score
- anomaly score
- abnormal sensors
- deterioration trends
- sensor performance
- anomaly precision/recall/F1
- detection delay
- maintenance recommendation

## Run

The project now includes two modes:
1. **Live Telemetry:** Generates physics-based data on the fly and streams it to the dashboard.
2. **Upload File:** Upload custom C-MAPSS telemetry for evaluation.

**Important:** Both the backend API server and the Streamlit frontend must be running simultaneously for the Live Telemetry mode to work.

### Using the batch file (Recommended for Windows)

Simply double-click the `run_frontend.bat` file, or run it in your terminal:
```powershell
.\run_frontend.bat
```
This automatically starts both the FastAPI simulation server on port 8000 and the Streamlit frontend on port 8501.

### Manual start (Two terminals)

If you prefer to start them manually, open two terminal windows:

**Terminal 1 (Backend API):**
```powershell
python -m pip install -r requirements.txt
python api_server.py
```

**Terminal 2 (Frontend UI):**
```powershell
python -m streamlit run app/app.py
```

Once running, select FD001–FD004 from the sidebar and use the **Live Telemetry** controls or upload either the NASA whitespace TXT or a 26-column CSV with or without the C-MAPSS header.

## Retrain all four datasets

```powershell
python train_all.py
```

This trains/updates LSTM, Random Forest and condition-aware anomaly artifacts for FD001–FD004.


## Plain-English maintenance briefing

The dashboard now generates a human-readable fleet briefing and an
engine-by-engine verbal maintenance summary from the same hybrid model
results shown in the tables.

The briefing explains:
- number of engines evaluated
- average estimated RUL
- average health score
- number of critical engines
- number of engines requiring attention
- number of engines suitable for routine monitoring
- engine-specific RUL, health, anomaly score, deterioration trend,
  abnormal sensors, and recommended action

A TXT report can be downloaded directly from the dashboard.

The language is deliberately simple for a hackathon presentation or
maintenance briefing. It is model-based guidance and is not a substitute
for qualified engineering/maintenance procedures.


## Bearing vibration simulation and analysis

The project now converts each C-MAPSS cycle into a physics-informed accelerometer time series. The bearing model derives Mach, airspeed, dynamic pressure, corrected core speed and mechanical load from the existing physics telemetry, then synthesises shaft-order vibration and bearing characteristic components (FTF, BPFO, BPFI and BSF), including degradation-dependent impulsive energy.

This is synthetic engineering telemetry for simulation/demo purposes; it is not a substitute for measured accelerometer data or certified maintenance limits.

### Generate bearing vibration + variable flight-time data

After generating the existing physics telemetry:

```powershell
python generate_bearing_timeseries.py --input FD001_physics_live_test.txt
```

Outputs are written to `bearing_output/`:
- `bearing_vibration_timeseries.csv` — high-frequency diagnostic waveform samples for the selected/generated cycles
- `bearing_vibration_summary.csv` — RMS, crest factor, kurtosis, bearing risk, shaft speed, Mach and variable flight-time fields per cycle
- `integrated_engine_health_demo.csv` — engine-level health/RUL/maintenance output after bearing integration
- `flight_timeline_demo.csv` — Mach/altitude/throttle-derived variable mission timing

The Streamlit dashboard also includes a **Bearing Vibration Analysis** section in both Upload File and Live Telemetry modes. It lets you select an engine/cycle, inspect the waveform and spectrum, and download the selected waveform and full cycle summary.

### Dependencies

`requirements.txt` now includes Plotly so environments that use the newer dashboard variant will not fail with `ModuleNotFoundError: No module named 'plotly'`.


## Bearing-integrated health and variable flight timing

The current version integrates the physics-informed bearing vibration model into the overall engine health decision. Bearing risk contributes to the composite health score and can reduce the effective RUL estimate and escalate the maintenance recommendation.

Flight timing is no longer a fixed per-cycle duration. Each telemetry cycle estimates a variable mission-segment duration from Mach-derived true airspeed, altitude, throttle/load proxy, and altitude change between cycles. The outputs include `cycle_duration_s`, cumulative `flight_time_s_cumulative`, `flight_time_min_cumulative`, `flight_distance_nm`, and operating-condition fields used to derive them. The high-frequency vibration waveform remains a short 0.5-second diagnostic capture per cycle so the export stays practical; its position in the mission is determined by the variable flight-time fields. Because C-MAPSS does not provide aircraft route distance or timestamps, this is a physics-informed mission-time estimate rather than a measured flight clock.
