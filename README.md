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

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Select FD001–FD004 and upload either the NASA whitespace TXT or a 26-column CSV with or without the C-MAPSS header.

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
