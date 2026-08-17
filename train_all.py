from pathlib import Path
import json, subprocess, sys
from model.multi_fd_lstm import train_dataset
from model.condition_aware_anomaly import fit
from model.rf_predictor import train_rf

ROOT=Path(__file__).parent
CONFIG={
 "FD001":{"epochs":8,"hidden":64,"layers":1,"batch":512},
 "FD002":{"epochs":10,"hidden":64,"layers":1,"batch":2048},
 "FD003":{"epochs":20,"hidden":64,"layers":1,"batch":2048},
 "FD004":{"epochs":10,"hidden":64,"layers":1,"batch":2048},
}
for ds,cfg in CONFIG.items():
 print("\n"+"="*70); print("TRAINING LSTM:",ds); print("="*70)
 train_dataset(ds,str(ROOT/f"train_{ds}.txt"),str(ROOT/"models"),**cfg)
 print("\n"+"="*70); print("TRAINING RANDOM FOREST:",ds); print("="*70)
 train_rf(ds,str(ROOT/f"train_{ds}.txt"),ROOT/"models"/ds,trees=120,seq_len=30,max_samples=20000)
 print("\n"+"="*70); print("TRAINING CONDITION-AWARE ANOMALY MODEL:",ds); print("="*70)
 out=ROOT/"models"/ds
 _,metrics,sensors=fit(str(ROOT/f"train_{ds}.txt"),ds,out/"anomaly_artifact.joblib")
 (out/"anomaly_metrics.json").write_text(json.dumps(metrics,indent=2)); sensors.to_csv(out/"anomaly_sensor_performance.csv",index=False)
print("\nAll FD001-FD004 hybrid models trained.")
