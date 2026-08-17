from __future__ import annotations
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
try:
    from .multi_fd_lstm import predict_file as predict_lstm
    from .rf_predictor import predict_file as predict_rf
except ImportError:
    from multi_fd_lstm import predict_file as predict_lstm
    from rf_predictor import predict_file as predict_rf

def _weights(dataset,root):
    lm=json.loads((Path(root)/"models"/dataset/"metrics.json").read_text())
    rm=json.loads((Path(root)/"models"/dataset/"rf_metrics.json").read_text())
    lmae=max(float(lm["regression"]["MAE_cycles"]),1e-6); rmae=max(float(rm["MAE_cycles"]),1e-6)
    a=1/lmae; b=1/rmae
    return a/(a+b),b/(a+b),lmae,rmae

def predict_file(path,dataset,root):
    root=Path(root)
    la=root/"models"/dataset/"lstm_artifact.joblib"; ra=root/"models"/dataset/"rf_artifact.joblib"
    lr=predict_lstm(path,str(la)); rm,p=predict_rf(path,str(ra))
    if not lr[["engine_id","cycle"]].equals(rm[["engine_id","cycle"]]):
        rm=rm.set_index(["engine_id","cycle"]).reindex(pd.MultiIndex.from_frame(lr[["engine_id","cycle"]])).reset_index()
    wl,wr,lmae,rmae=_weights(dataset,root)
    out=lr.copy(); out["rf_RUL_cycles"]=rm.iloc[:len(out)][1].to_numpy(float) if len(rm.columns)>1 and rm.columns[1] not in ["engine_id","cycle"] else p
    out["rf_RUL_cycles"]=p
    out["lstm_RUL_cycles"]=out["predicted_RUL_cycles"]
    out["ensemble_RUL_cycles"]=np.clip(wl*out["lstm_RUL_cycles"]+wr*out["rf_RUL_cycles"],0,125)
    out["ensemble_health_score"]=np.clip(100*out["ensemble_RUL_cycles"]/125,0,100)
    out["ensemble_status"]=np.where(out.ensemble_RUL_cycles>75,"HEALTHY",np.where(out.ensemble_RUL_cycles>40,"DEGRADING",np.where(out.ensemble_RUL_cycles>15,"WARNING","CRITICAL")))
    out["ensemble_weight_lstm"]=wl; out["ensemble_weight_rf"]=wr
    return out,{"lstm_weight":wl,"rf_weight":wr,"lstm_mae":lmae,"rf_mae":rmae}
