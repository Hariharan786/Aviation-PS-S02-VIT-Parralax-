from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd
try:
    from .multi_fd_lstm import predict_file as predict_lstm
    from .rf_predictor import predict_file as predict_rf
    from .bearing_vibration import build_bearing_summary, add_bearing_status
except ImportError:
    from multi_fd_lstm import predict_file as predict_lstm
    from rf_predictor import predict_file as predict_rf
    from bearing_vibration import build_bearing_summary, add_bearing_status

COLUMNS=[
    "engine_id","cycle","setting_1","setting_2","setting_3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr",
    "Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfR_dmd","W31","W32"
]

def _weights(dataset,root):
    lm=json.loads((Path(root)/"models"/dataset/"metrics.json").read_text())
    rm=json.loads((Path(root)/"models"/dataset/"rf_metrics.json").read_text())
    lmae=max(float(lm["regression"]["MAE_cycles"]),1e-6); rmae=max(float(rm["MAE_cycles"]),1e-6)
    a=1/lmae; b=1/rmae
    return a/(a+b),b/(a+b),lmae,rmae

def _load_telemetry(path):
    first=Path(path).read_text(encoding="utf-8-sig",errors="replace").splitlines()[0]
    header=any(x in first.lower() for x in ["engine_id","cycle","setting_1"])
    if "," in first:
        d=pd.read_csv(path,header=0 if header else None,names=None if header else COLUMNS)
    elif "\t" in first:
        d=pd.read_csv(path,sep="\t",header=0 if header else None,names=None if header else COLUMNS)
    else:
        d=pd.read_csv(path,sep=r"\s+",header=None,names=COLUMNS,engine="python")
    d.columns=COLUMNS
    return d

def _integrate_bearing_health(out: pd.DataFrame, path) -> pd.DataFrame:
    telem=_load_telemetry(path)
    # Inject ensemble health into telemetry so the vibration physics model correlates with true engine degradation
    if "ensemble_health_score" in out.columns:
        telem = telem.merge(out[["engine_id", "cycle", "ensemble_health_score"]], on=["engine_id", "cycle"], how="left")
    bearing=add_bearing_status(build_bearing_summary(telem, sample_rate_hz=2048, duration_s=0.5, seed=42))
    keep=["engine_id","cycle","bearing_status","bearing_risk_score","bearing_severity","rms_g","peak_g","crest_factor","kurtosis",
          "cycle_duration_s","flight_time_s_cumulative","flight_time_min_cumulative","flight_distance_nm","mach","altitude_ft","airspeed_mps","true_airspeed_mps","throttle_proxy_pct","load_factor","bpfo_hz","bpfi_hz","bsf_hz","ftf_hz"]
    bearing=bearing[keep]
    out=out.merge(bearing,on=["engine_id","cycle"],how="left")
    # Convert the existing engine-health score into a composite score that explicitly
    # accounts for bearing risk and anomaly evidence.
    bearing_health=100.0-out["bearing_risk_score"].fillna(0.0)
    anomaly_health=100.0*np.clip(out.get("anomaly_score",pd.Series(0,index=out.index)).astype(float)*10.0,0.0,100.0)
    out["overall_health_score"]=np.clip(0.65*out["ensemble_health_score"]+0.20*anomaly_health+0.15*bearing_health,0,100)
    out["bearing_health_score"]=bearing_health
    # Bearing degradation reduces the usable RUL estimate rather than replacing the
    # core RUL model; the factor is intentionally capped so vibration cannot erase the model.
    out["overall_RUL_cycles"]=np.clip(out["ensemble_RUL_cycles"]*(1.0-0.25*out["bearing_risk_score"].fillna(0.0)/100.0),0,125)
    def status(r):
        bs=str(r.get("bearing_status","NORMAL")); es=str(r.get("ensemble_status","HEALTHY")); ah=float(r.get("anomaly_score",0.0));
        if bs=="CRITICAL" or es=="CRITICAL" or ah>=0.30 or float(r["overall_health_score"])<20: return "CRITICAL"
        if bs=="WARNING" or es=="WARNING" or ah>=0.18 or float(r["overall_health_score"])<40: return "WARNING"
        if bs=="WATCH" or es=="DEGRADING" or ah>=0.10 or float(r["overall_health_score"])<60: return "DEGRADING"
        return "HEALTHY"
    out["overall_status"]=out.apply(status,axis=1)
    def rec(r):
        bs=str(r.get("bearing_status","NORMAL")); rul=float(r["overall_RUL_cycles"]); risk=float(r.get("bearing_risk_score",0));
        if r["overall_status"]=="CRITICAL" or bs=="CRITICAL" or rul<15: return "URGENT: ENGINEERING INSPECTION — bearing vibration materially affects health/RUL"
        if r["overall_status"]=="WARNING" or bs=="WARNING" or rul<35: return "ATTENTION: SCHEDULE PREVENTIVE BEARING / ENGINE INSPECTION"
        if bs=="WATCH" or risk>=35 or rul<60: return "WATCH: INCREASED VIBRATION MONITORING AND PLAN INSPECTION"
        return "ROUTINE MONITORING"
    out["overall_maintenance_recommendation"]=out.apply(rec,axis=1)
    out["maintenance_signal"]=out["overall_maintenance_recommendation"]
    out["ensemble_health_score"]=out["overall_health_score"]
    out["ensemble_RUL_cycles"]=out["overall_RUL_cycles"]
    out["ensemble_status"]=out["overall_status"]
    return out

def predict_file(path,dataset,root):
    root=Path(root)
    la=root/"models"/dataset/"lstm_artifact.joblib"; ra=root/"models"/dataset/"rf_artifact.joblib"
    lr=predict_lstm(path,str(la)); rm,p=predict_rf(path,str(ra))
    if not lr[["engine_id","cycle"]].equals(rm[["engine_id","cycle"]]):
        rm=rm.set_index(["engine_id","cycle"]).reindex(pd.MultiIndex.from_frame(lr[["engine_id","cycle"]])).reset_index()
    wl,wr,lmae,rmae=_weights(dataset,root)
    out=lr.copy(); out["rf_RUL_cycles"]=p; out["lstm_RUL_cycles"]=out["predicted_RUL_cycles"]
    out["ensemble_RUL_cycles"]=np.clip(wl*out["lstm_RUL_cycles"]+wr*out["rf_RUL_cycles"],0,125)
    out["ensemble_health_score"]=np.clip(100*out["ensemble_RUL_cycles"]/125,0,100)
    out["ensemble_status"]=np.where(out.ensemble_RUL_cycles>75,"HEALTHY",np.where(out.ensemble_RUL_cycles>40,"DEGRADING",np.where(out.ensemble_RUL_cycles>15,"WARNING","CRITICAL")))
    out["ensemble_weight_lstm"]=wl; out["ensemble_weight_rf"]=wr
    out=_integrate_bearing_health(out,path)
    return out,{"lstm_weight":wl,"rf_weight":wr,"lstm_mae":lmae,"rf_mae":rmae,"bearing_integrated":True,"health_model":"65% RUL + 20% anomaly-health + 15% bearing-health"}
