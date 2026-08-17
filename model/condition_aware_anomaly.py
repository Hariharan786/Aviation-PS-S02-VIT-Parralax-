
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

COLUMNS=[
"engine_id","cycle","setting_1","setting_2","setting_3","T2","T24","T30","T50",
"P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB",
"htBleed","Nf_dmd","PCNfR_dmd","W31","W32"
]
SETTINGS=["setting_1","setting_2","setting_3"]
SENSORS=COLUMNS[5:]
ALL_FEATURES=SETTINGS+SENSORS

def load(path):
    """Load NASA whitespace TXT or comma/tab-delimited C-MAPSS CSV."""
    path = str(path)
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(4096)

    first = sample.splitlines()[0] if sample.splitlines() else ""
    header = first.lower()
    has_header = any(x in header for x in ["engine_id", "cycle", "setting_1"])

    if "," in first:
        d = pd.read_csv(
            path, sep=",", header=0 if has_header else None,
            names=None if has_header else COLUMNS, engine="python"
        )
    elif "\t" in first:
        d = pd.read_csv(
            path, sep="\t", header=0 if has_header else None,
            names=None if has_header else COLUMNS, engine="python"
        )
    else:
        d = pd.read_csv(
            path, sep=r"\s+", header=None, names=COLUMNS, engine="python"
        )

    if d.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} C-MAPSS columns, received {d.shape[1]}."
        )

    # Normalize header names when a CSV has headers.
    by_lower={c.lower():c for c in COLUMNS}
    rename={}
    for c in d.columns:
        s=str(c).strip()
        if s.lower() in by_lower:
            rename[c]=by_lower[s.lower()]
    d=d.rename(columns=rename)

    if set(COLUMNS).issubset(d.columns):
        d=d[COLUMNS].copy()
    else:
        d.columns=COLUMNS

    for c in COLUMNS:
        d[c]=pd.to_numeric(d[c],errors="coerce")

    if d[["engine_id","cycle"]].isna().any(axis=1).any():
        raise ValueError("Invalid engine_id/cycle values in telemetry file.")

    return d

def add_rul(df,cap=125.0):
    d=df.copy()
    last=d.groupby("engine_id")["cycle"].max()
    d["RUL"]=np.minimum(d["engine_id"].map(last)-d["cycle"],cap)
    return d

def clean(df, cols):
    x=df[cols].copy().astype(float)
    x=x.replace([np.inf,-np.inf],np.nan)
    return x.fillna(x.median(numeric_only=True))

def make_conditions(df,dataset):
    k=6 if dataset in ("FD002","FD004") else 1
    scaler=StandardScaler().fit(clean(df,SETTINGS))
    z=scaler.transform(clean(df,SETTINGS))
    km=KMeans(n_clusters=k,n_init=20,random_state=42)
    km.fit(z)
    return scaler,km

def condition_labels(df,scaler,km):
    return km.predict(scaler.transform(clean(df,SETTINGS)))

def robust_stats(ref):
    med=ref[SENSORS].median()
    mad=(ref[SENSORS]-med).abs().median()
    scale=(1.4826*mad).replace(0,np.nan)
    # Avoid NaNs for near-constant sensors.
    scale=scale.fillna(ref[SENSORS].std().replace(0,np.nan)).fillna(1e-6)
    return med,scale

def fit(train_file,dataset,out_file):
    raw=add_rul(load(train_file))
    ids=np.array(sorted(raw.engine_id.unique()))
    rng=np.random.default_rng(42); rng.shuffle(ids)
    nval=max(1,int(.20*len(ids)))
    val_ids=set(ids[:nval]); tr_ids=set(ids[nval:])
    tr=raw[raw.engine_id.isin(tr_ids)].copy()
    va=raw[raw.engine_id.isin(val_ids)].copy()

    condition_scaler,km=make_conditions(tr,dataset)
    tr["condition"]=condition_labels(tr,condition_scaler,km)
    va["condition"]=condition_labels(va,condition_scaler,km)

    feature_scaler=StandardScaler().fit(clean(tr,ALL_FEATURES))
    models={}
    thresholds={}
    sensor_stats={}

    global_ref=tr[tr.RUL>=100].copy()
    if len(global_ref)<500:
        global_ref=tr.groupby("engine_id").head(5).copy()

    for cond in range(km.n_clusters):
        ref=tr[(tr.condition==cond)&(tr.RUL>=100)].copy()
        if len(ref)<200:
            ref=global_ref
        X=feature_scaler.transform(clean(ref,ALL_FEATURES))
        iso=IsolationForest(
            n_estimators=300,contamination=.03,
            random_state=42,n_jobs=-1
        ).fit(X)
        score=-iso.decision_function(X)
        thresholds[cond]={
            "warning":float(np.quantile(score,.95)),
            "critical":float(np.quantile(score,.99)),
            "healthy_reference_rows":int(len(ref))
        }
        models[cond]=iso
        med,scale=robust_stats(ref)
        sensor_stats[cond]={"median":med.to_dict(),"scale":scale.to_dict()}

    # Evaluate only clearly normal vs degraded samples to avoid ambiguous labels.
    va_scores=[]
    for cond,g in va.groupby("condition"):
        iso=models[int(cond)]
        X=feature_scaler.transform(clean(g,ALL_FEATURES))
        s=-iso.decision_function(X)
        th=thresholds[int(cond)]
        flag=(s>=th["warning"]).astype(int)
        proxy=np.where(g.RUL<=40,1,np.where(g.RUL>=80,0,-1))
        keep=proxy>=0
        if keep.any():
            va_scores.append(pd.DataFrame({
                "engine_id":g.loc[keep,"engine_id"].to_numpy(),
                "cycle":g.loc[keep,"cycle"].to_numpy(),
                "RUL":g.loc[keep,"RUL"].to_numpy(),
                "condition":int(cond),
                "anomaly_score":s[keep],
                "predicted_anomaly":flag[keep],
                "proxy_abnormal":proxy[keep]
            }))
    ev=pd.concat(va_scores,ignore_index=True)

    pr,rc,f1,_=precision_recall_fscore_support(
        ev.proxy_abnormal,ev.predicted_anomaly,
        average="binary",zero_division=0
    )
    cm=confusion_matrix(ev.proxy_abnormal,ev.predicted_anomaly).tolist()

    # Detection delay: first warning at/after the proxy degradation onset.
    delays=[]
    missed=0
    for eid,g in ev.groupby("engine_id"):
        onset=int(g.loc[g.RUL<=40,"cycle"].min()) if (g.RUL<=40).any() else None
        if onset is None: continue
        det=g.loc[(g.cycle>=onset)&(g.predicted_anomaly==1),"cycle"]
        if len(det):
            delays.append(int(det.min()-onset))
        else:
            missed+=1

    # Sensor performance: validation Spearman against RUL, and condition-aware
    # robust z-score statistics for interpretability.
    sensor_rows=[]
    for sensor in SENSORS:
        x=va[sensor].astype(float).to_numpy()
        y=va.RUL.astype(float).to_numpy()
        rx=pd.Series(x).rank().to_numpy(); ry=pd.Series(y).rank().to_numpy()
        spear=float(np.corrcoef(rx,ry)[0,1]) if np.std(rx)>0 and np.std(ry)>0 else 0.0
        sensor_rows.append({
            "sensor":sensor,
            "spearman_r":spear,
            "absolute_spearman":abs(spear),
            "signal":("STRONG" if abs(spear)>=.60 else "MODERATE" if abs(spear)>=.30 else "WEAK")
        })
    sensor_df=pd.DataFrame(sensor_rows).sort_values("absolute_spearman",ascending=False)

    artifact={
        "dataset":dataset,
        "condition_count":km.n_clusters,
        "settings":SETTINGS,
        "sensors":SENSORS,
        "all_features":ALL_FEATURES,
        "condition_scaler":condition_scaler,
        "condition_model":km,
        "feature_scaler":feature_scaler,
        "anomaly_models":models,
        "thresholds":thresholds,
        "sensor_stats":sensor_stats
    }
    joblib.dump(artifact,out_file)

    metrics={
        "dataset":dataset,
        "condition_count":km.n_clusters,
        "validation_engines":len(val_ids),
        "proxy_definition":"abnormal=RUL<=40; normal=RUL>=80; ambiguous 41-79 excluded",
        "precision":float(pr),
        "recall":float(rc),
        "f1":float(f1),
        "confusion_matrix":cm,
        "detected_engines":int(len(delays)),
        "missed_engines":int(missed),
        "mean_detection_delay_cycles":float(np.mean(delays)) if delays else None,
        "median_detection_delay_cycles":float(np.median(delays)) if delays else None,
        "sensor_performance":sensor_df.to_dict("records")
    }
    return artifact,metrics,sensor_df

def score(df,artifact):
    d=df.copy()
    cs=artifact["condition_scaler"]; km=artifact["condition_model"]
    d["condition"]=condition_labels(d,cs,km)
    fs=artifact["feature_scaler"]
    anomaly=[]; maxz=[]; counts=[]; tops=[]
    for cond,g in d.groupby("condition",sort=False):
        idx=g.index
        X=fs.transform(clean(g,ALL_FEATURES))
        iso=artifact["anomaly_models"][int(cond)]
        s=-iso.decision_function(X)
        anomaly.extend(zip(idx,s))
        stats=artifact["sensor_stats"][int(cond)]
        med=pd.Series(stats["median"]); scale=pd.Series(stats["scale"])
        z=(g[SENSORS]-med)/scale
        az=z.abs()
        maxz.extend(zip(idx,az.max(axis=1)))
        counts.extend(zip(idx,(az>=3).sum(axis=1)))
        for i,row in az.iterrows():
            names=row.sort_values(ascending=False)
            tops.append((i,", ".join(names[names>=2.5].head(3).index) or "None"))
    score_map=dict(anomaly); max_map=dict(maxz); count_map=dict(counts); top_map=dict(tops)
    d["anomaly_score"]=d.index.map(score_map)
    d["max_sensor_z"]=d.index.map(max_map)
    d["abnormal_sensor_count"]=d.index.map(count_map)
    d["abnormal_sensors"]=d.index.map(top_map)

    warnings=[]
    criticals=[]
    for _,r in d.iterrows():
        th=artifact["thresholds"][int(r.condition)]
        warnings.append(r.anomaly_score>=th["warning"])
        criticals.append(r.anomaly_score>=th["critical"])
    d["abnormal_condition"]=np.where(
        criticals,"CRITICAL ANOMALY",np.where(warnings,"WARNING","NORMAL")
    )
    d=d.sort_values(["engine_id","cycle"])
    d["persistent_3_cycles"]=(
        d.groupby("engine_id")["abnormal_condition"]
        .transform(lambda s:s.ne("NORMAL").rolling(3,min_periods=3).sum().ge(3))
    )
    d["anomaly_trend_5"]=(
        d.groupby("engine_id")["anomaly_score"]
        .transform(lambda s:s.diff(4)/4)
    )
    d["trend_status"]=np.select(
        [d.anomaly_trend_5>0.01,d.anomaly_trend_5<-0.01],
        ["WORSENING","IMPROVING"],default="STABLE"
    )
    d["maintenance_signal"]=np.select(
        [
            d.abnormal_condition.eq("CRITICAL ANOMALY"),
            d.abnormal_condition.eq("WARNING") & d.persistent_3_cycles,
            d.abnormal_condition.eq("WARNING")
        ],
        [
            "IMMEDIATE ENGINEERING INSPECTION",
            "SCHEDULE PREVENTIVE INSPECTION",
            "INCREASED MONITORING"
        ],
        default="ROUTINE MONITORING"
    )
    return d

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True,choices=["FD001","FD002","FD003","FD004"])
    ap.add_argument("--train",required=True)
    ap.add_argument("--out",default="models")
    args=ap.parse_args()
    out=Path(args.out)/args.dataset
    out.mkdir(parents=True,exist_ok=True)
    artifact,metrics,sensors=fit(
        args.train,args.dataset,out/"anomaly_artifact.joblib"
    )
    (out/"anomaly_metrics.json").write_text(json.dumps(metrics,indent=2))
    sensors.to_csv(out/"anomaly_sensor_performance.csv",index=False)
    print(json.dumps(metrics,indent=2))
