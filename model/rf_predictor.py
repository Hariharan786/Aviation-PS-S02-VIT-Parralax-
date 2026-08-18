from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

COLUMNS=["engine_id","cycle","setting_1","setting_2","setting_3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfR_dmd","W31","W32"]
SETTINGS=["setting_1","setting_2","setting_3"]
SENSORS=["T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfR_dmd","W31","W32"]
SEQ_LEN=30
RUL_CAP=125.0

def load_dataset(path):
    with open(path,"r",encoding="utf-8-sig",errors="replace") as f: sample=f.read(4096)
    first=sample.splitlines()[0] if sample.splitlines() else ""
    has_header=any(x in first.lower() for x in ["engine_id","setting_1","cycle"])
    if "," in first:
        d=pd.read_csv(path,sep=",",header=0 if has_header else None,names=None if has_header else COLUMNS,engine="python")
    elif "\t" in first:
        d=pd.read_csv(path,sep="\t",header=0 if has_header else None,names=None if has_header else COLUMNS,engine="python")
    else:
        d=pd.read_csv(path,sep=r"\s+",header=None,names=COLUMNS,engine="python")
    if d.shape[1]!=len(COLUMNS): raise ValueError(f"Expected 26 C-MAPSS columns, received {d.shape[1]}.")
    d.columns=COLUMNS
    for c in COLUMNS: d[c]=pd.to_numeric(d[c],errors="coerce")
    if d[COLUMNS].isna().any().any(): raise ValueError("Telemetry contains missing/non-numeric values.")
    d.engine_id=d.engine_id.astype(int); d.cycle=d.cycle.astype(int)
    return d

def add_rul(d):
    d=d.copy(); last=d.groupby("engine_id").cycle.max(); d["RUL"]=np.minimum(d.engine_id.map(last)-d.cycle,RUL_CAP); return d

def clean(d,cols):
    x=d[cols].copy().astype(float).replace([np.inf,-np.inf],np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0.0)

def make_windows(d, cluster_model, condition_scalers, seq_len=SEQ_LEN, labels=True):
    d=d.sort_values(["engine_id","cycle"]).copy()
    d[SENSORS]=d[SENSORS].astype(float)
    d["condition"]=cluster_model.predict(clean(d,SETTINGS))
    # Condition-specific normalization, following predictor.py's design.
    for cond,scaler in condition_scalers.items():
        idx=d.condition==int(cond)
        if idx.any(): d.loc[idx,SENSORS]=scaler.transform(clean(d.loc[idx],SENSORS))
    # EMA smoothing after normalization, following predictor.py.
    for c in SENSORS: d[c]=d.groupby("engine_id")[c].transform(lambda s:s.ewm(span=10,adjust=False).mean())
    X=[]; y=[]; meta=[]
    for eid,g in d.groupby("engine_id",sort=True):
        a=g[SENSORS].to_numpy(np.float32)
        if labels:
            target=g.RUL.to_numpy(np.float32)
            for end in range(seq_len-1,len(g)):
                X.append(a[end-seq_len+1:end+1].reshape(-1)); y.append(target[end]); meta.append((int(eid),int(g.iloc[end].cycle)))
        else:
            if len(a)<seq_len: a=np.vstack([np.repeat(a[:1],seq_len-len(a),axis=0),a])
            X.append(a[-seq_len:].reshape(-1)); meta.append((int(eid),int(g.iloc[-1].cycle)))
    X=np.asarray(X,np.float32); meta=pd.DataFrame(meta,columns=["engine_id","cycle"])
    return (X,np.asarray(y,np.float32),meta) if labels else (X,meta)

def train_rf(dataset,train_path,out_dir,trees=120,seq_len=30,seed=42,max_samples=20000):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    raw=add_rul(load_dataset(train_path))
    ids=np.array(sorted(raw.engine_id.unique())); rng=np.random.default_rng(seed); rng.shuffle(ids)
    nval=max(1,int(.2*len(ids))); val_ids=set(ids[:nval]); tr_ids=set(ids[nval:])
    tr=raw[raw.engine_id.isin(tr_ids)].copy(); va=raw[raw.engine_id.isin(val_ids)].copy()
    k=6 if dataset in ("FD002","FD004") else 1
    km=KMeans(n_clusters=k,n_init=20,random_state=seed).fit(clean(tr,SETTINGS))
    tr["condition"]=km.predict(clean(tr,SETTINGS)); va["condition"]=km.predict(clean(va,SETTINGS))
    scalers={}
    global_ref=tr
    for cond in range(k):
        ref=tr[tr.condition==cond]
        sc=StandardScaler().fit(clean(ref,SENSORS) if len(ref)>=20 else clean(global_ref,SENSORS)); scalers[cond]=sc
    Xtr,ytr,mtr=make_windows(tr,km,scalers,seq_len,True)
    Xva,yva,mva=make_windows(va,km,scalers,seq_len,True)
    if len(Xtr)>max_samples:
        idx=rng.choice(len(Xtr),max_samples,replace=False); Xfit=Xtr[idx]; yfit=ytr[idx]
    else: Xfit,yfit=Xtr,ytr
    rf=RandomForestRegressor(n_estimators=trees,max_features="sqrt",min_samples_leaf=2,n_jobs=-1,random_state=seed)
    rf.fit(Xfit,yfit)
    pred=np.clip(rf.predict(Xva),0,RUL_CAP)
    metrics={"dataset":dataset,"model":"RandomForestRegressor","trees":trees,"sequence_length":seq_len,"train_windows":int(len(Xtr)),"fit_windows":int(len(Xfit)),"validation_engines":len(val_ids),"MAE_cycles":float(mean_absolute_error(yva,pred)),"RMSE_cycles":float(np.sqrt(mean_squared_error(yva,pred))),"R2":float(r2_score(yva,pred))}
    pd.DataFrame({"engine_id":mva.engine_id,"cycle":mva.cycle,"actual_RUL":yva,"predicted_RUL":pred,"absolute_error":np.abs(yva-pred)}).to_csv(out/"rf_validation_predictions.csv",index=False)
    artifact={"dataset":dataset,"model":"RandomForestRegressor","rf_model":rf,"kmeans_model":km,"cluster_scalers":scalers,"valid_sensors":SENSORS,"settings":SETTINGS,"sequence_length":seq_len,"rul_cap":RUL_CAP}
    joblib.dump(artifact,out/"rf_artifact.joblib")
    (out/"rf_metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics

def predict_rf(df,artifact):
    X,meta=make_windows(df,artifact["kmeans_model"],artifact["cluster_scalers"],artifact["sequence_length"],False)
    p=np.clip(artifact["rf_model"].predict(X),0,artifact.get("rul_cap",RUL_CAP))
    return meta,p

def predict_file(path,artifact_path):
    art=joblib.load(artifact_path); return predict_rf(load_dataset(path),art)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True,choices=["FD001","FD002","FD003","FD004"]); ap.add_argument("--train",required=True); ap.add_argument("--out",default="models"); ap.add_argument("--trees",type=int,default=120); args=ap.parse_args()
    out=Path(args.out)/args.dataset; print(json.dumps(train_rf(args.dataset,args.train,out,args.trees),indent=2))
