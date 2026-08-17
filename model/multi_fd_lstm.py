
from __future__ import annotations
import argparse, copy, json, random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)

COLUMNS = [
    "engine_id","cycle","setting_1","setting_2","setting_3",
    "T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr",
    "Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd",
    "PCNfR_dmd","W31","W32"
]

ALL_SENSORS = [
    "T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr",
    "Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd",
    "PCNfR_dmd","W31","W32"
]

# FD001 benefits from removing near-constant channels. The other subsets
# retain the full telemetry because their multiple operating conditions make
# those channels useful.
FD001_SENSORS = [
    "T24","T30","T50","P30","Ps30","Nc","NRc","phi","epr"
]

def dataset_sensors(dataset):
    return FD001_SENSORS if dataset == "FD001" else ALL_SENSORS

def dataset_features(dataset):
    return ["setting_1","setting_2","setting_3"] + dataset_sensors(dataset)
SEQ_LEN = 30
RUL_CAP = 125.0
HEALTH_LABELS = ["HEALTHY","DEGRADING","WARNING","CRITICAL"]

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_dataset(path):
    """Robust C-MAPSS loader for TXT, CSV, or TSV input."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(4096)

    first = sample.splitlines()[0] if sample.splitlines() else ""
    has_header = any(
        token in first.lower()
        for token in ["engine_id", "setting_1", "cycle"]
    )

    if "," in first:
        df = pd.read_csv(
            path, sep=",",
            header=0 if has_header else None,
            names=None if has_header else COLUMNS,
            engine="python"
        )
    elif "\t" in first:
        df = pd.read_csv(
            path, sep="\t",
            header=0 if has_header else None,
            names=None if has_header else COLUMNS,
            engine="python"
        )
    else:
        df = pd.read_csv(
            path, sep=r"\s+",
            header=None, names=COLUMNS, engine="python"
        )

    # Header normalization.
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Telemetry schema error: expected {len(COLUMNS)} columns, "
            f"received {df.shape[1]}."
        )

    if not all(str(c) in COLUMNS for c in df.columns):
        df.columns = COLUMNS

    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[["engine_id", "cycle"]].isna().any(axis=1).any():
        raise ValueError(
            "Invalid engine_id/cycle. If uploading CSV, ensure it is a "
            "comma-separated 26-column C-MAPSS telemetry file."
        )

    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    return df[COLUMNS].copy()

def add_rul(df):
    d=df.copy()
    final=d.groupby("engine_id")["cycle"].max()
    d["RUL"]=np.minimum(d["engine_id"].map(final)-d["cycle"], RUL_CAP)
    return d

def clean(df, features=None):
    d=df.copy()
    features=features or dataset_features("FD001")
    d[features]=d[features].astype(float)
    d[features]=d[features].replace([np.inf,-np.inf],np.nan)
    d[features]=d[features].fillna(d[features].median(numeric_only=True))
    return d

def make_sequences(df, scaler, seq_len=SEQ_LEN, labels=True, features=None):
    features=features or dataset_features("FD001")
    d=clean(df,features).sort_values(["engine_id","cycle"]).copy()
    z=scaler.transform(d[features]).astype(np.float32)
    d[features]=z
    X=[]; y=[]; meta=[]
    for eid,g in d.groupby("engine_id",sort=True):
        g=g.sort_values("cycle")
        a=g[features].to_numpy(np.float32)
        if labels:
            target=g["RUL"].to_numpy(np.float32)
            for end in range(seq_len-1,len(g)):
                X.append(a[end-seq_len+1:end+1])
                y.append(target[end])
                meta.append((int(eid),int(g.iloc[end]["cycle"])))
        else:
            if len(g)<seq_len:
                a=np.vstack([np.repeat(a[:1],seq_len-len(g),axis=0),a])
            X.append(a[-seq_len:])
            meta.append((int(eid),int(g.iloc[-1]["cycle"])))
    X=np.asarray(X,np.float32)
    meta=pd.DataFrame(meta,columns=["engine_id","cycle"])
    return (X,np.asarray(y,np.float32),meta) if labels else (X,meta)

class SequenceDataset(Dataset):
    def __init__(self,X,y=None):
        self.X=torch.from_numpy(X)
        self.y=None if y is None else torch.from_numpy(y[:,None])
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        return self.X[i] if self.y is None else (self.X[i],self.y[i])

class EngineLSTM(nn.Module):
    def __init__(self,input_size,hidden=96,layers=2,dropout=.2):
        super().__init__()
        self.lstm=nn.LSTM(input_size,hidden,layers,batch_first=True,
                          dropout=dropout if layers>1 else 0)
        self.norm=nn.LayerNorm(hidden)
        self.head=nn.Sequential(
            nn.Linear(hidden,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1)
        )
    def forward(self,x):
        z,_=self.lstm(x)
        return self.head(self.norm(z[:,-1,:]))

def predict(model,X,device,batch=1024):
    dl=DataLoader(SequenceDataset(X),batch_size=batch,shuffle=False)
    out=[]; model.eval()
    with torch.no_grad():
        for xb in dl:
            out.append(model(xb.to(device)).cpu().numpy().ravel())
    return np.maximum(0,np.concatenate(out))

def regression_metrics(y,p):
    return {
        "MAE_cycles":float(mean_absolute_error(y,p)),
        "RMSE_cycles":float(np.sqrt(mean_squared_error(y,p))),
        "R2":float(r2_score(y,p))
    }

def health_class(rul):
    rul=np.asarray(rul,float)
    return np.where(rul>75,"HEALTHY",
           np.where(rul>40,"DEGRADING",
           np.where(rul>15,"WARNING","CRITICAL")))

def classification_metrics(y,p):
    yt=health_class(y); yp=health_class(p)
    rep=classification_report(
        yt,yp,labels=HEALTH_LABELS,target_names=HEALTH_LABELS,
        output_dict=True,zero_division=0
    )
    return {
        "accuracy":float(rep["accuracy"]),
        "macro_f1":float(rep["macro avg"]["f1-score"]),
        "weighted_f1":float(rep["weighted avg"]["f1-score"]),
        "report":rep,
        "confusion_matrix":confusion_matrix(yt,yp,labels=HEALTH_LABELS).tolist()
    }

def sensor_performance(df, dataset):
    # Dataset-level degradation signal, evaluated on held-out validation rows.
    sensors=dataset_sensors(dataset)
    d=clean(df, dataset_features(dataset))
    rows=[]
    y=d["RUL"].to_numpy(float)
    for sensor in sensors:
        x=d[sensor].to_numpy(float)
        if np.std(x)==0 or np.std(y)==0:
            pearson=0.0
        else:
            pearson=float(np.corrcoef(x,y)[0,1])
        # Spearman via rank correlation; avoids an extra dependency.
        rx=pd.Series(x).rank(method="average").to_numpy()
        ry=pd.Series(y).rank(method="average").to_numpy()
        spearman=float(np.corrcoef(rx,ry)[0,1]) if np.std(rx)>0 else 0.0
        rows.append({
            "sensor":sensor,
            "pearson_r":pearson,
            "spearman_r":spearman,
            "absolute_spearman":abs(spearman),
            "signal":("STRONG" if abs(spearman)>=.60 else
                      "MODERATE" if abs(spearman)>=.30 else "WEAK")
        })
    return pd.DataFrame(rows).sort_values("absolute_spearman",ascending=False)

def train_dataset(dataset, train_path, output_root, epochs=5, seq_len=30,
                  hidden=96, layers=2, batch=1024, lr=.001, seed=42):
    seed_everything(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=Path(output_root)/dataset
    out.mkdir(parents=True,exist_ok=True)

    raw=add_rul(load_dataset(train_path))
    ids=np.array(sorted(raw.engine_id.unique()))
    rng=np.random.default_rng(seed); rng.shuffle(ids)
    nval=max(1,int(len(ids)*.2))
    val_ids=set(ids[:nval]); train_ids=set(ids[nval:])
    tr=raw[raw.engine_id.isin(train_ids)].copy()
    va=raw[raw.engine_id.isin(val_ids)].copy()

    features=dataset_features(dataset)
    scaler=StandardScaler().fit(clean(tr,features)[features])
    Xtr,ytr,mtr=make_sequences(tr,scaler,seq_len,True,features)
    Xva,yva,mva=make_sequences(va,scaler,seq_len,True,features)

    model=EngineLSTM(len(features),hidden,layers,.2).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    loss_fn=nn.HuberLoss(delta=10.0)
    train_dl=DataLoader(SequenceDataset(Xtr,ytr),batch_size=batch,shuffle=True)
    val_dl=DataLoader(SequenceDataset(Xva,yva),batch_size=batch,shuffle=False)

    best=float("inf"); best_state=None
    for epoch in range(1,epochs+1):
        model.train(); total=0; count=0
        for xb,yb in train_dl:
            xb,yb=xb.to(device),yb.to(device)
            optimizer.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total += loss.item()*len(xb); count += len(xb)
        model.eval(); v=0; vc=0
        with torch.no_grad():
            for xb,yb in val_dl:
                l=loss_fn(model(xb.to(device)),yb.to(device))
                v += l.item()*len(xb); vc += len(xb)
        vl=v/vc
        print(f"[{dataset}] epoch {epoch}/{epochs} train={total/count:.3f} val={vl:.3f}")
        if vl<best:
            best=vl; best_state=copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    pred=predict(model,Xva,device)

    reg=regression_metrics(yva,pred)
    cls=classification_metrics(yva,pred)
    sensor=sensor_performance(va,dataset)

    pd.DataFrame({
        "engine_id":mva.engine_id,
        "cycle":mva.cycle,
        "actual_RUL":yva,
        "predicted_RUL":pred,
        "absolute_error":np.abs(yva-pred)
    }).to_csv(out/"validation_predictions.csv",index=False)

    sensor.to_csv(out/"sensor_performance.csv",index=False)
    (out/"classification_report.json").write_text(json.dumps(cls,indent=2))

    report={"dataset":dataset,"device":str(device),"train_engines":len(train_ids),
            "validation_engines":len(val_ids),"regression":reg,
            "classification":cls}
    (out/"metrics.json").write_text(json.dumps(report,indent=2))

    artifact={
        "dataset":dataset,
        "state":{k:v.cpu() for k,v in model.state_dict().items()},
        "input_size":len(features),"hidden":hidden,"layers":layers,"dropout":.2,
        "seq_len":seq_len,"rul_cap":RUL_CAP,"features":features,
        "columns":COLUMNS,"scaler":scaler
    }
    joblib.dump(artifact,out/"lstm_artifact.joblib")
    print(f"[{dataset}] REG={reg} CLASS_ACC={cls['accuracy']:.4f} MACRO_F1={cls['macro_f1']:.4f}")
    return report

def predict_file(test_path, artifact_path):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    art=joblib.load(artifact_path)
    model=EngineLSTM(art["input_size"],art["hidden"],art["layers"],art["dropout"]).to(device)
    model.load_state_dict(art["state"]); model.eval()

    df=load_dataset(test_path)
    X,meta=make_sequences(
        df,art["scaler"],art["seq_len"],False,art["features"]
    )
    p=predict(model,X,device)
    health=np.clip(100*p/art["rul_cap"],0,100)
    status=health_class(p)

    rec=[]
    for r,s in zip(p,status):
        if s=="CRITICAL" or r<=20:
            rec.append("CRITICAL: prioritize engineering inspection.")
        elif s=="WARNING":
            rec.append("WARNING: schedule preventive maintenance.")
        elif s=="DEGRADING":
            rec.append("DEGRADING: increase monitoring and plan maintenance.")
        else:
            rec.append("NORMAL: continue routine telemetry monitoring.")

    result=meta.copy()
    result["predicted_RUL_cycles"]=p
    result["health_score"]=health
    result["status"]=status
    result["maintenance_recommendation"]=rec
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["FD001","FD002","FD003","FD004"])
    ap.add_argument("--train")
    ap.add_argument("--output",default="models")
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--seq-len",type=int,default=30)
    ap.add_argument("--hidden",type=int,default=96)
    ap.add_argument("--layers",type=int,default=2)
    args=ap.parse_args()
    if args.dataset:
        train_dataset(args.dataset,args.train,args.output,args.epochs,
                      args.seq_len,args.hidden,args.layers)
