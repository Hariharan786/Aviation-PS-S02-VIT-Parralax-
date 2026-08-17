"""Generate bearing vibration data plus physics-derived variable flight timing from existing C-MAPSS telemetry."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'model'))
from bearing_vibration import build_bearing_summary, add_bearing_status, synthesize_cycle_waveform
COLUMNS=["engine_id","cycle","setting_1","setting_2","setting_3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfR_dmd","W31","W32"]

def load_telemetry(path: Path)->pd.DataFrame:
    first=path.read_text(encoding='utf-8-sig',errors='replace').splitlines()[0]
    header=any(x in first.lower() for x in ['engine_id','cycle','setting_1'])
    if ',' in first:
        d=pd.read_csv(path,header=0 if header else None,names=None if header else COLUMNS)
    elif '\t' in first:
        d=pd.read_csv(path,sep='\t',header=0 if header else None,names=None if header else COLUMNS)
    else:
        d=pd.read_csv(path,sep=r'\s+',header=None,names=COLUMNS,engine='python')
    if d.shape[1]!=26: raise ValueError(f'Expected 26 C-MAPSS columns, received {d.shape[1]}')
    d.columns=COLUMNS
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='FD001_physics_live_test.txt')
    ap.add_argument('--output-dir',default='bearing_output')
    ap.add_argument('--sample-rate',type=int,default=2048)
    ap.add_argument('--duration',type=float,default=None,help='Optional fixed waveform duration; omit to use physics-derived variable flight time')
    ap.add_argument('--engine',type=int,default=None,help='Only export one engine')
    args=ap.parse_args()
    df=load_telemetry(Path(args.input))
    if args.engine is not None: df=df[df.engine_id.astype(int)==args.engine]
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    summary=add_bearing_status(build_bearing_summary(df,args.sample_rate,args.duration,42))
    summary.to_csv(out/'bearing_vibration_summary.csv',index=False)
    chunks=[]
    for _,row in df.sort_values(['engine_id','cycle']).iterrows():
        wave,meta=synthesize_cycle_waveform(row,args.sample_rate,args.duration,42)
        chunks.append(wave)
    if chunks:
        pd.concat(chunks,ignore_index=True).to_csv(out/'bearing_vibration_timeseries.csv',index=False)
    print(f'Generated {len(summary)} cycle analyses in {out}')
    print(f'Raw time series: {out/"bearing_vibration_timeseries.csv"}')
    print(f'Summary: {out/"bearing_vibration_summary.csv"}')

if __name__=='__main__': main()
