from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

from utils.collector import *
from utils.add_features import *
from utils.deal_csv import *

from config import MIN_HISTORY_HOURS

MAD_EPS=1e-6 # ゼロ除算防止用の微小値

# 個人チューニングした異常検知のベースラインを算出する
# ある個人の過去データから平均的な値を算出し新しいデータがどれだけ外れているかを調べる

def summary_features(df):
    df,_=add_engineered_features(df)
    t=get_time(df)
    hour=(t/3600)%24
    # 10分単位の説明可能な特徴
    rows=[]
    for sid,seg in make_10min_segments(df):
        f,_=add_engineered_features(seg)
        steps=pd.to_numeric(f["Step_Diff"],errors="coerce").fillna(0)
        acc=pd.to_numeric(f["Acc_Mag"],errors="coerce")
        rows.append({
            "segment":sid,
            "steps_10min":float(steps.sum()),
            "activity_mean_proxy":float(acc.mean()),
            "acc_std":float(acc.std()),
            "gyro_mean":float(pd.to_numeric(f["Gyro_Mag"],errors="coerce").mean()),
        })
    return pd.DataFrame(rows)

def robust_stats(values):
    x=np.asarray(values,float); med=np.nanmedian(x); mad=np.nanmedian(np.abs(x-med))
    scale=max(1.4826*mad,MAD_EPS)
    return float(med),float(scale)

def build_baseline(csv_paths, out="baseline.json"):
    rows=[]
    for p in csv_paths:
        df=read_csv(p); s=summary_features(df)
        if len(s): rows.append(s)
    if not rows: raise ValueError("ベースライン用データがありません。")
    x=pd.concat(rows,ignore_index=True)
    features=["steps_10min","activity_mean_proxy","acc_std","gyro_mean"]
    baseline={"required_history_hours":MIN_HISTORY_HOURS,"features":{}}
    for f in features:
        med,scale=robust_stats(x[f])
        baseline["features"][f]={"median":med,"mad_scale":scale}
    Path(out).write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding="utf-8")
    print("saved:",out)

def score_10min(csv_path, baseline_path, z_threshold=3.5):
    b=json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    df=read_csv(csv_path); x=summary_features(df)
    results=[]
    for _,r in x.iterrows():
        deviations={}
        for f,s in b["features"].items():
            score=abs(float(r[f])-s["median"])/max(s["mad_scale"],MAD_EPS)
            if score>=z_threshold: deviations[f]={"value":float(r[f]),"baseline":s["median"],"score":score}
        results.append({"segment":int(r["segment"]),"anomalies":deviations})
    return results

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["build","score"],required=True)
    ap.add_argument("--csv",nargs="+",required=True)
    ap.add_argument("--out",default="baseline.json")
    args=ap.parse_args()
    if args.mode=="build": build_baseline(args.csv,args.out)
    else: print(json.dumps(score_10min(args.csv,args.out),ensure_ascii=False,indent=2))
