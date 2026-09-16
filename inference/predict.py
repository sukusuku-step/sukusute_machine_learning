from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib

from models import BehaviorActivityModel,DistanceModel
import utils.add_features, utils.collector, utils.deal_csv

from config import LABEL_SEC, WINDOW_LEN, N_WINDOWS

# 学習済みモデルを使い、10分毎でのデータからその推論を出力する
# 推論は歩数・加速度・相対距離からのそれぞれのラベル分類と活動量の値の推測

def behavior_infer(csv_path,artifact_dir):
    d=Path(artifact_dir); cfg=json.loads((d/"config.json").read_text(encoding="utf-8"))
    scaler=joblib.load(d/"scaler.joblib")
    model=BehaviorActivityModel(len(cfg["feature_cols"]),len(cfg["pedo_classes"]),len(cfg["acce_classes"]))
    model.load_state_dict(torch.load(d/"behavior_activity.pt",map_location="cpu")); model.eval()
    df=utils.collector.read_csv(csv_path); df,fc=utils.add_features.add_engineered_features(df)
    blocks=utils.add_features.make_10min_segments(df)
    xs=[]; valid_blocks=[]
    for sid,seg in blocks:
        w=utils.add_features.split_into_windows(seg,fc)
        if len(w)==N_WINDOWS:
            xs.append(w); valid_blocks.append((sid,seg))
    if not xs: return []
    x=utils.add_features.apply_scaler(np.stack(xs).astype(np.float32),scaler)
    # x [S,60,100,C] -> model [1,S,60,100,C]
    with torch.no_grad():
        po,ao,actlogit=model(torch.tensor(x).unsqueeze(0))
        activity_prob=actlogit.softmax(-1)[0]
        activity=int(activity_prob.argmax())+1
    results=[]
    for i,(sid,_) in enumerate(valid_blocks):
        pp=po[0,i].softmax(-1); aa=ao[0,i].softmax(-1)
        pi=int(pp.argmax()); ai=int(aa.argmax())
        results.append({
            "segment_start_sec":sid*LABEL_SEC,
            "pedo_label":cfg["pedo_classes"][pi],"pedo_confidence":float(pp[pi]),
            "acce_label":cfg["acce_classes"][ai],"acce_confidence":float(aa[ai]),
            # ここは「この時点まで蓄積したセッションからの活動量予測」
            "activity_level_from_accumulated_data":activity,
            "activity_confidence":float(activity_prob[activity-1])
        })
    return results

def distance_infer(csv_path,artifact_dir):
    d=Path(artifact_dir); cfg=json.loads((d/"config.json").read_text(encoding="utf-8"))
    scaler=joblib.load(d/"scaler.joblib")
    model=DistanceModel(n_classes=len(cfg["classes"]))
    model.load_state_dict(torch.load(d/"distance.pt",map_location="cpu")); model.eval()
    df=utils.collector.read_csv(csv_path); results={}
    for dc in utils.deal_csv.numeric_distance_columns(df):
        lc=utils.deal_csv.distance_label_col(dc)
        if lc not in df.columns: continue
        # 短い欠測のみ補間。長い欠測は学習/推論対象から除外する。
        arr=pd.to_numeric(df[dc],errors="coerce").interpolate(limit=20)
        rr=[]
        temp=pd.DataFrame({"Timestamp":df["Timestamp"],dc:arr,lc:df[lc]})
        for sid,seg in utils.deal_csv.make_10min_segments(temp):
            a=pd.to_numeric(seg[dc],errors="coerce").to_numpy()
            if np.isnan(a).mean()>0.2: continue
            a=pd.Series(a).interpolate(limit=20).ffill().bfill().to_numpy()
            if len(a)<N_WINDOWS*WINDOW_LEN: continue
            a=a[:N_WINDOWS*WINDOW_LEN].reshape(1,N_WINDOWS,WINDOW_LEN,1).astype(np.float32)
            a=utils.add_features.apply_scaler(a,scaler)
            with torch.no_grad(): p=model(torch.tensor(a))[0].softmax(-1)
            k=int(p.argmax())
            rr.append({"segment_start_sec":sid*LABEL_SEC,"label":cfg["classes"][k],"confidence":float(p[k])})
        results[dc]=rr
    return results

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True)
    ap.add_argument("--behavior-artifact",default="artifacts/behavior")
    ap.add_argument("--distance-artifact",default="artifacts/distance")
    args=ap.parse_args()
    if Path(args.behavior_artifact).exists():
        print(json.dumps({"behavior_activity":behavior_infer(args.csv,args.behavior_artifact)},ensure_ascii=False,indent=2))
    if Path(args.distance_artifact).exists():
        print(json.dumps({"distance":distance_infer(args.csv,args.distance_artifact)},ensure_ascii=False,indent=2))