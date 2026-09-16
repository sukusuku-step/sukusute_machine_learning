from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import joblib

from utils.collector import *
from utils.add_features import *
from utils.deal_csv import *
from models import DistanceModel

from config import N_WINDOWS, SEED

# 各相手との相対距離データからCNN+LSTMにより学習を行う
# 相対距離のラベル分類が推論結果となるモデル

# 学習データの順番をシャッフルする
np.random.seed(SEED); torch.manual_seed(SEED)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--out",default="artifacts/distance")
    ap.add_argument("--epochs",type=int,default=30)

    args=ap.parse_args()
    paths=glob.glob(os.path.join(args.data,"*.csv")) if os.path.isdir(args.data) else [args.data]

    examples=[]
    labels=[]

    for path in paths:
        df = read_csv(path)

        # CSV内のDistance列を測定相手ごとに処理
        for dc in numeric_distance_columns(df):
            lc = distance_label_col(dc)

            if lc not in df.columns:
                continue

            d = pd.to_numeric(df[dc], errors="coerce")
            temp = pd.DataFrame({"Timestamp": df["Timestamp"], dc: d, lc: df[lc]})

            for _, seg in make_10min_segments(temp):
                lab = label_from_segment(seg, lc)

                if not lab:
                    continue

                arr = seg[dc].to_numpy(dtype=np.float32)
                arr = arr.reshape(N_WINDOWS, WINDOW_LEN, 1)
                examples.append(arr)
                labels.append(lab)

    if not examples: 
        raise ValueError("Distance_i + Distance_i_Label の学習データがありません。")
    
    X=np.stack(examples)
    enc=LabelEncoder().fit(labels)
    y=enc.transform(labels)
    scaler=fit_scaler(X)
    X=apply_scaler(X,scaler)

    model=DistanceModel(n_classes=len(enc.classes_))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loss_fn=nn.CrossEntropyLoss()
    model.train()

    for ep in range(args.epochs):
        order=np.random.permutation(len(X)); total=0
        for i in order:
            xb=torch.tensor(X[i:i+1]); yb=torch.tensor(y[i:i+1])
            opt.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=loss.item()

        if (ep+1)%5==0 or ep==0: 
            print(f"epoch {ep+1}/{args.epochs} loss={total/len(X):.4f}")

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/"distance.pt")

    joblib.dump(scaler,out/"scaler.joblib")

    save_json({"classes":enc.classes_.tolist(),"window_sec":10,"label_sec":600},out/"config.json")
    print("saved:",out)

if __name__=="__main__": main()