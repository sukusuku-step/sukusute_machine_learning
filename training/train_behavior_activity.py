from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import joblib

from utils.collector import *
from utils.add_features import *
from utils.deal_csv import *
from models import BehaviorActivityModel

from config import N_WINDOWS, SEED

# 歩数・加速度データからCNN+LSTMにより学習を行う
# 歩数・加速度のラベル分類と活動量の値の推測が推論結果となるモデル

# 学習データの順番をシャッフルする
torch.manual_seed(SEED); np.random.seed(SEED)

def load_sessions(paths):
    sessions=[]

    for path in paths:
        df=read_csv(path)
        df,fc=add_engineered_features(df)
        act=activity_from_csv(df)

        if act is None or not (1<=act<=5): 
            continue

        blocks=[]

        for _,seg in make_10min_segments(df):
            w=split_into_windows(seg,fc)
            p=label_from_segment(seg,"Pedo_Label")
            a=label_from_segment(seg,"Acce_Label")
            if len(w)==N_WINDOWS and p and a:
                blocks.append((w,p,a))

        if blocks:
            x=np.stack([b[0] for b in blocks]).astype(np.float32)
            ps=[b[1] for b in blocks]
            acs=[b[2] for b in blocks]
            sessions.append((x,ps,acs,int(round(act))))

    return sessions,fc

def split_paths(paths):
    rng=np.random.default_rng(SEED); paths=list(paths); rng.shuffle(paths)
    n=max(1,int(len(paths)*0.2))

    return paths[n:],paths[:n]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--out",default="artifacts/behavior")
    ap.add_argument("--epochs",type=int,default=30)
    args=ap.parse_args()
    paths=glob.glob(os.path.join(args.data,"*.csv")) if os.path.isdir(args.data) else [args.data]

    if len(paths) < 2: 
        raise ValueError("汎用モデル評価のためCSVは複数ファイル推奨です。")
    
    train_paths,test_paths=split_paths(paths)
    train,fc=load_sessions(train_paths)
    test,_=load_sessions(test_paths)

    if not train: 
        raise ValueError("学習に使えるデータがありません。")
    
    # scalerは学習側だけでfitさせる
    scaler=fit_scaler(np.concatenate([s[0].reshape(-1,WINDOW_LEN,len(fc)) for s in train],axis=0)[None,...].reshape(-1,WINDOW_LEN,len(fc)))

    # ラベルは学習データに存在するものだけをクラス化
    pedo_classes=sorted(set(v for s in train for v in s[1]))
    acce_classes=sorted(set(v for s in train for v in s[2]))
    pmap={v:i for i,v in enumerate(pedo_classes)}
    amap={v:i for i,v in enumerate(acce_classes)}

    model=BehaviorActivityModel(len(fc),len(pedo_classes),len(acce_classes))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    ce_p=nn.CrossEntropyLoss(); ce_a=nn.CrossEntropyLoss(); ce_act=nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        order=np.random.permutation(len(train)); total=0
        model.train()

        for idx in order:
            x,ps,acs,act=train[idx]
            # セッション全体を一度に入力する（S=10分のサンプル）
            x=apply_scaler(scaler,x)
            xt=torch.tensor(x).unsqueeze(0)
            pm=torch.tensor([[pmap[v] for v in ps]])
            am=torch.tensor([[amap[v] for v in acs]])
            ym=torch.tensor([act-1])
            opt.zero_grad()
            po,ao,actlogit=model(xt)
            loss_p=ce_p(po.reshape(-1,len(pedo_classes)),pm.reshape(-1))
            loss_a=ce_a(ao.reshape(-1,len(acce_classes)),am.reshape(-1))
            loss_act=ce_act(actlogit.unsqueeze(0),ym)
            # 行動/姿勢を主タスク、activity_levelを補助タスクとして学習
            loss=loss_p+loss_a+0.5*loss_act
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            total+=loss.item()

        if (ep+1)%5==0 or ep==0:
            print(f"epoch {ep+1}/{args.epochs} loss={total/len(order):.4f}")

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/"behavior_activity.pt")

    joblib.dump(scaler,out/"scaler.joblib")

    save_json({"feature_cols":fc,"pedo_classes":pedo_classes,"acce_classes":acce_classes,
            "sample_hz":10,"window_sec":10,"label_sec":600,
            "activity_classes":[1,2,3,4,5],
            "activity_note":"activity_level is one label for the whole CSV/session; inference updates it from accumulated 10-minute blocks."},out/"config.json")
    print("saved:",out)

    if test:
        print(f"held-out files: {len(test)} (evaluation implementation can be added after subject-wise split is fixed)")

if __name__=="__main__": main()