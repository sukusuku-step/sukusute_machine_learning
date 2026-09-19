from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import joblib

from utils.collector import *
from utils.add_features import *
from utils.deal_csv import *
from models import DistanceModel

from config import (
    WINDOW_LEN,
    N_WINDOWS,
    SEED,
    EPOCHS,
    LEARNING_RATE,
    CSV_PATH, 
    DISTANCE_MODEL_PATH, 
)

# 各相手との相対距離データからCNN+LSTMにより学習を行う
# 相対距離のラベル分類が推論結果となるモデル

# 学習データの順番をシャッフルする
np.random.seed(SEED); torch.manual_seed(SEED)

def main():
    # CSVデータを読み取るためのパスを作成
    paths = (
        glob.glob(os.path.join(CSV_PATH, "*.csv"))
        if os.path.isdir(CSV_PATH)
        else [CSV_PATH]
    )

    examples=[]
    labels=[]

    for path in paths:
        # CSVファイルを読み込む
        df = read_csv(path)

        # CSV内のDistance列を測定相手ごとに処理する
        for dc in numeric_distance_columns(df):
            lc = distance_label_col(dc)

            if lc not in df.columns:
                continue

            d = pd.to_numeric(df[dc], errors="coerce")
            temp = pd.DataFrame({"Timestamp": df["Timestamp"], dc: d, lc: df[lc]})

            # 10分単位にデータを分割する
            for _, seg in make_10min_segments(temp):
                # 入力データが6000行であるかどうかの確認
                if len(seg) != N_WINDOWS * WINDOW_LEN:
                    continue

                lab = label_from_segment(seg, lc)

                if not lab:
                    continue

                distance = seg[dc].to_numpy(dtype=np.float32)

                # 相対距離でNaN（タイムアウト）だった位置を記録
                is_nan = np.isnan(distance).astype(np.float32)

                # NaNそのものは扱えないためモデルには入れない
                distance = np.nan_to_num(distance, nan=0.0)

                # Distance + NaNフラグ（NaNをIsNaNフラグとしてモデルに渡す方式）
                arr = np.stack([distance, is_nan], axis=-1)
                arr = arr.reshape(N_WINDOWS, WINDOW_LEN, 2)

                examples.append(arr)
                labels.append(lab)

    if not examples: 
        raise ValueError("Distance_i + Distance_i_Label の学習データがありません。")

    # Numpy配列にまとめる
    X=np.stack(examples)

    # 各ラベルを番号に変換する
    enc=LabelEncoder().fit(labels)
    y=enc.transform(labels)

    # データの標準化
    scaler=fit_scaler(X)
    X=apply_scaler(X,scaler)

    # モデルを作成する（CNN⇒LSTM⇒Attention）
    model=DistanceModel(n_classes=len(enc.classes_))

    # 最適化
    opt=torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    loss_fn=nn.CrossEntropyLoss() # 損失関数
    model.train()

    # 学習を進める
    for ep in range(EPOCHS):
        order=np.random.permutation(len(X))
        total=0

        for i in order:
            xb=torch.tensor(X[i:i+1])
            yb=torch.tensor(y[i:i+1])

            opt.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);
            opt.step()

            total+=loss.item()

        if (ep+1)%5==0 or ep==0: 
            print(f"epoch {ep+1}/{EPOCHS} loss={total/len(X):.4f}")

    # できた学習済みモデル及びその設定ファイルの保存
    out=Path(DISTANCE_MODEL_PATH)
    out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/"distance.pt")
    joblib.dump(scaler,out/"scaler.joblib")

    save_json(
        {
            "classes":enc.classes_.tolist(),
            "window_sec":10,
            "label_sec":600
        },out/"config.json"
    )
    print("saved:",out)

if __name__=="__main__": main()