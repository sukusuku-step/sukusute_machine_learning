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
    N_WINDOWS,
    WINDOW_LEN,
    SEGMENT_LEN,
    SEED,
    EPOCHS,
    LEARNING_RATE,
    CSV_PATH, 
    DISTANCE_MODEL_PATH
)

# 各相手との相対距離データからCNN+LSTMにより学習を行う
# 相対距離のラベル分類が推論結果となるモデル

# 学習データの順番をシャッフルする
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    # CSVデータを読み取るためのパスを作成
    paths = (
        # CSV_PATHで指定されたディレクトリ内にある.csvファイルを全て取得しリストに入れる
        glob.glob(os.path.join(CSV_PATH, "*.csv"))
        if os.path.isdir(CSV_PATH)
        else [CSV_PATH]
    )

    examples = []
    labels = []

    # 取得した各ファイルについて学習を行う
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
                if len(seg) != SEGMENT_LEN:
                    continue

                lab = label_from_segment(seg, lc)

                if not lab:
                    continue

                distance = seg[dc].to_numpy(dtype=np.float32)

                # NaNはそのまま保存しておく
                distance = distance.reshape(N_WINDOWS, WINDOW_LEN)

                examples.append(distance)
                labels.append(lab)

    if not examples: 
        raise ValueError("Distance_i + Distance_i_Label の学習データがありません。")

    # [サンプル数, 60, 100]
    distance_raw = np.stack(examples).astype(np.float32)

    # ラベル
    enc = LabelEncoder().fit(labels)
    y = enc.transform(labels)

    # NaNを除外してDistance専用の標準化基準を作る
    # 標準化後の0は「距離については中立値なので無視して、IsNaN=1を見てください」という意味
    scaler = fit_distance_scaler(distance_raw)

    # ch0 = 標準化Distance、ch1 = IsNaN
    # NaNはモデルには直接入れられないので、IsNaNフラグを用意して学習させる
    X = make_distance_features(distance_raw, scaler)

    if X.shape[-1] != 2:
        raise RuntimeError(
            f"Distance input must have 2 channels, got {X.shape}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Preprocessed input contains NaN or inf"
        )

    # モデルを作成する（CNN ⇒ LSTM ⇒ Attention）
    model = DistanceModel(in_channels=2, n_classes=len(enc.classes_))

    # 最適化
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss() # 損失関数
    model.train()

    # 学習を進める
    for ep in range(EPOCHS):
        order = np.random.permutation(len(X))
        total = 0

        for i in order:
            xb = torch.tensor(X[i:i+1])
            yb = torch.tensor(y[i:i+1])

            opt.zero_grad()
            loss = loss_fn(model(xb),yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total+=loss.item()

        if (ep+1)%5 == 0 or ep == 0: 
            print(f"epoch {ep+1} / {EPOCHS} loss = {total/len(X):.4f}")

    # できた学習済みモデル及びその設定ファイルの保存
    out = Path(DISTANCE_MODEL_PATH)
    out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(), out/"distance.pt")
    joblib.dump(scaler, out/"scaler.joblib")

    save_json(
        {
            "classes":enc.classes_.tolist(),
            "window_sec":10,
            "label_sec":600
        },out/"config.json"
    )
    print("saved:",out)

if __name__== "__main__": 
    main()