from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import joblib

from utils.collector import *
from utils.add_features import *
from utils.deal_csv import *
from models.behavior_classifier import BehaviorActivityModel

from config import (
    N_WINDOWS,
    WINDOW_LEN,
    SEGMENT_LEN,
    SEED,
    EPOCHS,
    LEARNING_RATE,
    CSV_PATH, 
    BEHAVIOR_MODEL_PATH
)

# 歩数・加速度データからCNN+LSTMにより学習を行う
# 歩数・加速度のラベル分類と活動量の値の推測が推論結果となるモデル

# 学習データの順番をシャッフルする
torch.manual_seed(SEED)
np.random.seed(SEED)

# CSVを読み込んで、歩数・加速度の10分単位の学習サンプルを作る関数
def load_sessions(paths):
    sessions = []

    for path in paths:
        df = read_csv(path) # CSVを読み込む
        df, fc = add_engineered_features(df) # 特徴量を追加
        act = activity_from_csv(df) # 活動量の値を取得

        if act is None or not (1 <= act <= 5):
            continue

        blocks = []

        # 10分単位の学習サンプルへ分割し、それを10秒単位へ分割
        for _, seg in make_10min_segments(df):
            w = split_into_windows(seg, fc)

            # 各学習サンプルのラベルを取得
            p = label_from_segment(seg, "Pedo_Label")
            a = label_from_segment(seg, "Acce_Label")

            if len(w) == N_WINDOWS and p and a:
                blocks.append((w,p,a))

        if blocks:
            x = np.stack([b[0] for b in blocks]).astype(np.float32)
            ps = [b[1] for b in blocks]
            acs = [b[2] for b in blocks]

            sessions.append((x, ps, acs, int(round(act))))

    return sessions, fc

def main():
    # CSVデータを読み取るためのパスを作成
    paths = (
        # CSV_PATHで指定されたディレクトリ内にある.csvファイルを全て取得しリストに入れる
        glob.glob(os.path.join(CSV_PATH, "*.csv"))
        if os.path.isdir(CSV_PATH)
        else [CSV_PATH]
    )

    if not paths:
        raise ValueError(
            f"CSVファイルが見つかりません: {CSV_PATH}"
        )

    # pathsリストの長さ = 学習データのファイル数
    if len(paths) < 2: 
        raise ValueError("汎用モデル評価のためCSVは複数ファイル推奨です。")

    # 時間分割した学習サンプルを取得
    train, fc = load_sessions(paths)

    if not train: 
        raise ValueError(
            f"学習に使えるデータがありません。"
            f"各CSVは最低{SEGMENT_LEN}行以上必要です。"
        )
    
    # 学習データから標準化の基準を設定
    scaler = fit_scaler(
        np.concatenate([s[0].reshape(-1, WINDOW_LEN, len(fc)) for s in train], axis=0)[None,...].reshape(-1, WINDOW_LEN, len(fc))
    )

    # ラベルは学習データに存在するものだけをクラス化
    pedo_classes = sorted(set(v for s in train for v in s[1]))
    acce_classes = sorted(set(v for s in train for v in s[2]))

    # 各ラベルを番号に変換する
    pmap = {v:i for i,v in enumerate(pedo_classes)}
    amap = {v:i for i,v in enumerate(acce_classes)}

    # モデルを作成する（CNN⇒BiLSTM⇒Attention）
    model = BehaviorActivityModel(len(fc), len(pedo_classes), len(acce_classes))

    # 最適化
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # 損失関数
    ce_p = nn.CrossEntropyLoss()
    ce_a = nn.CrossEntropyLoss()
    ce_act = nn.CrossEntropyLoss()

    # 学習を進める
    for ep in range(EPOCHS):
        order = np.random.permutation(len(train))
        total = 0
        model.train()

        for idx in order:
            x, ps, acs, act = train[idx]

            # セッション全体を一度に入力する（S=10分のサンプル）
            x = apply_scaler(x, scaler)
            xt = torch.tensor(x).unsqueeze(0)
            pm = torch.tensor([[pmap[v] for v in ps]])
            am = torch.tensor([[amap[v] for v in acs]])
            ym = torch.tensor([act-1], dtype=torch.long)
            opt.zero_grad()

            # 歩数・加速度・活動量を学習する
            po, ao, actlogit=model(xt)
            loss_p = ce_p(po.reshape(-1,len(pedo_classes)), pm.reshape(-1))
            loss_a = ce_a(ao.reshape(-1,len(acce_classes)), am.reshape(-1))
            loss_act = ce_act(actlogit, ym)

            # 行動/姿勢を主タスク、activity_levelを補助タスクとして学習
            loss = loss_p+loss_a + 0.5 * loss_act
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total+=loss.item()

        if (ep+1)%5 == 0 or ep == 0:
            print(f"epoch {ep+1} / {EPOCHS} loss = {total/len(order):.4f}")

    # できた学習済みモデル及びその設定ファイルの保存
    out = Path(BEHAVIOR_MODEL_PATH)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "behavior_activity.pt")
    joblib.dump(scaler, out/"scaler.joblib")

    save_json(
        {
            "feature_cols":fc,
            "pedo_classes":pedo_classes,
            "acce_classes":acce_classes,
            "sample_hz":10,
            "window_sec":10,
            "label_sec":600,
            "activity_classes":[1,2,3,4,5],
            "activity_note":"activity_level is one label for the whole CSV/session; inference updates it from accumulated 10-minute blocks."
        },
        out/"config.json"
    )
    print("saved:",out)

if __name__=="__main__": 
    main()