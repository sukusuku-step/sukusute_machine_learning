from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch

from models import BehaviorActivityModel

from utils.add_features import add_engineered_features, apply_scaler
from utils.deal_csv import split_into_windows

from config import (
    WINDOW_LEN,
    N_WINDOWS,
    SEGMENT_LEN,
    BEHAVIOR_MODEL_PATH,
    BASE_SENSOR_COLS
)

# ======================================================================
# 学習済みモデルを使い、歩数・加速度データからその推論（ラベル分類）を出力する
#
# 推論時の入力列:
#
# Steps
# Ax, Ay, Az
# Gx, Gy, Gz
# Mx, My, Mz
#
# 歩数・加速度の10分間データからのラベル分類の推論（10分間は6000行）
# 入力は6000行（10分）×10列（10種類のカラム）の行列の形式

# Timestampは入力不要
# 0.1秒周期で連続取得されたデータであることを前提として、
# 内部で0.1秒刻みのTimestampを生成する
# ======================================================================

# 生のセンサデータから入力用のDataFrameを作成するための関数
def _make_dataframe(data):
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(
            "歩数・加速度データは2次元配列で入力してください。"
            f" 現在のshape: {data.shape}"
        )

    if data.shape[1] != len(BASE_SENSOR_COLS):
        raise ValueError(
            f"入力データは{len(BASE_SENSOR_COLS)}列必要です。"
            f" 現在は{data.shape[1]}列です。"
        )

    # 元の計測データだけでDataFrameを作る
    df = pd.DataFrame(data, columns=BASE_SENSOR_COLS)

    # add_engineered_features()内のJerk計算にTimestampが必要なため内部生成する（0.1秒周期）
    df.insert(0, "Timestamp", np.arange(len(df), dtype=np.float64) * 0.1)

    return df

# 学習済みBehaviorActivityModel、Scaler、configを読み込む
def _load_model():
    model_dir = Path(BEHAVIOR_MODEL_PATH)

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    scaler = joblib.load(model_dir / "scaler.joblib")

    model = BehaviorActivityModel(
        len(config["feature_cols"]),
        len(config["pedo_classes"]),
        len(config["acce_classes"])
    )

    model.load_state_dict(
        torch.load(
            model_dir / "behavior_activity.pt",
            map_location="cpu"
        )
    )

    model.eval()

    return model, scaler, config

# 10分間の歩数・加速度データからPedo_Label / Acce_Labelを推論する
def behavior_infer(data):
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(
            "歩数・加速度データの入力形式が違います。"
            f" 現在のshape: {data.shape}"
        )

    if data.shape[1] != len(BASE_SENSOR_COLS):
        raise ValueError(
            f"入力データは{len(BASE_SENSOR_COLS)}列必要です。"
            f" 現在は{data.shape[1]}列です。"
        )

    if data.shape[0] != SEGMENT_LEN:
        raise ValueError(
            f"10分間のデータには{SEGMENT_LEN}サンプル必要です。"
            f" 現在は{data.shape[0]}サンプルです。"
        )

    # DataFrameへ変換しTimestampを生成
    df = _make_dataframe(data)

    # 学習済みモデル等を読み込む
    model, scaler, config = (_load_model())

    # 学習時と同じ補助特徴量を追加で生成
    df, feature_cols = (add_engineered_features(df))

    # 学習時と推論時の特徴量が同一かを確認
    if feature_cols != config["feature_cols"]:
        raise ValueError(
            "推論時の特徴量が"
            "学習時と一致していません。"
        )

    # 6000行
    # ↓
    # [60, 100, C]
    windows = split_into_windows(df, feature_cols)

    if len(windows) != N_WINDOWS:
        raise RuntimeError(
            f"10秒窓が{N_WINDOWS}個必要ですが、"
            f"{len(windows)}個生成されました。"
        )

    # 学習時に保存したScalerを使用して標準化
    x = apply_scaler(windows.astype(np.float32), scaler)

    # BehaviorActivityModel入力:
    #
    # [B, S, 60, 100, C]
    #
    # B = 1
    # S = 1
    x_tensor = (torch.from_numpy(x).unsqueeze(0).unsqueeze(0))

    with torch.no_grad():
        pedo_out, acce_out, _ = model(x_tensor)

    # [1, 1, n_pedo]
    pedo_prob = torch.softmax(pedo_out[0, 0], dim=-1)
    pedo_idx = int(pedo_prob.argmax())
    pedo_label = (config["pedo_classes"][pedo_idx])
    pedo_confidence = float(pedo_prob[pedo_idx])

    # [1, 1, n_acce]
    acce_prob = torch.softmax(acce_out[0, 0], dim=-1)
    acce_idx = int(acce_prob.argmax())
    acce_label = (config["acce_classes"][acce_idx])
    acce_confidence = float(acce_prob[acce_idx])

    # 推論結果を出力
    return {
        "pedo_label": pedo_label,
        "pedo_confidence": pedo_confidence,
        "acce_label": acce_label,
        "acce_confidence": acce_confidence
    }

# 任意時間の歩数・加速度データからactivity_levelを推論する
def activity_infer(data):
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(
            "歩数・加速度データの入力形式が違います。"
            f" 現在のshape: {data.shape}"
        )

    if data.shape[1] != len(BASE_SENSOR_COLS):
        raise ValueError(
            f"入力データは{len(BASE_SENSOR_COLS)}列必要です。"
            f" 現在は{data.shape[1]}列です。"
        )

    if len(data) < SEGMENT_LEN:
        raise ValueError(
            f"活動量の推論には最低{SEGMENT_LEN}サンプル必要です。"
            f" 現在は{len(data)}サンプルです。"
        )

    # DataFrameへ変換
    df = _make_dataframe(data)

    # モデル等を読み込む
    model, scaler, config = (_load_model())

    # 学習時と同じ補助特徴量を追加で生成
    df, feature_cols = (add_engineered_features(df))

    if feature_cols != config["feature_cols"]:
        raise ValueError(
            "推論時の特徴量が"
            "学習時と一致していません。"
        )

    # 10秒単位へ分割
    #
    # [N,C]
    # ↓
    # [10秒窓数,100,C]
    windows = split_into_windows(df,feature_cols)

    # 完全な10分区間数
    n_segments = (len(windows) // N_WINDOWS)

    if n_segments == 0:
        raise ValueError(
            "10分間のデータが"
            "1区間も作成できません。"
        )

    # 最後の10分未満のデータは切り捨て
    windows = windows[:n_segments * N_WINDOWS]

    # [S,60,100,C]
    windows = windows.reshape(
        n_segments,
        N_WINDOWS,
        WINDOW_LEN,
        len(feature_cols)
    )

    # 学習時と同じScalerを使用して標準化
    x = apply_scaler(windows.astype(np.float32), scaler)

    # [1,S,60,100,C]
    x_tensor = (torch.from_numpy(x).unsqueeze(0))

    with torch.no_grad():
        _, _, activity_out = model(x_tensor)

    activity_prob = torch.softmax(activity_out[0], dim=-1)
    activity_idx = int(activity_prob.argmax())
    activity_classes = config.get(
        "activity_classes",
        [1, 2, 3, 4, 5]
    )

    activity_level = (activity_classes[activity_idx])
    activity_confidence = float(activity_prob[activity_idx])

    # 推論結果を出力
    return {
        "activity_level": activity_level,
        "activity_confidence": activity_confidence
    }