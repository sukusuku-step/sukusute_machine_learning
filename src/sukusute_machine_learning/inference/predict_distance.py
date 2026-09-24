from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import torch

from models import DistanceModel

from utils.add_features import make_distance_features

from config import (
    WINDOW_LEN,
    N_WINDOWS,
    SEGMENT_LEN,
    DISTANCE_MODEL_PATH
)

# =========================================================================================
# 学習済みDistanceモデルを使い、ある1つのDistance_Nデータからその推論（相対距離ラベル）を出力する
#
# 入力:
#   [6000]
#
# または
#
#   [6000, 1]
#
# ある相手1人に対する相対距離の10分間データからのラベル分類の推論（10分間は6000行）
# 入力は6000行（10分）×1列（1種類の相対距離カラム）の行列の形式
#
# 入力データの中にはNaNを含んでいてもよい
# =========================================================================================

# 1つのDistance_Nについて、10分間の距離時系列からラベルを推論する
def distance_infer(data):
    data = np.asarray(data, dtype=np.float32)

    # [6000,1]の場合は[6000]へ入力データを変換する
    if data.ndim == 2:
        if data.shape[1] != 1:
            raise ValueError(
                "相対距離データは"
                "[6000]または[6000,1]"
                "形式で入力してください。"
                f" 現在のshape: {data.shape}"
            )
        data = data[:, 0]

    elif data.ndim != 1:
        raise ValueError(
            "相対距離データは"
            "[6000]または[6000,1]"
            "形式で入力してください。"
            f" 現在のshape: {data.shape}"
        )
    
    if len(data) != SEGMENT_LEN:
        raise ValueError(
            f"10分間のデータには{SEGMENT_LEN}サンプル必要です。"
            f" 現在は{len(data)}サンプルです。"
        )

    # 学習済みモデル・Scaler・設定ファイルを読み込む
    model_dir = Path(DISTANCE_MODEL_PATH)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    # Distance専用Scaler（学習時にNaNを除外してfitされたもの）
    scaler = joblib.load(model_dir / "scaler.joblib")

    # 学習時と同じ2chモデルにする
    model = DistanceModel(in_channels=2, n_classes=len(config["classes"]))

    model.load_state_dict(
        torch.load(
            model_dir / "distance.pt",
            map_location="cpu"
        )
    )

    model.eval()

    # --------------------------------------------------------
    # 学習時と同じDistance前処理を行う
    #
    # 生Distance:
    #
    # [6000]
    #
    # ↓
    #
    # make_distance_features()
    #
    # ↓
    #
    # [6000,2]
    #
    # ch0 = 標準化Distance
    # ch1 = IsNaN
    #
    # NaN位置:
    #
    # Distance_scaled = 0
    # IsNaN           = 1
    #
    # 通常位置:
    #
    # Distance_scaled = 標準化値
    # IsNaN           = 0
    # --------------------------------------------------------

    x = make_distance_features(data, scaler)

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Distance前処理後に"
            "NaNまたはinfが残っています。"
        )

    # [6000,2]
    # ↓
    # [60,100,2]
    x = x.reshape(N_WINDOWS, WINDOW_LEN, 2)

    # Batch次元追加
    #
    # [1,60,100,2]
    x_tensor = (torch.from_numpy(x).unsqueeze(0))

    # 推論を行う
    with torch.no_grad():
        output = model(x_tensor)

    prob = torch.softmax(output[0], dim=-1)
    class_idx = int(prob.argmax())
    label = (config["classes"][class_idx])
    confidence = float(prob[class_idx])

    # 推論結果を出力
    return {
        "label": label,
        "confidence": confidence
    }