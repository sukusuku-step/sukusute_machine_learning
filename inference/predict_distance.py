from __future__ import annotations
from pathlib import Path
import numpy as np, torch, joblib, json

from models import DistanceModel
from utils.add_features import apply_scaler

from config import WINDOW_LEN, N_WINDOWS, DISTANCE_MODEL_PATH

# 学習済みモデルを使い、計測データからその推論（ラベル分類）を出力する

# ある相手1人に対する相対距離の10分間データからのラベル分類の推論（10分間は6000行）
# 入力は["Distance_N"]の行列（6000行×1列、NaNを含んでいてもよい）
def distance_infer(data):
    # data（ある相手1人に対する相対距離の10分間データ）をNumpy配列へ変換
    data = np.asarray(data, dtype=np.float32)

    # 入力データの形式の確認
    if data.ndim == 2:
        if data.shape[1] != 1:
            raise ValueError(
                "相対距離データの入力形式が違います。"
                f"現在のshape: {data.shape}"
            )
        data = data[:, 0]

    # 入力データのサンプル数の確認
    if len(data) != N_WINDOWS * WINDOW_LEN:
        raise ValueError(
            f"10分間のデータには{N_WINDOWS * WINDOW_LEN}サンプル必要です。"
            f"現在は{len(data)}サンプルです。"
        )

    # データの記録がNaNであった場所を記録
    is_nan = np.isnan(data).astype(np.float32)

    # モデルにはNaNを入力しないようにする
    data = np.nan_to_num(data, nan=0.0)

    # Distance + NaNフラグ（NaNはIsNaNとしてモデルに渡す方式）
    x = np.stack([data, is_nan], axis=-1)
    x = x.reshape(N_WINDOWS, WINDOW_LEN, 2)
    x = x[np.newaxis, ...]

    # 学習済みモデル・Scaler・設定ファイルのパスからモデルを読み込む
    d = Path(DISTANCE_MODEL_PATH)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))

    # 学習時と同じ標準化基準を使用
    scaler = joblib.load(d / "scaler.joblib")

    # 学習時と同じモデル構造を作成
    model = DistanceModel(n_classes=len(cfg["classes"]))

    # 学習済みパラメータを読み込む
    model.load_state_dict(torch.load(d / "distance.pt", map_location="cpu"))

    model.eval() # 推論モードにする

    # データを学習時と同じScalerで標準化する
    x = apply_scaler(x, scaler)

    # CNN + LSTMモデルへ入力する形にする
    x_tensor = torch.tensor(x)

    # 学習モデルによる推論を行う
    with torch.no_grad():
        output = model(x_tensor)

    # 相対距離のラベル分類の推論（10分毎）
    prob = output[0].softmax(-1)
    class_idx = int(prob.argmax())
    label = cfg["classes"][class_idx]
    confidence = float(prob[class_idx])

    # 推論結果を返す
    return {
        "label": label,
        "confidence": confidence
    }