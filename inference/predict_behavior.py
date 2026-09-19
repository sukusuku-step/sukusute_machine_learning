from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib, json

from models import BehaviorActivityModel
import utils.add_features

from config import WINDOW_LEN, N_WINDOWS, BEHAVIOR_MODEL_PATH

# 学習済みモデルを使い、計測データからその推論（ラベル分類）を出力する

# 歩数・加速度の10分間データからのラベル分類の推論（10分間は6000行）
# 入力は["Steps", "Ax", "Ay", "Az","Gx", "Gy", "Gz", "Mx", "My", "Mz"]の行列（6000行×10列）
def behavior_infer(data):
    # data（歩数・加速度の10分間データ）をNumpy配列へ変換
    data = np.asarray(data, dtype=np.float32)

    # 入力する引数の形式の確認
    if data.ndim != 2:
        raise ValueError(
            "歩数・加速度データの入力形式が違います。"
            f"現在のshape: {data.shape}"
        )

    # 入力データのサンプル数の確認
    if data.shape[0] != WINDOW_LEN * N_WINDOWS:
        raise ValueError(
            f"10分間のデータには{WINDOW_LEN * N_WINDOWS}サンプル必要です。"
            f"現在は{data.shape[0]}サンプルです。"
        )
    
    # 学習済みモデル・Scaler・設定ファイルのパスからモデルを読み込む
    d = Path(BEHAVIOR_MODEL_PATH)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))

    # 学習時と同じ標準化基準を使用
    scaler = joblib.load(d / "scaler.joblib")

    # 学習時と同じモデル構造を作成
    model = BehaviorActivityModel(
        len(cfg["feature_cols"]),
        len(cfg["pedo_classes"]),
        len(cfg["acce_classes"])
    )

    # 学習済みパラメータを読み込む
    model.load_state_dict(torch.load(d / "behavior_activity.pt", map_location="cpu"))

    model.eval() # 推論モードにする

    base_cols = [
        "Steps",
        "Ax", "Ay", "Az",
        "Gx", "Gy", "Gz",
        "Mx", "My", "Mz"
    ]

    # 入力データのカラム数の確認
    if data.shape[1] != len(base_cols):
        raise ValueError(
            f"入力データは{len(base_cols)}列の形式を要求しています。"
            f"現在は{data.shape[1]}列です。"
        )

    # 入力のNumPy配列からDataFrameを作成する
    df = pd.DataFrame(data, columns=base_cols)

    # 学習時と同じ補助特徴量を追加する
    df, feature_cols = (utils.add_features.add_engineered_features(df))

    # 学習時と推論時の特徴量が一致しているかを確認
    if feature_cols != cfg["feature_cols"]:
        raise ValueError(
            "推論時の特徴量が学習時と一致していません。"
        )

    # 10分間を10秒単位に分割する
    windows = utils.add_features.split_into_windows(df, feature_cols)

    # 分割したデータを学習時と同じScalerで標準化する
    x = utils.add_features.apply_scaler(windows.astype(np.float32), scaler)

    # CNN + LSTMモデルへ入力する形にする
    x_tensor = torch.tensor(x).unsqueeze(0)

    # 学習モデルによる推論を行う
    with torch.no_grad():
        pedo_out, acce_out, activity_out = model(x_tensor)

    # 歩数データのラベル分類の推論（10分毎）
    pedo_prob = pedo_out[0].softmax(-1)
    pedo_idx = int(pedo_prob.argmax())
    pedo_label = cfg["pedo_classes"][pedo_idx]
    pedo_confidence = float(pedo_prob[pedo_idx])

    # 加速度データのラベル分類の推論（10分毎）
    acce_prob = acce_out[0].softmax(-1)
    acce_idx = int(acce_prob.argmax())
    acce_label = cfg["acce_classes"][acce_idx]
    acce_confidence = float(acce_prob[acce_idx])

    # 推論結果を返す
    return {
        "pedo_label": pedo_label,
        "pedo_confidence": pedo_confidence,
        "acce_label": acce_label,
        "acce_confidence": acce_confidence,
    }

# 任意の長さ（1日分）の歩数・加速度データからの活動量の値の推論
# 入力は["Steps", "Ax", "Ay", "Az","Gx", "Gy", "Gz", "Mx", "My", "Mz"]の行列（任意の行数×10列）
def activity_infer(data):
    # data（1日分の歩数・加速度データ）をNumpy配列へ変換
    data = np.asarray(data, dtype=np.float32)

    # 入力する引数の形式の確認
    if data.ndim != 2:
        raise ValueError(
            "歩数・加速度データの入力形式が違います。"
            f"現在のshape: {data.shape}"
        )

    base_cols = [
        "Steps",
        "Ax", "Ay", "Az",
        "Gx", "Gy", "Gz",
        "Mx", "My", "Mz"
    ]

    # 入力データのカラム数の確認
    if data.shape[1] != len(base_cols):
        raise ValueError(
            f"入力データは{len(base_cols)}列の形式を要求しています。"
            f"現在は{data.shape[1]}列です。"
        )

    # 活動量の値の推論には最低10分間のデータを必要とする
    if len(data) < N_WINDOWS * 100:
        raise ValueError(
            f"活動量の値の推論には最低{N_WINDOWS * 100}サンプル必要です。"
            f"現在は{len(data)}サンプルです。"
        )

    # 学習済みモデル・Scaler・設定ファイルを読み込む
    d = Path(BEHAVIOR_MODEL_PATH)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))

    # 学習時と同じ標準化基準を使用
    scaler = joblib.load(d / "scaler.joblib")

    # 学習時と同じモデル構造を作成
    model = BehaviorActivityModel(
        len(cfg["feature_cols"]),
        len(cfg["pedo_classes"]),
        len(cfg["acce_classes"])
    )

    # 学習済みパラメータを読み込む
    model.load_state_dict(torch.load(d / "behavior_activity.pt", map_location="cpu"))

    model.eval() # 推論モードにする

    # 入力のNumPy配列からDataFrameを作成する
    df = pd.DataFrame(data, columns=base_cols)

    # 学習時と同じ補助特徴量を追加する
    df, feature_cols = (utils.add_features.add_engineered_features(df))

    # 学習時と推論時の特徴量が一致しているかを確認
    if feature_cols != cfg["feature_cols"]:
        raise ValueError(
            "推論時の特徴量が学習時と一致していません。"
        )

    # 10分間を10秒単位に分割する
    windows = utils.add_features.split_into_windows(df, feature_cols)

    # 10分単位に変換できる部分だけ使用
    n_segments = len(windows) // N_WINDOWS
    if n_segments == 0:
        raise ValueError(
            "10分間のデータが1区間も作成できません。"
        )
    windows = windows[:n_segments * N_WINDOWS] # 端数の10秒は切り捨てる
    windows = windows.reshape(n_segments, N_WINDOWS, windows.shape[1], windows.shape[2])

    # 分割したデータを学習時と同じScalerで標準化する
    x = utils.add_features.apply_scaler(windows.astype(np.float32), scaler)

    # CNN + LSTMモデルへ入力する形にする
    x_tensor = torch.tensor(x).unsqueeze(0)

    # 学習モデルによる推論を行う
    with torch.no_grad():
        _, _, activity_out = model(x_tensor)

    # データ全体に対する活動量の値の推論
    activity_prob = activity_out[0].softmax(-1)
    activity = int(activity_prob.argmax()) + 1
    activity_confidence = float(activity_prob[activity - 1])

    # 推論結果を返す
    return {
        "activity_level": activity,
        "activity_confidence": activity_confidence
    }