from __future__ import annotations
import numpy as np
import pandas as pd

from sukusute_machine_learning.utils.deal_csv import make_10min_segments
from sukusute_machine_learning.utils.add_features import add_engineered_features

from sukusute_machine_learning.config import SAMPLE_HZ, MIN_HISTORY_HOURS, BASE_SENSOR_COLS

MAD_EPS=1e-6 # ゼロ除算防止用の微小値

# 過去データから個人チューニングした異常検知用のベースライン（しきい値）を算出する

# ある人の普段の特徴量を統計的にまとめる処理
def robust_stats(values):
    x = np.asarray(values, dtype=np.float32)

    median = np.nanmedian(x) # 中央値を算出
    mad = np.nanmedian(np.abs(x - median))
    mad_scale = max(1.4826 * mad, MAD_EPS)

    return float(median), float(mad_scale)

# ==================================================================================
# 個人チューニングしたベースラインを過去データから算出する
#
# 入力:
# [N, 10]の「任意の行数×10列」の行列形式
#
# 列順:
# Steps
# Ax, Ay, Az
# Gx, Gy, Gz
# Mx, My, Mz
#
# Timestampは入力不要（0.1秒周期で連続取得されたデータであることを前提として内部生成する）
# 行数に関して、ベースラインの算出に必要な分の過去データが溜まっていなければ受け付けない
# ==================================================================================

def build_baseline(data):
    # data（過去データの行列）をNumpy配列へ変換
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2 or data.shape[1] != 10:
        raise ValueError(
            "入力データの形式が違います。"
            f"現在のshape: {data.shape}"
        )
    
    if data.shape[1] != len(BASE_SENSOR_COLS):
        raise ValueError(
            f"入力データは{len(BASE_SENSOR_COLS)}列必要です。"
            f" 現在は{data.shape[1]}列です。"
        )

    # ベースライン算出に必要な分の履歴が溜まっていなければ即returnとする
    if len(data) < MIN_HISTORY_HOURS * 60 * 60 * SAMPLE_HZ:
        return None

    # Steps + Ax～Mz からDataFrameを作成
    df = pd.DataFrame(data, columns=BASE_SENSOR_COLS)

    # Timestampを内部生成
    df.insert(0, "Timestamp", np.arange(len(df), dtype=np.float64) / SAMPLE_HZ)

    # ベースライン算出に使う特徴量を追加する
    df, _ = add_engineered_features(df)

    # データを10分単位に分割（余りは切り捨てる）
    segments = make_10min_segments(df)

    if not segments:
        return None

    rows = []

    # 各10分間の特徴量を計算する
    for _, seg in segments:
        # 累計歩数の変化量の総和
        steps = pd.to_numeric(
            seg["Step_Diff"], 
            errors="coerce"
        ).fillna(0)

        # 加速度の大きさ
        acc = pd.to_numeric(
            seg["Acc_Mag"], 
            errors="coerce"
        )

        # ジャイロ（角速度）の大きさ
        gyro = pd.to_numeric(
            seg["Gyro_Mag"], 
            errors="coerce"
        )

        # 地磁気の大きさ
        mag = pd.to_numeric(
            seg["Mag_Mag"], 
            errors="coerce"
        )

        rows.append({
            "steps_10min": float(steps.sum()), # 累計歩数の変化量の総和
            "activity_mean_proxy": float(acc.mean()), # 加速度の大きさの平均
            "acc_std": float(acc.std()), # 加速度の大きさの標準偏差
            "gyro_mean": float(gyro.mean()), # ジャイロの大きさの平均
            "mag_mean": float(mag.mean()) # 地磁気の大きさの平均
        })

    # 10分ごとの結果をまとめる
    summary = pd.DataFrame(rows)

    features = [
        "steps_10min",
        "activity_mean_proxy",
        "acc_std",
        "gyro_mean",
        "mag_mean"
    ]

    baseline = {"features": {}}

    # 個人チューニングした「普段」の指標を算出する
    for feature in features:
        median, mad_scale = robust_stats(summary[feature])

        baseline["features"][feature] = {
            "median": median, # 普段の値
            "mad_scale": mad_scale # 普段の標準偏差（ばらつき）
        }

    # 最終的に算出したベースラインを返す
    # featuresの中の5つの特徴量に関して「普段の10分間での値とばらつき」を算出する
    return baseline
