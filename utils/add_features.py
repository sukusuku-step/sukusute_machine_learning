import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import BASE_SENSOR_COLS

# 生データから補助特徴量を算出する関数（CNNには生データ+補助特徴量を同時入力する）
def add_engineered_features(df):
    x = df.copy()

    steps = pd.to_numeric(x["Steps"], errors="coerce").interpolate().ffill().bfill()
    x["Step_Diff"] = steps.diff().fillna(0).clip(lower=0) # 累計歩数から計算する歩数の変化量

    ax = pd.to_numeric(x["Ax"], errors="coerce")
    ay = pd.to_numeric(x["Ay"], errors="coerce")
    az = pd.to_numeric(x["Az"], errors="coerce")
    gx = pd.to_numeric(x["Gx"], errors="coerce")
    gy = pd.to_numeric(x["Gy"], errors="coerce")
    gz = pd.to_numeric(x["Gz"], errors="coerce")
    mx = pd.to_numeric(x["Mx"], errors="coerce")
    my = pd.to_numeric(x["My"], errors="coerce")
    mz = pd.to_numeric(x["Mz"], errors="coerce")

    # 加速度の大きさ（スカラー）を計算
    x["Acc_Mag"] = np.sqrt(ax**2 + ay**2 + az**2)
    x["Gyro_Mag"] = np.sqrt(gx**2 + gy**2 + gz**2)
    x["Mag_Mag"] = np.sqrt(mx**2 + my**2 + mz**2)

    # 加速度変化量（Acc_Diff）と加速度変化率（Jerk）を計算
    acc_diff = x["Acc_Mag"].diff()
    x["Acc_Diff"] = acc_diff.fillna(0) 
    dt = pd.to_numeric(x["Timestamp"], errors="coerce").diff()
    x["Jerk"] = (acc_diff / dt).replace([np.inf, -np.inf], np.nan).fillna(0)

    # 生データ + 補助特徴量 のリストを作る
    feature_cols = BASE_SENSOR_COLS + ["Step_Diff","Acc_Mag","Gyro_Mag","Mag_Mag","Acc_Diff","Jerk"]

    x[feature_cols] = x[feature_cols].apply(pd.to_numeric, errors="coerce")
    x[feature_cols] = x[feature_cols].interpolate().ffill().bfill().fillna(0)

    return x, feature_cols

# 生データの平均・標準偏差を使って標準化の基準を作る
def fit_scaler(arr):
    scaler = StandardScaler()
    scaler.fit(arr.reshape(-1, arr.shape[-1]))

    return scaler

# 標準化の基準に従ってデータの標準化を行う
def apply_scaler(arr, scaler):
    shape = arr.shape
    standardized_shape = scaler.transform(arr.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    return standardized_shape