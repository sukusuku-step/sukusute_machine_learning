import numpy as np
import pandas as pd

BASE_COLUMNS = [
    "Steps",
    "Ax", "Ay", "Az",
    "Gx", "Gy", "Gz",
    "Mx", "My", "Mz"
]

# 歩数・加速度・相対距離のデータから特徴量抽出を行う関数
def extract_features(df):
    features = df[BASE_COLUMNS].copy()

    # 歩数の変化
    features["Step_Diff"] = (df["Steps"].diff().fillna(0))

    # 加速度の大きさ
    features["Acc_Mag"] = np.sqrt(df["Ax"] ** 2 + df["Ay"] ** 2 + df["Az"] ** 2)

    # 加速度変化
    features["Acc_Diff"] = (features["Acc_Mag"].diff().fillna(0))

    # ジャーク
    features["Jerk"] = (features["Acc_Diff"].diff().fillna(0))

    # ジャイロ
    features["Gyro_Mag"] = np.sqrt(df["Gx"] ** 2 + df["Gy"] ** 2 + df["Gz"] ** 2)

    # 地磁気
    features["Mag_Mag"] = np.sqrt(df["Mx"] ** 2 + df["My"] ** 2 + df["Mz"] ** 2)

    # 相対距離
    distance_cols = [c for c in df.columns if c.startswith("Distance_") and not c.endswith("_Label")]

    for col in distance_cols:
        features[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if distance_cols:
        distance = features[distance_cols]

        features["Distance_Min"] = distance.min(axis=1)
        features["Distance_Max"] = distance.max(axis=1)
        features["Distance_Mean"] = distance.mean(axis=1)
        features["Distance_Std"] = (distance.std(axis=1).fillna(0))
        features["Device_Count"] = (distance.notna().sum(axis=1))
        features["Near_Count_1m"] = (distance <= 1.0).sum(axis=1)
        features["Near_Count_2m"] = (distance <= 2.0).sum(axis=1)
        features["Near_Count_5m"] = (distance <= 5.0).sum(axis=1)

    features = features.replace([np.inf, -np.inf], np.nan)
    features = (features.ffill().bfill().fillna(0))

    # ここで抽出した特徴量をCNNへ入力することとなる
    return features.astype(np.float32), distance_cols