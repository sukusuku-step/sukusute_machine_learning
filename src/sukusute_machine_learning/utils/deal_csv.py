from __future__ import annotations
import re
import numpy as np
import pandas as pd

from sukusute_machine_learning.config import LABEL_SEC, WINDOW_LEN, SEGMENT_LEN

# 生データ（CSV）からTimestampを取得する関数（0.1刻み）
def get_time(df):
    t = pd.to_numeric(df["Timestamp"], errors="coerce")

    # 0.1刻みであるかを一応確認する
    if t.notna().sum() < len(df) * 0.8:
        raise ValueError("Timestamp must be elapsed seconds such as 0.0, 0.1, 0.2 ...")
    
    return t.to_numpy()

# 生データをラベル付けの10分単位で分割する
def make_10min_segments(df):
    t = get_time(df)

    work = df.copy()
    work["_t"] = t
    work["_segment"] = np.floor(work["_t"] / LABEL_SEC).astype(int)

    segments = []

    for sid, g in work.groupby("_segment", sort=True):
        g = g.drop(columns=["_t", "_segment"])

        # 10分 = 6000行そろっている区間だけ採用
        # 10分ごとに分けて最後に余る区間は全て切り捨てとする
        if len(g) == int(SEGMENT_LEN):
            segments.append((int(sid), g))

    return segments

# ある10分のサンプルを10秒ごとの入力データへ分割する
def split_into_windows(seg, feature_cols):
    a = seg[feature_cols].to_numpy(dtype=np.float32)
    n = len(a) // WINDOW_LEN

    if n == 0:
        return np.empty((0, WINDOW_LEN, len(feature_cols)), np.float32)
    
    return a[:n*WINDOW_LEN].reshape(n, WINDOW_LEN, len(feature_cols))

# 指定した10分のサンプルのラベルを取得する
def label_from_segment(seg, col):
    vals = seg[col].dropna().astype(str).str.strip()
    vals = vals[vals != ""]

    if len(vals) == 0:
        return None
    
    return vals.iloc[0]

# 生データから活動量（activity_level）の値を取得する
def activity_from_csv(df):
    value = pd.to_numeric(df["activity_level"].iloc[0], errors="coerce")

    if pd.isna(value):
        return None
    
    return float(value)

# 生データからDistanceのカラムを探して番号順に取得する関数
def numeric_distance_columns(df):
    return sorted(
        [c for c in df.columns if re.fullmatch(r"Distance_\d+", str(c))],
        key = lambda c: int(c.split("_")[1])
    )

# Distance列から対応するラベル列のカラム名を取得する関数
def distance_label_col(distance_col):
    return f"{distance_col}_Label"
