import pandas as pd
from config import BLOCK_SECONDS, SAMPLE_RATE

# 読み込むCSVに必要なカラム
REQUIRED_COLUMNS = [
    "Timestamp",
    "Steps",
    "Ax", "Ay", "Az",
    "Gx", "Gy", "Gz",
    "Mx", "My", "Mz"
]

# CSVファイルを読み込むための関数
def load_csv(path):
    df = pd.read_csv(path)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"必要なカラムがありません: {col}")

    df["Timestamp"] = pd.to_numeric(
        df["Timestamp"],
        errors="coerce"
    )

    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df

# 読み込んだCSVデータをラベル付けの区間で分割して学習サンプルにする
def split_into_blocks(df):
    samples_per_block = BLOCK_SECONDS * SAMPLE_RATE

    blocks = []

    for start in range(0, len(df), samples_per_block):
        block = df.iloc[start:start + samples_per_block].copy()

        if len(block) != samples_per_block:
            continue

        blocks.append(block)

    return blocks

# 分割した区間のラベル（正解データ）を取得する関数
def get_label(block, column):
    if column not in block.columns:
        return None

    values = block[column].dropna()

    if len(values) == 0:
        return None

    return values.astype(str).mode().iloc[0]