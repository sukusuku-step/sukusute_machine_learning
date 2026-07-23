import pandas as pd

# CSVファイルからデータを読み込む関数
def load_csv(path):
    df = pd.read_csv(path)

    x = df[["x1", "x2"]].values
    y = df["label"].values

    return x, y