import requests
import pandas as pd

import config

# APIを叩いてDBから過去データをJSONで取得する関数
def load_training_data():
    # サーバから過去データを受け取る
    response = requests.get(config.API_URL)

    # ステータスコードが200以外なら例外を送出
    response.raise_for_status()

    return response.json()

# CSVファイルからデータを読み込む関数
def load_csv(path):
    df = pd.read_csv(path)

    x = df[["x1", "x2"]].values
    y = df["label"].values

    return x, y