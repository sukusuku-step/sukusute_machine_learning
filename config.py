# ハイパーパラメータの設定

SAMPLE_HZ = 10 # 1秒間に何回データを取得するか（計測は0.1秒刻み）
LABEL_SEC = 600 # 1つの学習データを何秒間分にするか（10分ごとのラベリング）
SHORT_WINDOW_SEC = 10 # CNNに入力する時間幅

WINDOW_LEN = SAMPLE_HZ * SHORT_WINDOW_SEC # 10分間に何行のデータがあるか
N_WINDOWS = LABEL_SEC // SHORT_WINDOW_SEC # 10分間を何個の10秒区間に分割するか

# 相対距離データを除くCSVのカラム
BASE_SENSOR_COLS = ["Steps","Ax","Ay","Az","Gx","Gy","Gz","Mx","My","Mz"]
IMU_COLS = ["Ax","Ay","Az","Gx","Gy","Gz","Mx","My","Mz"]

# 相対距離データのラベルと対応する重み付けの値
LABEL_WEIGHTS_DEFAULT={
    "測定値なし/タイムアウト":0.0,
    "一人":0.0,
    "接近（同じ部屋）":0.5,
    "接近（同じ行動）":1.0,
}

BATCH_SIZE = 8 # 1回の学習で使うデータ数
EPOCHS = 30 # エポック数
LEARNING_RATE = 0.001 # 学習率

SEED = 42 # 学習の際に順番をランダム化するための乱数のシード値

CNN_CHANNELS = 128 # CNNが抽出する特徴量の次元
LSTM_HIDDEN = 128 # LSTMの中間層の次元
LSTM_LAYERS = 2 # LSTMのレイヤー数

MIN_HISTORY_HOURS=24 # ベースラインを作るために必要な最低履歴時間
DEFAULT_HISTORY_DAYS=14 # 履歴データを使う場合の参照する過去の期間

# 測定した実データCSVファイルへのパス
CSV_PATH = "sensor/collected_data/data.csv"

# 学習済みモデルを格納するフォルダへのパス
BEHAVIOR_MODEL_PATH = "models/saved/behavior"
DISTANCE_MODEL_PATH = "models/saved/distance"

# 過去データを取得するためのAPIのURL
API_URL = "http://localhost:8000/api/children"