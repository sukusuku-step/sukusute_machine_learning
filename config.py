# ハイパーパラメータの設定

SAMPLE_RATE = 10 # 1秒間に何回データを取得するか（計測は0.1秒刻み）
BLOCK_SECONDS = 600 # 1つの学習データを何秒間分にするか（10分ごとのラベリング）
WINDOW_SECONDS = 10 # CNNに入力する時間幅

WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS # 10分間に何行のデータがあるか
NUM_WINDOWS = BLOCK_SECONDS // WINDOW_SECONDS # 10分間を何個の10秒区間に分割するか

BATCH_SIZE = 8 # 1回の学習で使うデータ数
EPOCHS = 30 # エポック数
LEARNING_RATE = 0.001 # 学習率

CNN_CHANNELS = 128 # CNNが抽出する特徴量の次元
LSTM_HIDDEN = 128 # LSTMの中間層の次元
LSTM_LAYERS = 2 # LSTMのレイヤー数

# 測定した実データCSVファイルへのパス
CSV_PATH = "sensor/collected_data/data.csv"

# 学習済みモデルを格納するフォルダへのパス
OUTPUT_DIR = "models/saved"
MODEL_PATH = "models/saved/model.pth"
SCALER_PATH = "models/saved/scaler.pkl"
METADATA_PATH = "models/saved/metadata.pkl"

# 過去データを取得するためのAPIのURL
API_URL = "http://localhost:8000/api/children"