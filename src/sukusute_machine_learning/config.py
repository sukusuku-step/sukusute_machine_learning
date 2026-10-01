# ハイパーパラメータの設定

SAMPLE_HZ = 10 # 計測では1秒間に何回データを計測するか（計測は0.1秒刻み）
LABEL_SEC = 600 # 1つの学習サンプルを何分単位にするか（600秒 = 10分ごとのラベリング）
SHORT_WINDOW_SEC = 10 # CNNに入力する際のデータの時間幅の単位（10秒ごとに分割して入力する）

N_WINDOWS = LABEL_SEC // SHORT_WINDOW_SEC # 10分の学習サンプルはCNN入力時に何個の10秒区間に分割されるか
WINDOW_LEN = SAMPLE_HZ * SHORT_WINDOW_SEC # 1つの10秒区間には何行のデータがあるか
SEGMENT_LEN = WINDOW_LEN * N_WINDOWS # 10分間のサンプルには合計で何行のデータがあるか

# 相対距離データを除くCSVのカラム
BASE_SENSOR_COLS = ["Steps","Ax","Ay","Az","Gx","Gy","Gz","Mx","My","Mz"]
IMU_COLS = ["Ax","Ay","Az","Gx","Gy","Gz","Mx","My","Mz"]

# 相対距離データのラベルと対応する重み付けの値
LABEL_WEIGHTS_DEFAULT={
    "測定値なし/タイムアウト": 0.0,
    "一人": 0.0,
    "接近（同じ部屋）": 0.5,
    "接近（同じ行動）": 1.0
}

BATCH_SIZE = 8 # 1回の学習で使うデータ数
EPOCHS = 30 # 学習のエポック数
LEARNING_RATE = 0.001 # 学習率

SEED = 42 # 学習の際に順番をランダム化するための乱数のシード値

CNN_CHANNELS = 128 # CNNが抽出する特徴量の次元
LSTM_HIDDEN = 128 # LSTMの中間層の次元
LSTM_LAYERS = 2 # LSTMのレイヤー数

MIN_HISTORY_HOURS = 0.08 # ベースラインを作るために必要な最低履歴時間（24時間）
DEFAULT_HISTORY_DAYS = 14 # 履歴データを使う場合の参照する過去の期間（14日間）

# 学習用のラベリング済み実測データCSVファイルが格納されているフォルダのパス
CSV_PATH = "collected_data"

# 作成した学習済みモデル及びその設定ファイルを格納するフォルダのパス
BEHAVIOR_MODEL_PATH = "models/saved/behavior"
DISTANCE_MODEL_PATH = "models/saved/distance"

# 学習モデル評価の際のテスト用計測データのCSVファイルが格納されているフォルダのパス
EVALUATE_CSV_PATH = "collected_data/evaluate"