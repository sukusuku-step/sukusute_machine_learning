# ハイパーパラメータの設定

LEARNING_RATE = 0.001 # 学習率
EPOCHS = 100 # エポック数
BATCH_SIZE = 32 # 1回の学習で使うデータ数

INPUT_SIZE = 9 # 入力特徴量の数
HIDDEN_SIZE1 = 32 # 中間層のニューロン数
HIDDEN_SIZE2 = 16 # 中間層のニューロン数
OUTPUT_SIZE = 3 # 出力数の数（クラス分類の数）

# 学習用のダミーデータ（乱数）の標本サイズ
DUMMY_DATA_SIZE = 1000

# 測定した実データCSVファイルへのパス
CSV_PATH = "sensor/collected_data/data.csv"

# 学習済みモデル（.pthファイル）を格納するフォルダへのパス
MODEL_SAVE_PATH = "models/saved/mlp_prototype.pth"