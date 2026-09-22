# PyTorchで機械学習を回す

## 環境構築

### 1. 仮想環境の作成（初回のみ）

```bash
python3 -m venv .venv
```

### 2. 仮想環境の有効化

```bash
source .venv/bin/activate
```

### 3. 必要なライブラリのインストール

```bash
pip install -r requirements.txt // 結構時間がかかる
```

---

## 学習

ラベリング済み計測データの*.csvファイルを `collected_data` ディレクトリにあるだけ用意しておく  
`collected_data` ディレクトリ直下に置かれた*.csvファイルを全て学習する  

計測データのCSVはM5で作成する以下の形式で必ず固定  
[Timestamp,Steps,Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz,Start,Distance_1,Distance_2,...] + ラベリング

### 実行方法

```bash
 // 学習済みモデル及びその設定ファイルを保存するためのフォルダを予め作成
mkdir models/saved/behavior
mkdir models/saved/distance

// 学習を実行（学習済みモデル及びその設定ファイルができる）
python -m training.train_behavior_activity
python -m training.train_distance
```

### 学習アルゴリズム

以下の2種類の汎用モデルを学習する（ここでは個人チューニングは行わない）

- 歩数・加速度データの数値から `Pedo_Label`、`Acce_Label`、`activity_level` を推定する分類モデル
- 複数の `Distance_N` の数値データから `Distance_N_Label` を推定する分類モデル（相手デバイスは区別しない）

センサデータは0.1秒周期で記録されるため、ラベルのある10分間データは6000サンプル（6000行）となる  
ラベルと一対一対応するこの10分間の学習サンプルを1つの単位として全体の学習を行う

### 歩数・加速度・活動量ラベルの予測モデル

CSVから以下の時系列センサデータを使用する

* Steps
* Ax, Ay, Az
* Gx, Gy, Gz
* Mx, My, Mz

さらに、以下の特徴量を追加で生成する

* 歩数変化量 `Step_Diff`
* 加速度の大きさ `Acc_Mag`
* ジャイロの大きさ `Gyro_Mag`
* 地磁気の大きさ `Mag_Mag`
* 加速度変化量 `Acc_Diff`
* 加速度変化率 `Jerk`

特徴量は学習データから求めた平均値・標準偏差を用いて標準化する

10分間のデータを10秒×60区間に分割し、各10秒区間をCNNに入力して10秒単位での特徴ベクトルを生成する

歩数・加速度の分類モデルを作成する学習アルゴリズムのフローは以下の通り

```text
10分データ
    ↓
10秒 × 60区間へ分割
    ↓
CNNへ入力
    ↓
10秒ごとの特徴 × 60を取得
    ↓
BiLSTMへ入力
    ↓
Attention機構
    ↓
その10分のラベルが正解データ
    ├─→ Pedo_Label
    └─→ Acce_Label
```

CNNは各10秒間のセンサデータの特徴を抽出し、BiLSTMは60個の特徴から10分間の時間変化を学習させる  
その後、Attention機構を用いて10分間のうち分類に重要な区間を重み付けする

`activity_level` はCSV全体に対する1～5の値の予測であるため、各10分区間から得られた特徴をさらにBiLSTMへ入力する

活動量の値の予測モデルを作成する学習アルゴリズムのフローは以下の通り

```text
10分特徴①
10分特徴②
10分特徴③
    ↓
Session BiLSTMへさらに入力
    ↓
activity_level（1～の5の値）が正解データ
```

### 相対距離ラベルの予測モデル

CSV内には `Distance_2`、`Distance_4`、`Distance_12` のように複数相手の相対距離カラムが存在する場合がある

これらに関して、この学習では相手デバイスを区別せず、各 `Distance_N` を独立した学習サンプルとして扱う

```text
Distance_2  + Distance_2_Label
Distance_4  + Distance_4_Label
Distance_12 + Distance_12_Label
        ↓
全て共通の相対距離分類モデルへ入力（それぞれを区別せずに学習する）
```

Distanceには計測タイムアウトなどによるNaNが含まれる
NaNそのものはニューラルネットワークへ入力せず、以下の2チャネルへ変換する

```text
ch0 : 標準化したDistance
ch1 : IsNaN
```

`IsNaN` は以下の値を取る。

```text
正常な距離値 : 0
NaN          : 1
```

Distanceデータの標準化ではNaNを平均値・標準偏差の計算から除外する
NaNの位置のDistanceは標準化後に0とし、`IsNaN=1` によってNaNであることをモデルへ伝える

相対距離の分類モデルを作成する学習アルゴリズムのフローは以下の通り

```text
Distance + IsNaN（2チャネル）
      ↓
10秒 × 60区間へ分割
      ↓
CNNへ入力
      ↓
10秒ごとの特徴 × 60を取得
      ↓
BiLSTMへ入力
      ↓
Attention機構
      ↓
Distance_N_Labelが正解データ
```

---

## 推論

### 実行方法

```bash
python -m inference.predict
```

### 内容

* 学習済みモデルを読み込み
* サンプルデータを1つ乱数値で生成して入力
* 学習済みモデルで健康状態を推論
* 推論結果を表示

---

## モデル評価

### 実行方法

```bash
python -m inference.evaluate
```

### 内容

* 学習済みモデルを読み込み
* ランダムなダミーデータを100件生成
* 100件すべてについて推論を実行
* 正解ラベルと推論結果を比較
* 最終的なAccuracy（正解率）を表示

---
