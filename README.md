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

CSVから以下の時系列センサデータ及びラベルを抜き出して学習に使用する

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

CSVから`Distance_N`の時系列センサデータ及びラベルを抜き出して学習に使用する

CSV内には `Distance_2`、`Distance_4`、`Distance_12` のように複数相手の相対距離カラムが存在する場合がある

これらに関して、この学習では相手デバイスを区別せず各 `Distance_N` を独立した学習サンプルとして扱う

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
NaN         : 1
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

## 推論用の関数

学習済みモデルを用いた推論には以下の関数を用意している

| 関数 | 推論内容 | 入力形式 |
|---|---|---|
| `behavior_infer()` | 歩行・姿勢ラベル | 10分間の `Steps, Ax～Mz` |
| `activity_infer()` | 活動量の値（1～5） | 10分以上の `Steps, Ax～Mz` |
| `distance_infer()` | 相対距離ラベル | 10分間の1つの `Distance_N` |

`behavior_infer()` と `distance_infer()` は0.1秒周期で取得した10分間、すなわち6000サンプルを入力する

### サーバ側での入力データ作成

DBには以下のようなカラムで各児童ごとで計測データが保存されることを想定（dateはTimestampとStartから作成）  
（distance_1などはID=1の相手デバイスとの相対距離ということを意味する）

```text
child_id | date | steps | ax | ay | az | gx | gy | gz | mx | my | mz | distance_1 | distance_2 | ...
````

モデルへの入力には `child_id` と `date` は使用しない

ただし、DBからデータを取得するときは時系列順を保証するため、必ず `date` の昇順でデータを取得することが必要

### 1. DBから10分間のデータを取得する

計測データは0.1秒周期で保存されるため、10分間では6000サンプルとなる

歩行・姿勢および相対距離を推論する場合、DBから推論対象となる10分間のデータを `Date` の昇順で6000行取得する  
（活動量の推論を行う場合は任意のDBから推論対象となる任意の行数のデータを同様に取得する）

SQLAlchemyを用いる場合、サーバ側で例えば以下の様にDBからデータを取り出せばよい

```python
from datetime import timedelta
from sqlalchemy import select

end_time = start_time + timedelta(minutes=10)

stmt = (
 select(Measurement)
 .where(Measurement.Date >= start_time, Measurement.Date < end_time)
 .order_by(Measurement.Date.asc())
)

rows = session.execute(stmt).scalars().all()

if len(rows) != 6000:
 raise ValueError("10分間の計測データは6000行必要です。")
```

取得した上記の `rows` を `pandas.DataFrame` に変換した後、そこから必要なカラムのみNumPy配列として取り出す  
変換したNumPy配列（行列の形式）が各推論関数が受け付ける入力形式となっている

以下が `pandas.DataFrame` に変換したものの例である（表形式となる）

```text
steps | ax | ay | az | ... | mz | distance_2 | distance_4 | ...
---------------------------------------------------------------
100   | ...                  | ...| 1.25       | NaN
100   | ...                  | ...| 1.24       | NaN
101   | ...                  | ...| 1.22       | 2.31
...
```

なお、相対距離データに関しては空欄（NaN）を許容している

`pandas.DataFrame` の形式のデータは必要なカラムを選択して `to_numpy()` によりNumPy配列にできる

- `behavior_infer()`：`steps, ax～mz`
- `distance_infer()`：対象となる1つの `distance_N`

### 2. 歩行・姿勢モデルへの入力

以下の10個のカラムだけを取り出してNumpy配列にする

```python
BEHAVIOR_COLUMNS = [
    "Steps",
    "Ax", "Ay", "Az",
    "Gx", "Gy", "Gz",
    "Mx", "My", "Mz",
]

behavior_data = df[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)
```

10分間であれば、

```python
behavior_data.shape
# (6000, 10) が出力
```

となる

この配列をそのまま、

```python
result = behavior_infer(behavior_data)
```

として推論関数に渡すことで入力とできる

`Timestamp` は推論関数内で0.1秒周期を前提に生成するため、サーバ側で用意する必要はない

### 3. 相対距離モデルへの入力

相対距離は `Distance_N` ごとに個別に推論する

例えば `Distance_1` の場合、`Distance_1` カラムを取り出してNumpy配列にする

```python
distance_data = df["Distance_1"].to_numpy(dtype=np.float32)
```

入力shapeは、

```python
distance_data.shape
# (6000,)が出力
```

となる

この配列をそのまま、

```python
result = distance_infer(distance_data)
```

として推論関数に渡すことで入力とできる

`Distance_N` に含まれるNaNは、そのまま残して入力する
サーバ側で0や平均値に置換する必要はなく、NaN処理は `distance_infer()` 内部で行う

複数の相手を推論する場合は、

```python
for column in ["Distance_1", "Distance_2", "Distance_3",]:
    distance_data = df[column].to_numpy(dtype=np.float32)
    result = distance_infer(distance_data)
```

のように各 `Distance_N` を個別に推論関数へ入力するようにする

### 4. 活動量モデルへの入力

`activity_infer()` の入力列は `behavior_infer()` と同じ形式となる

```python
activity_data = df[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)
result = activity_infer(activity_data)
```

ただし、活動量は10分単位ではなく対象期間全体（任意の長さ）から推論する

例えば1時間分の活動量を求める場合は、

```text
1時間 × 3600秒 × 10 Hz = 36000行
```

の分を取得して入力すればよい

---

## サーバ側の処理フロー

```text
DB
 ↓
Date昇順で対象期間を取得
 ↓
pandas.DataFrame
 │
 ├─ Steps + Ax～Mz
 │      ├─ 10分間 → behavior_infer()
 │      └─ 対象期間全体 → activity_infer()
 │
 └─ Distance_N
        └─ 各列10分間 → distance_infer()
```

サーバ側では、DBから取得したレコードを直接モデルへ渡すのではなくそれをDataFrameへ変換し、必要なカラムだけを選択してそれをさらにNumPy配列へ変換してから各推論関数へ渡せばよい

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
