import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# 動作確認用ダミーデータ生成スクリプト

MINUTES = 120 # 何分のダミーデータを作るか
OUTPUT_DIR = Path("./collected_data/dummy") # 保存先ディレクトリ
OUTPUT_FILENAME = "dummy_sensor_data1.csv" # 作成するファイル名

RANDOM_SEED = None # 乱数のシード

# ラベル定義
PEDO_LABELS = ["静止", "歩行（ゆっくり）", "歩行（通常速度）", "早歩き/走る", "足踏み状態"]
ACCE_LABELS = ["立ち状態", "走行状態", "座り状態", "前傾姿勢", "寝転がり状態"]

DISTANCE_LABEL_TIMEOUT = "測定値なし/タイムアウト"
DISTANCE_LABELS_VALID = ["一人", "接近（同じ行動）", "接近（同じ部屋）"]

# 歩数データラベルごとの「1秒あたりの歩数発生確率」
PEDO_STEP_RATE = {
    "静止": 0.0,
    "歩行（ゆっくり）": 1.0,
    "歩行（通常速度）": 1.8,
    "早歩き/走る": 2.6,
    "足踏み状態": 1.4
}

# 加速度データラベルごとの重力の基準（Ax, Ay, Az のおおよその向き）
ACCE_GRAVITY_BASE = {
    "立ち状態": (0.02, 0.05, 0.98),
    "走行状態": (0.05, 0.08, 0.95),
    "座り状態": (0.03, -0.10, 0.97),
    "前傾姿勢": (0.30, 0.10, 0.90),
    "寝転がり状態": (0.85, 0.05, 0.15)
}

DT = 0.1  # サンプリング周期（秒）
BLOCK_SECONDS = 10 * 60  # 10分（ラベル切り替え単位）
BLOCK_ROWS = int(BLOCK_SECONDS / DT)

# ランダムな個数・ランダムなデバイスIDのDistance_Nカラムを決定する
def choose_distance_ids(rng, min_n=1, max_n=4, id_pool_max=30):
    n = rng.randint(min_n, max_n)
    ids = rng.sample(range(1, id_pool_max + 1), n)

    return ids

# 1つのDistance_Nカラム用の (値, ラベル) 列を事前生成する
def make_distance_generator(rng, total_rows):
    values = [None] * total_rows
    labels = [None] * total_rows

    val = round(rng.uniform(0.5, 10.0), 2)
    label = rng.choice(DISTANCE_LABELS_VALID)
    label_hold = rng.randint(30, 150)  # ラベルを保持する行数

    i = 0
    while i < total_rows:
        # 一定確率でNaN(タイムアウト)区間を挿入
        if rng.random() < 0.03:
            gap_len = rng.randint(5, 80)
            for j in range(i, min(i + gap_len, total_rows)):
                values[j] = None
                labels[j] = DISTANCE_LABEL_TIMEOUT
            
            i += gap_len

            # タイムアウト明けは値・ラベルをリセット
            val = round(rng.uniform(0.5, 10.0), 2)
            label = rng.choice(DISTANCE_LABELS_VALID)
            label_hold = rng.randint(30, 150)

            continue

        # 通常の値生成（ランダムウォーク）
        val += rng.uniform(-0.3, 0.3)
        val = max(0.0, min(30.0, val))
        values[i] = round(val, 2)

        label_hold -= 1

        if label_hold <= 0:
            label = rng.choice(DISTANCE_LABELS_VALID)
            label_hold = rng.randint(30, 150)
        
        labels[i] = label
        i += 1

    return values, labels

def generate(total_rows, rng, start_dt):
    n_ax = ["Timestamp", "Steps", "Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz", "Start"]

    distance_ids = choose_distance_ids(rng)
    distance_cols = {}
    for did in distance_ids:
        distance_cols[did] = make_distance_generator(rng, total_rows)

    header = list(n_ax)
    for did in distance_ids:
        header.append(f"Distance_{did}")
        header.append(f"Distance_{did}_Label")
    header += ["Pedo_Label", "Acce_Label", "activity_level"]

    rows = []

    steps = 0
    mx, my, mz = 10.0 + rng.uniform(-2, 2), 20.0 + rng.uniform(-2, 2), 30.0 + rng.uniform(-2, 2)

    pedo_label = None
    acce_label = None
    block_idx = -1

    # ノイズ用の位相
    phase_a = rng.uniform(0, 2 * math.pi)
    phase_g = rng.uniform(0, 2 * math.pi)

    for i in range(total_rows):
        t = round(i * DT, 1)

        # 10分ブロックごとにラベルを再抽選
        cur_block = i // BLOCK_ROWS
        if cur_block != block_idx:
            block_idx = cur_block
            pedo_label = rng.choice(PEDO_LABELS)
            acce_label = rng.choice(ACCE_LABELS)

        # Steps（単調増加）
        step_rate = PEDO_STEP_RATE[pedo_label]
        if step_rate > 0 and rng.random() < step_rate * DT:
            steps += 1

        # 加速度
        base_ax, base_ay, base_az = ACCE_GRAVITY_BASE[acce_label]
        motion_amp = 0.02 if pedo_label == "静止" else min(0.15, 0.03 + step_rate * 0.04)
        ax = base_ax + motion_amp * math.sin(2 * math.pi * 1.8 * t + phase_a) + rng.uniform(-0.01, 0.01)
        ay = base_ay + motion_amp * math.cos(2 * math.pi * 1.8 * t + phase_a) + rng.uniform(-0.01, 0.01)
        az = base_az + motion_amp * 0.5 * math.sin(2 * math.pi * 3.6 * t + phase_a) + rng.uniform(-0.01, 0.01)

        # 角速度
        gyro_amp = 0.05 if pedo_label == "静止" else min(0.6, 0.1 + step_rate * 0.2)
        gx = gyro_amp * math.sin(2 * math.pi * 0.5 * t + phase_g) + rng.uniform(-0.005, 0.005)
        gy = gyro_amp * math.cos(2 * math.pi * 0.5 * t + phase_g) + rng.uniform(-0.005, 0.005)
        gz = gyro_amp * 0.7 * math.sin(2 * math.pi * 0.8 * t + phase_g) + rng.uniform(-0.005, 0.005)

        # 地磁気
        mx += rng.uniform(-0.01, 0.02)
        my += rng.uniform(-0.01, 0.02)
        mz += rng.uniform(-0.01, 0.02)

        row = {
            "Timestamp": f"{t:.1f}",
            "Steps": steps,
            "Ax": round(ax, 4),
            "Ay": round(ay, 4),
            "Az": round(az, 4),
            "Gx": round(gx, 4),
            "Gy": round(gy, 4),
            "Gz": round(gz, 4),
            "Mx": round(mx, 2),
            "My": round(my, 2),
            "Mz": round(mz, 2),
            "Start": start_dt.strftime("%Y/%m/%d %H:%M") if i == 0 else "",
        }

        for did in distance_ids:
            vals, labs = distance_cols[did]
            v = vals[i]
            row[f"Distance_{did}"] = "" if v is None else v
            row[f"Distance_{did}_Label"] = labs[i]

        row["Pedo_Label"] = pedo_label
        row["Acce_Label"] = acce_label
        # activity_level は元データと同様、記録開始行のみに1〜5のランダム値を記録
        row["activity_level"] = rng.randint(1, 5) if i == 0 else ""

        rows.append(row)

    return header, rows

def main():
    rng = random.Random(RANDOM_SEED)

    total_rows = int(round(MINUTES * 60 / DT))

    start_dt = (datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=int(MINUTES)))

    header, rows = generate(total_rows, rng, start_dt)

    # 保存先ディレクトリが存在しなければ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"生成完了: {output_path}")
    print(f"  データ長: {MINUTES}分")
    print(f"  行数: {total_rows}行")
    print(f"  カラム数: {len(header)}カラム")
    print(f"  ファイル名: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()