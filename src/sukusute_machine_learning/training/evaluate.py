from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from utils.collector import *
from utils.add_features import *
from utils.deal_csv import *
from models import BehaviorActivityModel, DistanceModel

from config import (
    N_WINDOWS,
    WINDOW_LEN,
    SEGMENT_LEN,
    EVALUATE_CSV_PATH,
    BEHAVIOR_MODEL_PATH,
    DISTANCE_MODEL_PATH
)

# 構築した2種類の学習済みモデルの評価用プログラム
# 同じ形式のCSVファイルを読み取り、ラベルの推論と（ラベリングされていれば）正解率を計算する

# cudaが使えればcudaを使い、使えなければcpuを使う設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_csv_paths():
    # EVALUATE_CSV_PATHの中の全CSVを取得する
    paths = (
            # CSV_PATHで指定されたディレクトリ内にある.csvファイルを全て取得しリストに入れる
            glob.glob(os.path.join(EVALUATE_CSV_PATH, "*.csv"))
            if os.path.isdir(EVALUATE_CSV_PATH)
            else [EVALUATE_CSV_PATH]
        )

    if not paths:
        raise ValueError(
            f"評価用CSVが見つかりません: "
            f"{EVALUATE_CSV_PATH}"
        )

    return paths

# ============================================================
# Behavior / Activityモデル
# ============================================================

def load_behavior_model():
    model_dir = Path(BEHAVIOR_MODEL_PATH)

    config_path = model_dir / "config.json"
    scaler_path = model_dir / "scaler.joblib"
    model_path = model_dir / "behavior_activity.pt"

    config = load_json(config_path)
    scaler = joblib.load(scaler_path)

    feature_cols = config["feature_cols"]
    pedo_classes = config["pedo_classes"]
    acce_classes = config["acce_classes"]

    model = BehaviorActivityModel(
        in_channels=len(feature_cols),
        n_pedo=len(pedo_classes),
        n_acce=len(acce_classes)
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return (
        model,
        scaler,
        config
    )

def evaluate_behavior_csv(path, model, scaler, config, stats):
    print()
    print("=" * 70)
    print(f"Behavior / Activity: {path}")
    print("=" * 70)

    df = read_csv(path)

    # 学習時と同じ特徴量を生成
    df, _ = add_engineered_features(df)

    feature_cols = config["feature_cols"]
    pedo_classes = config["pedo_classes"]
    acce_classes = config["acce_classes"]

    # 学習時に使用した特徴量が存在することを確認
    missing = [
        c for c in feature_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}: 必要な特徴量がありません: "
            f"{missing}"
        )

    blocks = []
    segment_ids = []
    true_pedo = []
    true_acce = []

    # 10分単位に分割
    for sid, seg in make_10min_segments(df):
        w = split_into_windows(seg, feature_cols)

        if len(w) != N_WINDOWS:
            continue

        blocks.append(w)
        segment_ids.append(sid)

        # ラベル付きCSVなら正解ラベルも取得
        # ラベルなしならNone

        if "Pedo_Label" in seg.columns:
            p = label_from_segment(seg, "Pedo_Label")
        else:
            p = None

        if "Acce_Label" in seg.columns:
            a = label_from_segment(seg, "Acce_Label")
        else:
            a = None

        true_pedo.append(p)
        true_acce.append(a)

    if not blocks:
        print(
            "完全な10分区間が存在しないため、"
            "Behaviorモデルの評価をスキップします。"
        )
        return

    # [S, 60, 100, C]
    x = np.stack(blocks).astype(np.float32)

    # 学習時と同じScaler（標準化基準）を使用
    x = apply_scaler(x, scaler)

    # [1, S, 60, 100, C]
    xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pedo_logits, acce_logits, activity_logits = model(xt)
        pedo_prob = torch.softmax(pedo_logits, dim=-1)
        acce_prob = torch.softmax(acce_logits, dim=-1)
        activity_prob = torch.softmax(activity_logits, dim=-1)

    pedo_pred = (pedo_prob.argmax(dim=-1).cpu().numpy()[0])
    acce_pred = (acce_prob.argmax(dim=-1).cpu().numpy()[0])
    pedo_conf = (pedo_prob.max(dim=-1).values.cpu().numpy()[0])
    acce_conf = (acce_prob.max(dim=-1).values.cpu().numpy()[0])

    # 10分単位のPedo / Acce結果

    for i, sid in enumerate(segment_ids):
        p_pred = pedo_classes[int(pedo_pred[i])]
        a_pred = acce_classes[int(acce_pred[i])]

        start_min = sid * 10
        end_min = start_min + 10

        print()
        print(
            f"[{start_min:4d} - "
            f"{end_min:4d} min]"
        )

        print(
            f"  Pedo : {p_pred} "
            f"(confidence={pedo_conf[i]:.3f})"
        )

        print(
            f"  Acce : {a_pred} "
            f"(confidence={acce_conf[i]:.3f})"
        )

        # Pedo_Labelが存在する場合のみ評価
        if true_pedo[i] is not None:
            correct = (p_pred == true_pedo[i])

            print(
                f"         true={true_pedo[i]} "
                f"{'OK' if correct else 'NG'}"
            )

            stats["pedo_total"] += 1

            if correct:
                stats["pedo_correct"] += 1

        # Acce_Labelが存在する場合のみ評価
        if true_acce[i] is not None:
            correct = (a_pred == true_acce[i])

            print(
                f"         true={true_acce[i]} "
                f"{'OK' if correct else 'NG'}"
            )

            stats["acce_total"] += 1

            if correct:
                stats["acce_correct"] += 1

    # activity_level

    activity_idx = int(activity_prob.argmax(dim=-1).item())

    activity_classes = config.get(
        "activity_classes",
        [1, 2, 3, 4, 5]
    )

    activity_pred = activity_classes[activity_idx]
    activity_conf = float(activity_prob.max().item())

    print()
    print(
        f"  activity_level : "
        f"{activity_pred} "
        f"(confidence={activity_conf:.3f})"
    )

    # activity_level列が存在する場合だけ正解判定
    if "activity_level" in df.columns:
        true_activity = activity_from_csv(df)

        if true_activity is not None:
            true_activity = int(round(true_activity))

            correct = (activity_pred == true_activity)

            print(
                f"                   "
                f"true={true_activity} "
                f"{'OK' if correct else 'NG'}"
            )

            stats["activity_total"] += 1

            if correct:
                stats["activity_correct"] += 1

# ============================================================
# Distanceモデル
# ============================================================

def load_distance_model():
    model_dir = Path(DISTANCE_MODEL_PATH)
    config_path = (model_dir / "config.json")
    scaler_path = (model_dir / "scaler.joblib")
    model_path = (model_dir / "distance.pt")
    config = load_json(config_path)
    scaler = joblib.load(scaler_path)
    classes = config["classes"]

    model = DistanceModel(
        in_channels=2,
        n_classes=len(classes)
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return (model, scaler, config)

def evaluate_distance_csv(path, model, scaler, config, stats):
    print()
    print("=" * 70)
    print(f"Distance: {path}")
    print("=" * 70)

    df = read_csv(path)

    classes = config["classes"]

    distance_cols = numeric_distance_columns(df)

    if not distance_cols:
        print(
            "Distance_N列が存在しないため"
            "スキップします。"
        )
        return

    for dc in distance_cols:
        lc = distance_label_col(dc)

        d = pd.to_numeric(
            df[dc],
            errors="coerce"
        )

        # ラベル無しCSVにも対応するため、
        # Distance_N_Labelは存在するときだけ追加
        temp_data = {
            "Timestamp": df["Timestamp"],
            dc: d
        }

        if lc in df.columns:
            temp_data[lc] = df[lc]

        temp = pd.DataFrame(temp_data)

        for sid, seg in make_10min_segments(temp):
            if len(seg) != SEGMENT_LEN:
                continue

            distance = (seg[dc].to_numpy(dtype=np.float32))

            # [60,100]
            distance = distance.reshape(N_WINDOWS, WINDOW_LEN)

            # 学習時と同じScalerを使用
            #
            # [60,100]
            #   ↓
            # [60,100,2]
            #
            # ch0 = Distance_scaled
            # ch1 = IsNaN
            x = make_distance_features(distance, scaler)

            if not np.isfinite(x).all():
                raise RuntimeError(
                    f"{path} / {dc}: "
                    f"前処理後にNaNまたはinfが"
                    f"残っています。"
                )

            # [1,60,100,2]
            xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(xt)
                prob = torch.softmax(logits, dim=-1)

            pred_idx = int(
                prob.argmax(dim=-1).item())

            pred_label = classes[pred_idx]

            confidence = float(prob.max().item())

            start_min = sid * 10
            end_min = start_min + 10

            print()
            print(
                f"[{dc}] "
                f"{start_min:4d} - "
                f"{end_min:4d} min"
            )

            print(
                f"  prediction : "
                f"{pred_label} "
                f"(confidence={confidence:.3f})"
            )

            # ラベル付きの場合だけ正解と比較

            if lc in seg.columns:

                true_label = (label_from_segment(seg, lc))

                if true_label is not None:

                    correct = (pred_label == true_label)

                    print(
                        f"  true       : "
                        f"{true_label} "
                        f"{'OK' if correct else 'NG'}"
                    )

                    stats["distance_total"] += 1

                    if correct:
                        stats["distance_correct"] += 1

# ============================================================
# 評価結果の出力
# ============================================================

def print_accuracy(name, correct, total):
    if total == 0:
        print(
            f"{name:<20}: "
            f"ラベルなし"
        )
        return

    accuracy = correct / total

    print(
        f"{name:<20}: "
        f"{accuracy:.3f} "
        f"({correct}/{total})"
    )

def main():
    print(
        f"device: {DEVICE}"
    )

    paths = get_csv_paths()

    print(
        f"evaluation CSV files: "
        f"{len(paths)}"
    )

    # モデル読み込み
    behavior_model, behavior_scaler, behavior_config = (load_behavior_model())
    distance_model, distance_scaler, distance_config = (load_distance_model())

    stats = {
        "pedo_correct": 0,
        "pedo_total": 0,

        "acce_correct": 0,
        "acce_total": 0,

        "activity_correct": 0,
        "activity_total": 0,

        "distance_correct": 0,
        "distance_total": 0,
    }

    # フォルダ内の全CSVを評価
    for path in paths:

        evaluate_behavior_csv(
            path,
            behavior_model,
            behavior_scaler,
            behavior_config,
            stats
        )

        evaluate_distance_csv(
            path,
            distance_model,
            distance_scaler,
            distance_config,
            stats
        )

    # ラベル付きデータについて精度を計算して表示
    print()
    print("=" * 70)
    print("Evaluation summary")
    print("=" * 70)

    print_accuracy(
        "Pedo_Label",
        stats["pedo_correct"],
        stats["pedo_total"]
    )

    print_accuracy(
        "Acce_Label",
        stats["acce_correct"],
        stats["acce_total"]
    )

    print_accuracy(
        "activity_level",
        stats["activity_correct"],
        stats["activity_total"]
    )

    print_accuracy(
        "Distance_Label",
        stats["distance_correct"],
        stats["distance_total"]
    )

if __name__ == "__main__":
    main()