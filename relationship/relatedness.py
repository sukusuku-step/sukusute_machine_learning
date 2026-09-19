from __future__ import annotations

from config import LABEL_WEIGHTS_DEFAULT

# 入力: 過去の相対距離のラベル分類の推論結果の集計（任意の長さの配列）
# 出力: あるデバイスから見た相手デバイスとの関連度スコア（0から1のfloatの値）

# 過去の相対距離のラベル分類からその相手との関連度スコアを計算する関数
def calc_relatedness(labels, weight_map=None, decay=0.9):
    # ラベルを定義に従って数値に変換する
    weight_map = weight_map or LABEL_WEIGHTS_DEFAULT

    # 入力のラベル列が無い場合
    if not labels:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0

    # 後ろにある推論結果ほど有効なデータとして見る
    for i, label in enumerate(labels):
        if label not in weight_map:
            continue

        # 最新のデータほど大きな重みを与えるようにする
        age = len(labels) - 1 - i
        time_weight = decay ** age

        weighted_sum += weight_map[label] * time_weight
        weight_total += time_weight

    # 有効な推論結果がない場合
    if weight_total == 0:
        return 0.0

    # 出力する関連度スコアの計算（0～1の範囲に標準化する）
    score = weighted_sum / weight_total
    relatedness_score = float(max(0.0, min(1.0, score)))

    return relatedness_score