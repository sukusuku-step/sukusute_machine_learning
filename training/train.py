import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import *
from preprocessing import load_csv, split_into_blocks, get_label
from feature_extraction import extract_features
from dataset import SensorDataset
from models.cnn import SensorCNN
from models.lstm import SensorLSTM
from models.classifier import MultiTaskClassifier

# 実際に学習済みモデルを作るプログラム
class CNNLSTM(nn.Module):
    def __init__(self, input_features, pedo_classes, acce_classes, activity_classes):
        super().__init__()

        self.cnn = SensorCNN(input_features, CNN_CHANNELS)
        self.lstm = SensorLSTM(CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS)
        self.classifier = MultiTaskClassifier(LSTM_HIDDEN, pedo_classes, acce_classes, activity_classes)

    def forward(self, x):
        batch, windows, samples, features = x.shape

        x = x.reshape(batch * windows, samples, features)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.reshape(batch, windows, CNN_CHANNELS)
        x = self.lstm(x)

        return self.classifier(x)

def main():
    if len(sys.argv) < 2:
        print("python train.py data.csv")
        return

    csv_path = sys.argv[1]

    # CSV読み込み
    print("CSV読み込み...")
    df = load_csv(csv_path)

    # 特徴量抽出
    print("特徴量抽出...")
    features, distance_cols = extract_features(df)

    # 10分の学習サンプル作成
    print("10分の学習サンプル作成...")
    raw_blocks = split_into_blocks(df)

    blocks = []

    for block_df in raw_blocks:
        index = block_df.index

        block_features = features.loc[index].values

        pedo = get_label(block_df, "Pedo_Label")
        acce = get_label(block_df, "Acce_Label")
        activity = get_label(block_df, "activity_level")

        if (pedo is None or acce is None or activity is None):
            continue

        blocks.append({
            "features": block_features,
            "pedo": pedo,
            "acce": acce,
            "activity": activity
        })

    if len(blocks) < 2:
        raise ValueError(
            "学習に必要な学習サンプルが不足しています"
        )

    print(f"10分の学習サンプルの数: {len(blocks)}")

    # ラベル変換
    pedo_encoder = LabelEncoder()
    acce_encoder = LabelEncoder()
    activity_encoder = LabelEncoder()

    pedo_encoder.fit([b["pedo"] for b in blocks])
    acce_encoder.fit([b["acce"] for b in blocks])
    activity_encoder.fit([b["activity"] for b in blocks])

    for b in blocks:
        b["pedo"] = pedo_encoder.transform([b["pedo"]])[0]
        b["acce"] = acce_encoder.transform([b["acce"]])[0]
        b["activity"] = activity_encoder.transform([b["activity"]])[0]

    # 8:2で分割
    split = int(len(blocks) * 0.8)

    train_blocks = blocks[:split]
    val_blocks = blocks[split:]

    # 標準化
    scaler = StandardScaler()

    train_data = np.concatenate(
        [
            b["features"]
            for b in train_blocks
        ],
        axis=0
    )

    scaler.fit(train_data)

    for b in blocks:
        b["features"] = scaler.transform(b["features"]).astype(np.float32)

    # データセット
    train_dataset = SensorDataset(train_blocks)
    val_dataset = SensorDataset(val_blocks)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # モデル
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = CNNLSTM(
        features.shape[1],
        len(pedo_encoder.classes_),
        len(acce_encoder.classes_),
        len(activity_encoder.classes_)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")

    # 学習
    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0

        for x, y_pedo, y_acce, y_activity in train_loader:
            x = x.to(device)
            y_pedo = y_pedo.to(device)
            y_acce = y_acce.to(device)
            y_activity = y_activity.to(device)

            optimizer.zero_grad()

            pred_pedo, pred_acce, pred_activity = model(x)

            loss = (
                criterion(pred_pedo, y_pedo) +
                criterion(pred_acce, y_acce) +
                criterion(pred_activity, y_activity)
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y_pedo, y_acce, y_activity in val_loader:
                x = x.to(device)
                y_pedo = y_pedo.to(device)
                y_acce = y_acce.to(device)
                y_activity = y_activity.to(device)

                pred_pedo, pred_acce, pred_activity = model(x)

                loss = (
                    criterion(pred_pedo, y_pedo) +
                    criterion(pred_acce, y_acce) +
                    criterion(pred_activity, y_activity)
                )

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(model.state_dict(), MODEL_PATH)

    # 保存情報
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "feature_columns": list(features.columns),
        "distance_columns": distance_cols,
        "pedo_classes": list(pedo_encoder.classes_),
        "acce_classes": list(acce_encoder.classes_),
        "activity_classes": list(activity_encoder.classes_)
    }

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("学習完了")
    print(f"モデル: {MODEL_PATH}")

if __name__ == "__main__":
    main()