from __future__ import annotations
import torch
import torch.nn as nn

from models.cnn import CNNEncoder

from config import CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS

# 各Distance_Nのデータは、相手デバイスごとで区別はせず全て単純な数値とラベルの学習データと見る
# 得られた各Distance_Nの10秒の特徴ベクトルを、10分束ねてBiLSTM ⇒ Attentionでまとめて学習する

# 相対距離データの学習用モデル（Distance + IsNaN の2チャネル）
class DistanceModel(nn.Module):
    def __init__(self, in_channels=2, n_classes=3):
        super().__init__()

        self.in_channels = in_channels

        # CNNから10秒単位の特徴ベクトルを受け取る
        self.encoder = CNNEncoder(in_channels, CNN_CHANNELS)

        # LSTMにより10分間の時間的な特徴を学習する
        self.lstm = nn.LSTM(
            CNN_CHANNELS, 
            LSTM_HIDDEN, 
            num_layers=LSTM_LAYERS, 
            batch_first=True,
            dropout=0.2 if LSTM_LAYERS > 1 else 0.0, 
            bidirectional=True
        )
        self.attn = nn.Linear(LSTM_HIDDEN*2, 1)

        # 10分間の特徴から相対距離のラベルを予測
        self.head = nn.Linear(LSTM_HIDDEN*2, n_classes)

    # CSV全体を処理する関数
    def forward(self, x):
        # x [B,60,100,2]、ch0 = Distance、ch1 = IsNaN
        B, W, T, C = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, "
                f"got {C}"
            )

        # 各10秒を独立してCNNへ渡す
        z = self.encoder(x.reshape(B*W, T, C))

        # [B,60,CNN_CHANNELS]
        z = z.reshape(B, W, -1)

        # 60個の10秒の特徴ベクトルをLSTMへ入力
        h, _ = self.lstm(z)

        # Attention
        a = torch.softmax(self.attn(h).squeeze(-1), dim=1)

        # 10分間を1つの特徴ベクトルにする
        pooled = (h * a.unsqueeze(-1)).sum(dim=1)

        # 最終的な出力結果（相対距離のラベル分類）
        return self.head(pooled)