from __future__ import annotations
import torch
import torch.nn as nn

from models.cnn import CNNEncoder

from config import CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS

# 各10秒の特徴ベクトルを10分束ねてLSTMにより10分での時間的な特徴を抽出する

# 相対距離データの学習用モデル
class DistanceModel(nn.Module):
    def __init__(self, in_channels=1, n_classes=3):
        super().__init__()

        # CNNから10秒単位の特徴ベクトルを受け取る
        self.encoder = CNNEncoder(in_channels, CNN_CHANNELS)

        # LSTMにより10分間の時間的な特徴を学習する
        self.lstm = nn.LSTM(
            CNN_CHANNELS, 
            LSTM_HIDDEN, 
            num_layers=LSTM_LAYERS, 
            batch_first=True,
            dropout=0.2, 
            bidirectional=True
        )
        self.attn = nn.Linear(LSTM_HIDDEN*2,1)

        # 10分間の特徴から相対距離のラベルを予測
        self.head = nn.Linear(LSTM_HIDDEN*2,n_classes)

    # CSV全体を処理する関数
    def forward(self, x):
        # x [B,60,100,1]
        B,W,T,C=x.shape

        z=self.encoder(x.reshape(B*W,T,C)).reshape(B,W,-1)
        h,_=self.lstm(z)
        a=torch.softmax(self.attn(h).squeeze(-1),dim=1)
        p=(h*a.unsqueeze(-1)).sum(1)

        # 最終的な出力は相対距離のラベルの予測
        return self.head(p)