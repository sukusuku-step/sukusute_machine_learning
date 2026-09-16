from __future__ import annotations
import torch
import torch.nn as nn

from models.cnn import CNNEncoder

# 各10秒の特徴ベクトルを10分束ねてLSTMにより10分での時間的な特徴を抽出する

# 相対距離データの学習用モデル
class DistanceModel(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, emb=96, hidden=96):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=2, batch_first=True,
                            dropout=0.2, bidirectional=True)
        self.attn = nn.Linear(hidden*2,1)
        self.head = nn.Linear(hidden*2,n_classes)
    def forward(self, x):
        # x [B,60,100,1]
        B,W,T,C=x.shape
        z=self.encoder(x.reshape(B*W,T,C)).reshape(B,W,-1)
        h,_=self.lstm(z)
        a=torch.softmax(self.attn(h).squeeze(-1),dim=1)
        p=(h*a.unsqueeze(-1)).sum(1)
        return self.head(p)