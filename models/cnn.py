import torch.nn as nn

from config import CNN_CHANNELS

# 10秒ごとに分割したデータからCNNにより特徴ベクトルを作る
class CNNEncoder(nn.Module):
    def __init__(self, in_channels, emb=CNN_CHANNELS):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 5, padding=2),
            nn.BatchNorm1d(96), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(96, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128, emb), nn.LayerNorm(emb), nn.GELU())

    def forward(self, x):
        return self.proj(self.net(x.transpose(1,2)))