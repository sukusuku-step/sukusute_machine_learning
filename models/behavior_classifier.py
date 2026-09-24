from __future__ import annotations
import torch
import torch.nn as nn

from models.cnn import CNNEncoder

from config import CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS

# 各10秒の特徴ベクトルを10分束ねてLSTMにより10分での時間的な特徴を抽出する
# 60個の10秒CNN embedding ⇒ BiLSTM ⇒ attention
# activity_levelはCSV全体のラベルなので、session_featureで累積10分系列を学習する

# 歩数・加速度データの学習用モデル
class BehaviorActivityModel(nn.Module):
    def __init__(self, in_channels, n_pedo, n_acce):
        super().__init__()

        # CNNから10秒単位の特徴ベクトルを受け取る
        self.encoder = CNNEncoder(in_channels, CNN_CHANNELS)

        # LSTMにより10分間の時間的な特徴を学習する
        self.temporal = nn.LSTM(
            CNN_CHANNELS,
            LSTM_HIDDEN, 
            num_layers=LSTM_LAYERS, 
            batch_first=True,
            dropout=0.2 if LSTM_LAYERS > 1 else 0.0,
            bidirectional=True
        )
        self.attn = nn.Linear(LSTM_HIDDEN*2, 1)
        self.pedo_head = nn.Linear(LSTM_HIDDEN*2, n_pedo)
        self.acce_head = nn.Linear(LSTM_HIDDEN*2, n_acce)

        # 複数の10分区間をさらに時系列として処理する
        self.session_lstm = nn.LSTM(
            LSTM_HIDDEN*2, 
            LSTM_HIDDEN, 
            num_layers=1,
            batch_first=True, 
            bidirectional=True
        )

        # セッション全体（10分単位に分割して切り捨てた部分を除く）からactivity_levelの値を予測する
        self.activity_head = nn.Linear(LSTM_HIDDEN*2, 5)

    # 1つの10分区間を処理する関数
    def encode_10min(self, x10):
        # x10 [B,60,100,C]
        B,W,T,C = x10.shape

        z = self.encoder(x10.reshape(B*W,T,C)).reshape(B,W,-1)
        h, _ = self.temporal(z)
        a = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        pooled = (h*a.unsqueeze(-1)).sum(dim=1)

        return pooled, self.pedo_head(pooled), self.acce_head(pooled)

    # CSV全体を処理する関数
    def forward(self, x10):
        # x10 [B,S,60,100,C]
        _, S, _, _, _ = x10.shape

        pooled = []
        pedo = []
        acce = []

        for s in range(S):
            p, po, ac = self.encode_10min(x10[:,s])

            pooled.append(p)
            pedo.append(po)
            acce.append(ac)

        # [B, S, LSTM_HIDDEN*2]
        seq = torch.stack(pooled, dim=1)

        # 双方向LSTMにする
        _, (h_n, _) = self.session_lstm(seq)
        session_feature = torch.cat([h_n[-2], h_n[-1]], dim=1)

        # [B, 5]
        activity = self.activity_head(session_feature)

        # 最終的な出力結果（歩数・加速度のラベル分類と活動量の値の予測）
        return (
            torch.stack(pedo, dim=1),
            torch.stack(acce, dim=1),
            activity
        )