from __future__ import annotations
import torch
import torch.nn as nn

from models.cnn import CNNEncoder

from config import CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS

# 各10秒の特徴ベクトルを10分束ねてLSTMにより10分での時間的な特徴を抽出する
# 60個の10秒CNN embedding -> BiLSTM -> attention
# activity_levelはCSV全体のラベルなので、session_headで累積10分系列を学習する

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
            dropout=0.2, 
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

        # セッション全体からactivity_lebelの値を予測する
        self.activity_head = nn.Linear(LSTM_HIDDEN*2, 5)

    # 1つの10分区間を処理する関数
    def encode_10min(self, x10):
        # x10 [B,60,100,C]
        B,W,T,C = x10.shape

        z = self.encoder(x10.reshape(B*W,T,C)).reshape(B,W,-1)
        h,_ = self.temporal(z)
        a = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        pooled = (h*a.unsqueeze(-1)).sum(dim=1)

        return pooled, self.pedo_head(pooled), self.acce_head(pooled)

    # CSV全体を処理する関数
    def forward(self, x10, session_mask=None):
        # x10 [B,S,60,100,C]
        B,S,W,T,C = x10.shape

        pooled = []
        pedo=[]
        acce=[]

        for s in range(S):
            p,po,ac = self.encode_10min(x10[:,s])
            pooled.append(p); pedo.append(po); acce.append(ac)

        seq = torch.stack(pooled,1)
        sh,_ = self.session_lstm(seq)

        if session_mask is None:
            last = sh[:,-1]
        else:
            lengths = session_mask.sum(1).long().clamp_min(1)
            last = sh[torch.arange(B, device=sh.device), lengths-1]

        activity = self.activity_head(last).squeeze(-1)

        # 最終的な出力は歩数・加速度のラベルと活動量の値の予測
        return torch.stack(pedo,1), torch.stack(acce,1), activity