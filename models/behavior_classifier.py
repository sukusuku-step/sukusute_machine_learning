from __future__ import annotations
import torch
import torch.nn as nn

from models.cnn import CNNEncoder

# 各10秒の特徴ベクトルを10分束ねてLSTMにより10分での時間的な特徴を抽出する
# 60個の10秒CNN embedding -> BiLSTM -> attention
# activity_levelはCSV全体のラベルなので、session_headで累積10分系列を学習する

# 歩数・加速度データの学習用モデル
class BehaviorActivityModel(nn.Module):
    def __init__(self, in_channels, n_pedo, n_acce, emb=128, hidden=128):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, emb)
        self.temporal = nn.LSTM(emb, hidden, num_layers=2, batch_first=True,
                                dropout=0.2, bidirectional=True)
        self.attn = nn.Linear(hidden*2, 1)
        self.pedo_head = nn.Linear(hidden*2, n_pedo)
        self.acce_head = nn.Linear(hidden*2, n_acce)
        # セッション全体の10分ベクトル列 -> activity 1..5
        self.session_lstm = nn.LSTM(hidden*2, hidden, num_layers=1,
                                    batch_first=True, bidirectional=True)
        self.activity_head = nn.Linear(hidden*2, 1)

    def encode_10min(self, x10):
        # x10 [B,60,100,C]
        B,W,T,C = x10.shape
        z = self.encoder(x10.reshape(B*W,T,C)).reshape(B,W,-1)
        h,_ = self.temporal(z)
        a = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        pooled = (h*a.unsqueeze(-1)).sum(dim=1)
        return pooled, self.pedo_head(pooled), self.acce_head(pooled)

    def forward(self, x10, session_mask=None):
        # x10 [B,S,60,100,C]
        B,S,W,T,C = x10.shape
        pooled = []
        pedo=[]; acce=[]
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
        return torch.stack(pedo,1), torch.stack(acce,1), activity