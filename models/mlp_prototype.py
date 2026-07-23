import torch.nn as nn

import config

# 単純な全結合ネットワーク（MLP）の学習モデル
class SukusuteNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 入力層⇒中間層1⇒中間層2⇒出力層（3クラス）でMLPを構築する
        self.net = nn.Sequential(
            nn.Linear(config.INPUT_SIZE, config.HIDDEN_SIZE1),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_SIZE1, config.HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_SIZE2, config.OUTPUT_SIZE)
        )

    def forward(self,x):
        return self.net(x)