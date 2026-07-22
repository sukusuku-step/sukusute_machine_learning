import torch.nn as nn

# 単純な全結合ネットワーク（MLP）で学習モデルを構築
class SukusuteNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(9,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,3)
        )

    def forward(self,x):
        return self.net(x)