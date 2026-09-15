import torch.nn as nn

# 分類モデルの作成
class MultiTaskClassifier(nn.Module):
    def __init__(self, hidden_size, pedo_classes, acce_classes, activity_classes):
        super().__init__()

        self.pedo = nn.Linear(
            hidden_size,
            pedo_classes
        )

        self.acce = nn.Linear(
            hidden_size,
            acce_classes
        )

        self.activity = nn.Linear(
            hidden_size,
            activity_classes
        )

    def forward(self, x):
        return (self.pedo(x), self.acce(x), self.activity(x))