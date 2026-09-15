import torch.nn as nn

# CNNの構築
class SensorCNN(nn.Module):
    def __init__(self, input_features, output_features=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(
                input_features,
                64,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(
                64,
                output_features,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(output_features),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)