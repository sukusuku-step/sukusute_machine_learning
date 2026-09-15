import torch.nn as nn

# LSTMの構築
class SensorLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

    def forward(self, x):
        x, _ = self.lstm(x)

        return x[:, -1, :]