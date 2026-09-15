import torch
from torch.utils.data import Dataset

from config import WINDOW_SIZE, NUM_WINDOWS

# 特徴量からCNNに実際に渡せる形のデータセットを作る
class SensorDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        block = self.blocks[index]

        x = block["features"]

        x = x.reshape(NUM_WINDOWS, WINDOW_SIZE, -1)

        # このデータセットをCNNへ入力する
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(block["pedo"], dtype=torch.long),
            torch.tensor(block["acce"], dtype=torch.long),
            torch.tensor(block["activity"], dtype=torch.long)
        )