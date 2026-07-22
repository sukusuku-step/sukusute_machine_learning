import torch
import torch.nn as nn

from dataset.dataset import SukusuteDataset
from features.dummy_features import create_dummy_data
from models.mlp_prototype import SukusuteNet
from torch.utils.data import DataLoader

# データセットを利用してモデルの学習を行う

N = 1000 # 学習用のダミーデータの特徴量の標本サイズ
X, y = create_dummy_data(N) # ダミーデータの生成

dataset = SukusuteDataset(X,y) # PyTorch用のデータセットを作成

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

model = SukusuteNet()
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    total_loss = 0

    for x_batch,y_batch in loader:
        pred = model(x_batch)
        loss = criterion(pred,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(epoch,total_loss)

# できた学習モデルをsavedディレクトリへ保存する（上書き保存）
torch.save(model.state_dict(), "models/saved/mlp_prototype.pth")
print("モデルを保存しました")