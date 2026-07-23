import torch
import torch.nn as nn

import config
from dataset.dataset import SukusuteDataset
from features.dummy_features import create_dummy_data
from models.mlp_prototype import SukusuteNet
from torch.utils.data import DataLoader

# データセットを利用してモデルの学習を行う

X, y = create_dummy_data(config.DUMMY_DATA_SIZE) # ダミーデータの生成

dataset = SukusuteDataset(X,y) # PyTorch用のデータセットを作成

loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

model = SukusuteNet()
model.train()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config.LEARNING_RATE
)

for epoch in range(config.EPOCHS):
    total_loss = 0

    for x_batch,y_batch in loader:
        pred = model(x_batch)
        loss = criterion(pred,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(epoch,total_loss)

# できた学習モデルの保存（上書き保存）
torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
print("モデルを保存しました")