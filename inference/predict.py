import torch

from models.mlp_prototype import SukusuteNet
from features.dummy_features import create_dummy_data

# サンプルデータを1つ生成して推論結果を見るためのプログラム

if __name__ == "__main__":
    # 学習済みモデルを読み込む
    model = SukusuteNet()
    model.load_state_dict(torch.load("models/saved/mlp_prototype.pth"))

    # 推論モード
    model.eval()

    # サンプルデータをランダム生成
    X_test, y_test = create_dummy_data(1)
    sample = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        output = model(sample)
        prediction = torch.argmax(output).item() # 学習モデルによる推論結果を得る

    labels = ["元気", "いつもよりも元気がない", "転倒の可能性"]
    print(f"予測結果: {labels[prediction]}") # 推論結果