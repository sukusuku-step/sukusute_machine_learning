import torch

from models.classifier import SukusuteNet
from features.dummy_features import create_dummy_data

# サンプルデータを100個生成して学習モデルを評価するためのプログラム

if __name__ == "__main__":
    # 学習済みモデルを読み込む
    model = SukusuteNet()
    model.load_state_dict(torch.load("models/saved/mlp_prototype.pth"))

    # 推論モード
    model.eval()

    # サンプルデータを100個ランダム生成
    X_test, y_test = create_dummy_data(100)
    sample = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        output = model(sample)
        prediction = torch.argmax(output, dim=1) # 学習モデルによる推論結果を得る

    y_test = torch.tensor(y_test, dtype=torch.long)
    accuracy = (prediction == y_test).float().mean() # 正解率を計算

    labels = ["元気", "いつもよりも元気がない", "転倒の可能性"]

    # 正解ラベルと予測結果を表示
    print("No | 正解 | 予測")
    print("-" * 30)
    for i in range(len(y_test)):
        print(
            f"{i:3d} | "
            f"{labels[y_test[i].item()]} | "
            f"{labels[prediction[i].item()]}"
        )

    print(f"Accuracy: {accuracy.item() * 100:.2f}%") # 正解率を表示