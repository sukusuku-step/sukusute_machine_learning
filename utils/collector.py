import pandas as pd
from pathlib import Path
import json

# ラベリング済みCSVを読み込む関数
def read_csv(path):
    # 日本語CSV/UTF-8系を優先し、Excel由来のUTF-8-SIGにも対応
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("CSV", b"", 0, 1, "Unsupported encoding")

# PythonデータをJSONファイルとして保存する関数
def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")