from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np,pandas as pd

from config import LABEL_WEIGHTS_DEFAULT

# 相対距離の値・推論結果の一定時間での集計からある相手デバイスとの関連度スコアを算出する
# ある相手デバイスとの関連度の説明文も出力する

def calc_relatedness(csv_path,prediction_json=None,weight_map=None,out=None):
    weight_map=weight_map or LABEL_WEIGHTS_DEFAULT
    df=pd.read_csv(csv_path,encoding="utf-8-sig")
    rows=[]
    predictions={}
    if prediction_json and Path(prediction_json).exists():
        data=json.loads(Path(prediction_json).read_text(encoding="utf-8"))
        predictions=data.get("distance",{})
    for dc in [c for c in df.columns if re.fullmatch(r"Distance_\d+",str(c))]:
        lc=f"{dc}_Label"
        if lc not in df.columns: continue
        labels=df[lc].fillna("").astype(str).str.strip()
        w=labels.map(lambda s:weight_map.get(s,np.nan))
        known=w.notna()
        if not known.any(): continue
        score=float(w[known].mean())
        counts=labels[known].value_counts()
        total=int(known.sum())
        same_action=float(counts.get("接近（同じ行動）",0))/total
        conversation=float(counts.get("接近（会話）",0)+counts.get("接近（会話程度）",0))/total
        same_room=float(counts.get("接近（同じ部屋）",0)+counts.get("接近（同じ部屋程度）",0))/total
        # 推論結果があれば、現在の観測状態も付加
        pred=predictions.get(dc,[])
        current=pred[-1] if pred else None
        explanation=f"ID={dc.split('_')[1]}とは同じ行動の観測比率が{same_action*100:.1f}%です。"
        if current:
            explanation+=f" 直近の推論状態は「{current['label']}」です。"
        rows.append({
            "person_id":int(dc.split("_")[1]),
            "relatedness_score":score,
            "same_room_ratio":same_room,
            "conversation_ratio":conversation,
            "same_action_ratio":same_action,
            "latest_inferred_state":current,
            "explanation":explanation
        })
    if out: Path(out).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    return rows

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True)
    ap.add_argument("--prediction-json")
    ap.add_argument("--out",default="relatedness.json")
    args=ap.parse_args()
    print(json.dumps(calc_relatedness(args.csv,args.prediction_json,out=args.out),ensure_ascii=False,indent=2))
