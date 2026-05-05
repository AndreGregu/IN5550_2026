import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

tfidf_path = ROOT / "outputs" / "submission_tfidf_user_split.csv"
transformer_path = ROOT / "outputs" / "submission_transformer_split.csv"
out_path = ROOT / "outputs" / "submission_hybrid.csv"

tfidf = pd.read_csv(tfidf_path)
trans = pd.read_csv(transformer_path)

df = tfidf.merge(
    trans,
    on=["user_id", "text_id"],
    suffixes=("_tfidf", "_trans"),
)

w_tfidf = 0.4
w_trans = 0.6

df["pred_valence"] = (
    w_tfidf * df["pred_valence_tfidf"]
    + w_trans * df["pred_valence_trans"]
)

df["pred_arousal"] = (
    w_tfidf * df["pred_arousal_tfidf"]
    + w_trans * df["pred_arousal_trans"]
)

df["pred_valence"] = df["pred_valence"].clip(-2, 2)
df["pred_arousal"] = df["pred_arousal"].clip(0, 2)

df[["user_id", "text_id", "pred_valence", "pred_arousal"]].to_csv(
    out_path,
    index=False,
)

print(f"Saved hybrid submission to {out_path}")
