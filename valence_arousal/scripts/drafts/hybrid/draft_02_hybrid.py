import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

tfidf = pd.read_csv(ROOT / "outputs" / "submission_tfidf_user_split.csv")
trans = pd.read_csv(ROOT / "outputs" / "submission_transformer.csv")

# Merge
df = tfidf.merge(
    trans,
    on=["user_id", "text_id"],
    suffixes=("_tfidf", "_trans")
)

# Weighted average (tune these!)
w_tfidf = 0.4
w_trans = 0.6

df["pred_valence"] = (
    w_tfidf * df["pred_valence_tfidf"] +
    w_trans * df["pred_valence_trans"]
)

df["pred_arousal"] = (
    w_tfidf * df["pred_arousal_tfidf"] +
    w_trans * df["pred_arousal_trans"]
)

out_path = ROOT / "outputs" / "submission_hybrid.csv"

df[["user_id", "text_id", "pred_valence", "pred_arousal"]].to_csv(out_path, index=False)

print(f"Saved hybrid submission to {out_path}")
