
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

train_path = ROOT / "datasets" / "train.csv"
test_path = ROOT / "datasets" / "test.csv"
out_path = ROOT / "outputs" / "submission_tfidf.csv"

out_path.parent.mkdir(exist_ok=True)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train["text"] = train["text"].fillna("")
test["text"] = test["text"].fillna("")

for df in [train, test]:
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["is_words"] = df["is_words"].astype(int)

features = ["text", "text_length", "word_count", "is_words", "collection_phase"]
y = train[["valence", "arousal"]]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=20000,
            ),
            "text",
        ),
        (
            "numeric",
            StandardScaler(),
            ["text_length", "word_count", "is_words", "collection_phase"],
        ),
    ]
)

model = MultiOutputRegressor(
    Pipeline(
        [
            ("features", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
)

model.fit(train[features], y)
preds = model.predict(test[features])

submission = pd.DataFrame(
    {
        "user_id": test["user_id"],
        "text_id": test["text_id"],
        "pred_valence": preds[:, 0],
        "pred_arousal": preds[:, 1],
    }
)

submission.to_csv(out_path, index=False)
print(f"Saved submission to {out_path}")
