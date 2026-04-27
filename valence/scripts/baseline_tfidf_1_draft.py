import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]

train_path = ROOT / "datasets" / "train.csv"
test_path = ROOT / "datasets" / "test.csv"
out_path = ROOT / "outputs" / "submission_tfidf.csv"

out_path.parent.mkdir(exist_ok=True)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Basic cleanup
train["text"] = train["text"].fillna("")
test["text"] = test["text"].fillna("")

# Targets
y = train[["valence", "arousal"]]

# Simple text model
model = MultiOutputRegressor(
    Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000
        )),
        ("regressor", Ridge(alpha=1.0))
    ])
)

model.fit(train["text"], y)

preds = model.predict(test["text"])

submission = pd.DataFrame({
    "user_id": test["user_id"],
    "text_id": test["text_id"],
    "pred_valence": preds[:, 0],
    "pred_arousal": preds[:, 1],
})

submission.to_csv(out_path, index=False)
print(f"Saved submission to {out_path}")

