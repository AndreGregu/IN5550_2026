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
out_path = ROOT / "outputs" / "submission_tfidf_user_split.csv"

out_path.parent.mkdir(exist_ok=True)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train["text"] = train["text"].fillna("")
test["text"] = test["text"].fillna("")


# ----------------------------
# Feature engineering
# ----------------------------
for df in [train, test]:
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["is_words"] = df["is_words"].astype(int)

    # Simple arousal/intensity features
    df["exclamation_count"] = df["text"].str.count("!")
    df["question_count"] = df["text"].str.count(r"\?")
    df["uppercase_count"] = df["text"].str.count(r"[A-Z]")
    df["uppercase_ratio"] = df["uppercase_count"] / (df["text_length"] + 1)


# ----------------------------
# User baseline features
# Uses only training labels
# ----------------------------
user_means = train.groupby("user_id")[["valence", "arousal"]].mean()
user_means.columns = ["user_valence_mean", "user_arousal_mean"]

train = train.merge(user_means, on="user_id", how="left")
test = test.merge(user_means, on="user_id", how="left")

global_valence_mean = train["valence"].mean()
global_arousal_mean = train["arousal"].mean()

test["user_valence_mean"] = test["user_valence_mean"].fillna(global_valence_mean)
test["user_arousal_mean"] = test["user_arousal_mean"].fillna(global_arousal_mean)


features = [
    "text",
    "text_length",
    "word_count",
    "is_words",
    "collection_phase",
    "exclamation_count",
    "question_count",
    "uppercase_count",
    "uppercase_ratio",
    "user_valence_mean",
    "user_arousal_mean",
]

numeric_features = [
    "text_length",
    "word_count",
    "is_words",
    "collection_phase",
    "exclamation_count",
    "question_count",
    "uppercase_count",
    "uppercase_ratio",
    "user_valence_mean",
    "user_arousal_mean",
]

target_cols = ["valence", "arousal"]


def build_model():
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
                numeric_features,
            ),
        ]
    )

    return MultiOutputRegressor(
        Pipeline(
            [
                ("features", preprocessor),
                ("regressor", Ridge(alpha=1.0)),
            ]
        )
    )


# ----------------------------
# Train separate models
# One for essays, one for feeling-word lists
# ----------------------------
all_predictions = []

for is_words_value in [0, 1]:
    train_part = train[train["is_words"] == is_words_value].copy()
    test_part = test[test["is_words"] == is_words_value].copy()

    model = build_model()

    X_train = train_part[features]
    y_train = train_part[target_cols]

    X_test = test_part[features]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    part_submission = pd.DataFrame(
        {
            "user_id": test_part["user_id"],
            "text_id": test_part["text_id"],
            "pred_valence": preds[:, 0],
            "pred_arousal": preds[:, 1],
        }
    )

    all_predictions.append(part_submission)


submission = pd.concat(all_predictions, axis=0)

# Restore original test order
submission = test[["user_id", "text_id"]].merge(
    submission,
    on=["user_id", "text_id"],
    how="left",
)

submission.to_csv(out_path, index=False)

print(f"Saved submission to {out_path}")
