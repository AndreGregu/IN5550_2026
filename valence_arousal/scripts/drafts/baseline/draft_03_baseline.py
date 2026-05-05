import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATH SETUP
# ============================================================

# __file__ is the path to this script.
# resolve() makes it absolute.
# parents[1] moves one level up from scripts/ to the project root.
ROOT = Path(__file__).resolve().parents[1]

# Input paths
train_path = ROOT / "datasets" / "train.csv"
test_path = ROOT / "datasets" / "test.csv"

# Output path for the final submission file
out_path = ROOT / "outputs" / "submission_tfidf_user_split.csv"

# Create outputs/ folder if it does not already exist
out_path.parent.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

# Load train and test data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Replace missing text values with empty strings.
# This prevents TF-IDF from crashing on NaN values.
train["text"] = train["text"].fillna("")
test["text"] = test["text"].fillna("")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Apply the same feature engineering to both train and test.
for df in [train, test]:

    # Number of characters in the text.
    # Longer texts may contain more emotional/contextual information.
    df["text_length"] = df["text"].str.len()

    # Number of whitespace-separated words.
    # Useful because essays and word lists differ strongly in length.
    df["word_count"] = df["text"].str.split().str.len()

    # Convert Boolean is_words column to integer:
    # False -> 0 = essay
    # True  -> 1 = feeling-word list
    df["is_words"] = df["is_words"].astype(int)

    # Count exclamation marks.
    # This is a simple intensity/arousal cue.
    df["exclamation_count"] = df["text"].str.count("!")

    # Count question marks.
    # Questions may indicate uncertainty, stress, or emotional activation.
    df["question_count"] = df["text"].str.count(r"\?")

    # Count uppercase letters.
    # More uppercase letters can indicate stronger emotional expression.
    df["uppercase_count"] = df["text"].str.count(r"[A-Z]")

    # Ratio of uppercase letters to total text length.
    # +1 avoids division by zero for empty strings.
    df["uppercase_ratio"] = df["uppercase_count"] / (df["text_length"] + 1)


# ============================================================
# USER BASELINE FEATURES
# ============================================================

# Compute each user's average valence and arousal from the training set.
# This captures stable user-level emotional tendencies.
#
# Example:
# A user who is generally positive in training may also be more positive in test.
user_means = train.groupby("user_id")[["valence", "arousal"]].mean()

# Rename columns so they can be used as features
user_means.columns = ["user_valence_mean", "user_arousal_mean"]

# Add user mean features to the training data.
# Since these are computed from training labels, they are known for train users.
train = train.merge(user_means, on="user_id", how="left")

# Add user mean features to the test data.
# Some test users may not exist in train, so their values will become NaN.
test = test.merge(user_means, on="user_id", how="left")

# Compute global training means.
# These are used as fallback values for unseen test users.
global_valence_mean = train["valence"].mean()
global_arousal_mean = train["arousal"].mean()

# Fill missing user means in the test set.
# This happens when a test user was not present in the training set.
test["user_valence_mean"] = test["user_valence_mean"].fillna(global_valence_mean)
test["user_arousal_mean"] = test["user_arousal_mean"].fillna(global_arousal_mean)


# ============================================================
# DEFINE INPUT FEATURES AND TARGETS
# ============================================================

# All model input columns.
# "text" is handled separately by TF-IDF.
# The remaining features are numeric metadata/user/intensity features.
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

# Numeric features that should be standardized.
# Text is excluded because it is processed with TF-IDF instead.
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

# The model predicts two target values:
# valence = how positive/negative the emotion is
# arousal = how energized/calm the emotion is
target_cols = ["valence", "arousal"]


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def build_model():
    """
    Builds a complete machine learning pipeline.

    The pipeline does three things:
    1. Converts text into TF-IDF features.
    2. Standardizes numeric features.
    3. Trains Ridge regression models to predict valence and arousal.

    A fresh model is returned each time because we train separate models
    for essays and feeling-word lists.
    """

    # ColumnTransformer lets us apply different preprocessing
    # to different columns in the same dataframe.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                # Name of this transformer
                "text",

                # TF-IDF converts text into weighted word/ngram features.
                # It captures how informative words and phrases are.
                TfidfVectorizer(
                    lowercase=True,       # Convert text to lowercase
                    ngram_range=(1, 2),   # Use unigrams and bigrams
                    min_df=2,             # Ignore terms appearing only once
                    max_features=20000,   # Keep at most 20,000 text features
                ),

                # Apply TF-IDF to the text column
                "text",
            ),
            (
                # Name of this transformer
                "numeric",

                # Standardize numeric features to mean 0 and variance 1.
                # This helps Ridge regression treat features more fairly.
                StandardScaler(),

                # Apply scaling to these numeric columns
                numeric_features,
            ),
        ]
    )

    # Ridge is a linear regression model with L2 regularization.
    # Regularization helps prevent overfitting, especially with many TF-IDF features.
    ridge_model = Ridge(alpha=1.0)

    # MultiOutputRegressor allows Ridge to predict two targets:
    # one model for valence and one model for arousal.
    model = MultiOutputRegressor(
        Pipeline(
            [
                # First preprocess the raw input columns
                ("features", preprocessor),

                # Then fit Ridge regression on the processed features
                ("regressor", ridge_model),
            ]
        )
    )

    return model


# ============================================================
# TRAIN SEPARATE MODELS FOR EACH TEXT TYPE
# ============================================================

# Store predictions from both text-type-specific models here
all_predictions = []

# The dataset has two different text formats:
# is_words = 0 -> essay
# is_words = 1 -> feeling-word list
#
# We train separate models because these two formats are linguistically different.
for is_words_value in [0, 1]:

    # Select only one text type from train and test
    train_part = train[train["is_words"] == is_words_value].copy()
    test_part = test[test["is_words"] == is_words_value].copy()

    # Build a fresh model for this text type
    model = build_model()

    # Input features for this subset
    X_train = train_part[features]

    # Target values for this subset
    y_train = train_part[target_cols]

    # Test features for this subset
    X_test = test_part[features]

    # Train model
    model.fit(X_train, y_train)

    # Predict valence and arousal for this subset
    preds = model.predict(X_test)

    # Create submission rows for this subset
    part_submission = pd.DataFrame(
        {
            "user_id": test_part["user_id"],
            "text_id": test_part["text_id"],

            # First prediction column is valence
            "pred_valence": preds[:, 0],

            # Second prediction column is arousal
            "pred_arousal": preds[:, 1],
        }
    )

    # Store this subset's predictions
    all_predictions.append(part_submission)


# ============================================================
# COMBINE PREDICTIONS
# ============================================================

# Combine predictions from:
# - essay model
# - feeling-word-list model
submission = pd.concat(all_predictions, axis=0)


# ============================================================
# RESTORE ORIGINAL TEST ORDER
# ============================================================

# Because we split the test set into two groups, prediction rows may no longer
# be in the same order as the original test.csv.
#
# This merge restores the original order using user_id and text_id.
submission = test[["user_id", "text_id"]].merge(
    submission,
    on=["user_id", "text_id"],
    how="left",
)


# ============================================================
# SAVE FINAL SUBMISSION
# ============================================================

# Save predictions in the format expected by the evaluator:
# user_id, text_id, pred_valence, pred_arousal
submission.to_csv(out_path, index=False)

print(f"Saved submission to {out_path}")
