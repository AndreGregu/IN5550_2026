import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "datasets" / "train.csv"
TEST_PATH = ROOT / "datasets" / "test.csv"
OUT_PATH = ROOT / "outputs" / "submission_transformer_split.csv"

MODEL_NAME = "/fp/projects01/ec403/hf_models/models--distilbert-base-multilingual-cased/snapshots/45c032ab32cc946ad88a166f7cb282f58c753c2e"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-5
WEIGHT_DECAY = 0.01
SEED = 42
VAL_SIZE = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train["text"] = train["text"].fillna("")
test["text"] = test["text"].fillna("")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%m-%d-%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.sort_values(["user_id", "timestamp"])

    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["is_words"] = df["is_words"].astype(int)

    df["exclamation_count"] = df["text"].str.count("!")
    df["question_count"] = df["text"].str.count(r"\?")
    df["uppercase_count"] = df["text"].str.count(r"[A-Z]")
    df["uppercase_ratio"] = df["uppercase_count"] / (df["text_length"] + 1)

    df["time_since_first"] = (
        df.groupby("user_id")["timestamp"]
        .transform(lambda x: (x - x.min()).dt.total_seconds() / 86400)
        .fillna(0)
    )

    df["time_since_previous"] = (
        df.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(86400)
        .fillna(0)
    )

    return df


train = add_features(train)
test = add_features(test)


# ============================================================
# USER BASELINE FEATURES
# ============================================================

user_means = train.groupby("user_id")[["valence", "arousal"]].mean()
user_means.columns = ["user_valence_mean", "user_arousal_mean"]

train = train.merge(user_means, on="user_id", how="left")
test = test.merge(user_means, on="user_id", how="left")

test["user_valence_mean"] = test["user_valence_mean"].fillna(train["valence"].mean())
test["user_arousal_mean"] = test["user_arousal_mean"].fillna(train["arousal"].mean())


NUMERIC_FEATURES = [
    "text_length",
    "word_count",
    "is_words",
    "collection_phase",
    "exclamation_count",
    "question_count",
    "uppercase_count",
    "uppercase_ratio",
    "time_since_first",
    "time_since_previous",
    "user_valence_mean",
    "user_arousal_mean",
]

TARGET_COLS = ["valence", "arousal"]


# ============================================================
# DATASET
# ============================================================

class EmotionDataset(Dataset):
    def __init__(self, texts, numeric_features, tokenizer, targets=None):
        self.texts = list(texts)
        self.numeric_features = numeric_features
        self.tokenizer = tokenizer
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "numeric": self.numeric_features[idx],
        }

        if self.targets is not None:
            item["labels"] = self.targets[idx]

        return item


# ============================================================
# MODEL
# ============================================================

class TransformerRegressor(nn.Module):
    def __init__(self, model_name, num_numeric_features):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(hidden_size + num_numeric_features, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask, numeric):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_embedding, numeric], dim=1)

        return self.regressor(combined)


# ============================================================
# TRAIN / VALIDATE HELPERS
# ============================================================

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        numeric = batch["numeric"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        preds = model(input_ids, attention_mask, numeric)
        loss = loss_fn(preds, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_loss(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            numeric = batch["numeric"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            preds = model(input_ids, attention_mask, numeric)
            loss = loss_fn(preds, labels)

            total_loss += loss.item()

    return total_loss / len(loader)


def predict(model, loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            numeric = batch["numeric"].to(DEVICE)

            preds = model(input_ids, attention_mask, numeric)
            all_preds.append(preds.cpu().numpy())

    return np.vstack(all_preds)


# ============================================================
# MAIN TRAINING LOOP: SEPARATE MODELS FOR ESSAYS AND WORD LISTS
# ============================================================

print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

all_predictions = []

for is_words_value in [0, 1]:
    print("\n" + "=" * 60)
    print(f"Training model for is_words={is_words_value}")
    print("=" * 60)

    train_part = train[train["is_words"] == is_words_value].copy()
    test_part = test[test["is_words"] == is_words_value].copy()

    train_part = train_part.reset_index(drop=True)
    test_part = test_part.reset_index(drop=True)

    train_split, val_split = train_test_split(
        train_part,
        test_size=VAL_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    scaler = StandardScaler()

    train_numeric = torch.tensor(
        scaler.fit_transform(train_split[NUMERIC_FEATURES]),
        dtype=torch.float32,
    )

    val_numeric = torch.tensor(
        scaler.transform(val_split[NUMERIC_FEATURES]),
        dtype=torch.float32,
    )

    test_numeric = torch.tensor(
        scaler.transform(test_part[NUMERIC_FEATURES]),
        dtype=torch.float32,
    )

    train_targets = torch.tensor(
        train_split[TARGET_COLS].values,
        dtype=torch.float32,
    )

    val_targets = torch.tensor(
        val_split[TARGET_COLS].values,
        dtype=torch.float32,
    )

    train_dataset = EmotionDataset(
        train_split["text"],
        train_numeric,
        tokenizer,
        train_targets,
    )

    val_dataset = EmotionDataset(
        val_split["text"],
        val_numeric,
        tokenizer,
        val_targets,
    )

    test_dataset = EmotionDataset(
        test_part["text"],
        test_numeric,
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TransformerRegressor(
        MODEL_NAME,
        num_numeric_features=len(NUMERIC_FEATURES),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    num_training_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            loss_fn,
        )

        print(
            f"is_words={is_words_value} | "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    print(f"Best validation loss for is_words={is_words_value}: {best_val_loss:.4f}")

    model.load_state_dict(best_state_dict)
    model.to(DEVICE)

    preds = predict(model, test_loader)

    part_submission = pd.DataFrame(
        {
            "user_id": test_part["user_id"].values,
            "text_id": test_part["text_id"].values,
            "pred_valence": preds[:, 0],
            "pred_arousal": preds[:, 1],
        }
    )

    all_predictions.append(part_submission)


# ============================================================
# SAVE SUBMISSION
# ============================================================

submission = pd.concat(all_predictions, axis=0)

submission = test[["user_id", "text_id"]].merge(
    submission,
    on=["user_id", "text_id"],
    how="left",
)

OUT_PATH.parent.mkdir(exist_ok=True)
submission.to_csv(OUT_PATH, index=False)

print(f"\nSaved submission to {OUT_PATH}")
