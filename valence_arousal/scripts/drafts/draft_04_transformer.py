import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "datasets" / "train.csv"
TEST_PATH = ROOT / "datasets" / "test.csv"
OUT_PATH = ROOT / "outputs" / "submission_transformer.csv"

MODEL_NAME = "/fp/projects01/ec403/hf_models/models--distilbert-base-multilingual-cased/snapshots/45c032ab32cc946ad88a166f7cb282f58c753c2e"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

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
        errors="coerce"
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


scaler = StandardScaler()

train_numeric = scaler.fit_transform(train[NUMERIC_FEATURES])
test_numeric = scaler.transform(test[NUMERIC_FEATURES])

train_numeric = torch.tensor(train_numeric, dtype=torch.float32)
test_numeric = torch.tensor(test_numeric, dtype=torch.float32)

train_targets = torch.tensor(
    train[["valence", "arousal"]].values,
    dtype=torch.float32
)


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
            nn.Dropout(0.2),
            nn.Linear(hidden_size + num_numeric_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
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
# TRAINING SETUP
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = EmotionDataset(
    train["text"],
    train_numeric,
    tokenizer,
    train_targets,
)

test_dataset = EmotionDataset(
    test["text"],
    test_numeric,
    tokenizer,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
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


# ============================================================
# TRAIN
# ============================================================

print(f"Using device: {DEVICE}")
print(f"Training examples: {len(train_dataset)}")
print(f"Test examples: {len(test_dataset)}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        numeric = batch["numeric"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        preds = model(input_ids, attention_mask, numeric)
        loss = loss_fn(preds, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f}")


# ============================================================
# PREDICT
# ============================================================

model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        numeric = batch["numeric"].to(DEVICE)

        preds = model(input_ids, attention_mask, numeric)
        all_preds.append(preds.cpu().numpy())

preds = np.vstack(all_preds)


# ============================================================
# SAVE SUBMISSION
# ============================================================

submission = pd.DataFrame(
    {
        "user_id": test["user_id"].values,
        "text_id": test["text_id"].values,
        "pred_valence": preds[:, 0],
        "pred_arousal": preds[:, 1],
    }
)

OUT_PATH.parent.mkdir(exist_ok=True)
submission.to_csv(OUT_PATH, index=False)

print(f"Saved submission to {OUT_PATH}")
