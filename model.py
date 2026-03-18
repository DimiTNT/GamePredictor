"""
model.py
--------
Trains a Random Forest classifier to predict match outcomes (H / D / A).
Saves the trained model to models/rf_model.pkl.

Usage:
    python src/model.py
"""

import os
import sqlite3
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

DB_PATH   = "data/football.db"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/rf_model.pkl"
ENC_PATH   = f"{MODEL_DIR}/label_encoder.pkl"

FEATURES = [
    "HomeForm", "AwayForm",
    "HomeGoalsAvg", "AwayGoalsAvg",
    "HomeConcedeAvg", "AwayConcedeAvg",
]

TARGET = "FTR"   # H / D / A


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", con)
    con.close()
    return df


# ── Train ─────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)   # A=0, D=1, H=2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # ── Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n📊 Test accuracy : {acc:.1%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)

    # ── Feature importance
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(fi.round(3))

    return model, le


# ── Save ──────────────────────────────────────────────────────────────────────

def save(model, le):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENC_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"\n✅ Model saved → {MODEL_PATH}")


# ── Predict helper (used by app.py) ──────────────────────────────────────────

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENC_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le


def predict_match(home_team: str, away_team: str, df: pd.DataFrame):
    """
    Returns predicted outcome label + probabilities for a given fixture,
    using that team's most recent form stats from the dataset.
    """
    model, le = load_model()

    def latest_stats(team: str, home: bool) -> dict:
        mask = (df["HomeTeam"] == team) if home else (df["AwayTeam"] == team)
        row = df[mask].sort_values("Date").iloc[-1] if mask.any() else None
        if row is None:
            return {k: 1.2 for k in FEATURES}
        prefix = "Home" if home else "Away"
        return {
            f"{prefix}Form":        row.get(f"{prefix}Form", 1.2),
            f"{prefix}GoalsAvg":    row.get(f"{prefix}GoalsAvg", 1.2),
            f"{prefix}ConcedeAvg":  row.get(f"{prefix}ConcedeAvg", 1.2),
        }

    h = latest_stats(home_team, home=True)
    a = latest_stats(away_team, home=False)
    X = pd.DataFrame([{**h, **a}])[FEATURES]

    proba = model.predict_proba(X)[0]
    pred  = le.inverse_transform([proba.argmax()])[0]
    proba_dict = dict(zip(le.classes_, proba))

    return pred, proba_dict


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📂 Loading data...")
    df = load_data()
    print(f"   {len(df)} matches loaded")

    print("🤖 Training model...")
    model, le = train(df)
    save(model, le)
