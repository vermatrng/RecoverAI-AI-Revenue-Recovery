
from pathlib import Path
import random
import csv
import pickle

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
except ImportError as e:
    raise SystemExit("Install dependencies first: pip install -r backend/requirements.txt")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
MODEL = ROOT / "ml" / "recovery_model.pkl"

random.seed(42)
methods = ["UPI", "Card", "Net Banking", "Debit Card"]
reasons = ["Bank timeout", "Network error", "Insufficient balance", "Card declined", "Authentication failed"]

rows = []
for i in range(5000):
    amount = random.randint(200, 25000)
    method = random.choice(methods)
    reason = random.choice(reasons)
    previous_successes = random.randint(0, 20)
    retry_count = random.randint(0, 3)
    hour = random.randint(0, 23)

    score = 0.35
    if reason in ["Bank timeout", "Network error"]:
        score += 0.25
    if previous_successes >= 10:
        score += 0.20
    elif previous_successes >= 5:
        score += 0.10
    if retry_count == 0:
        score += 0.08
    elif retry_count >= 2:
        score -= 0.18
    if method == "UPI":
        score += 0.05
    if 16 <= hour <= 19:
        score += 0.04
    if amount > 18000:
        score -= 0.03

    p = max(0.03, min(0.97, score))
    recovered = 1 if random.random() < p else 0

    rows.append([amount, method, reason, previous_successes, retry_count, hour, recovered])

DATA.parent.mkdir(parents=True, exist_ok=True)
with DATA.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["amount","payment_method","failure_reason","previous_successes","retry_count","hour","recovered"])
    w.writerows(rows)

X = [r[:-1] for r in rows]
y = [r[-1] for r in rows]

import pandas as pd
df = pd.DataFrame(rows, columns=["amount","payment_method","failure_reason","previous_successes","retry_count","hour","recovered"])
X = df.drop(columns=["recovered"])
y = df["recovered"]

cat = ["payment_method","failure_reason"]
num = ["amount","previous_successes","retry_count","hour"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", "passthrough", num)
])

model = Pipeline([
    ("pre", pre),
    ("rf", RandomForestClassifier(
        n_estimators=220, max_depth=12, random_state=42,
        class_weight="balanced"
    ))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
acc = accuracy_score(yte, pred)

with MODEL.open("wb") as f:
    pickle.dump(model, f)

print(f"Dataset: {DATA}")
print(f"Model:   {MODEL}")
print(f"Validation accuracy: {acc:.3f}")
