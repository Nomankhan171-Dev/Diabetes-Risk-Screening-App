
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_FILE = Path("diabetes.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"

if not DATA_FILE.exists():
    raise SystemExit(
        "Missing diabetes.csv. Add a dataset with columns: "
        "Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "
        "DiabetesPedigreeFunction, Age, Outcome"
    )

df = pd.read_csv(DATA_FILE)

required = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

X = df[required[:-1]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print("Saved model to:", MODEL_PATH)
