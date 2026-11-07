"""Train five classifiers on the UCI Adult Income dataset."""
from pathlib import Path
import io
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
DATA_PATH = DATA_DIR / "adult.data"
DATA_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
TARGET = "income"
RANDOM_STATE = 42
TEST_SIZE = 0.20

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    TARGET,
]


def download_dataset() -> Path:
    """Download the UCI archive if the dataset is not already in data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return DATA_PATH

    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        data_member = next(
            name for name in archive.namelist()
            if name.lower().endswith("adult.data")
        )
        DATA_PATH.write_bytes(archive.read(data_member))
    return DATA_PATH


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read UCI's headerless CSV and normalize its binary income target."""
    frame = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )
    labels = frame.pop(TARGET).astype(str).str.strip().str.rstrip(".")
    target = labels.map({"<=50K": 0, ">50K": 1})
    if target.isna().any():
        unexpected = labels[target.isna()].drop_duplicates().tolist()
        raise ValueError(f"Unexpected income labels: {unexpected}")
    if frame.shape[1] < 12 or len(frame) < 500:
        raise ValueError("The assignment requires at least 12 features and 500 rows.")
    return frame, target.astype(int)


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_columns),
        ("categorical", categorical, categorical_columns),
    ])


def get_auc_scores(model, features: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def main() -> None:
    for folder in (MODEL_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    features, target = load_dataset(download_dataset())
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    classifiers = {
        "logistic_regression": (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "decision_tree": (
            "Decision Tree",
            DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "knn": (
            "K-Nearest Neighbors",
            KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
        ),
        "naive_bayes": ("Gaussian Naive Bayes", GaussianNB()),
        "svm": (
            "Support Vector Machine",
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=10000),
        ),
    }

    rows = []
    for key, (display_name, classifier) in classifiers.items():
        print(f"Training {display_name}...")
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        scores = get_auc_scores(pipeline, X_test)
        rows.append({
            "Model": display_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(y_test, scores),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions),
        })
        joblib.dump(pipeline, MODEL_DIR / f"{key}.joblib", compress=3)

    results = pd.DataFrame(rows).sort_values("F1", ascending=False)
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    test_data = X_test.copy()
    test_data[TARGET] = y_test.map({0: "<=50K", 1: ">50K"})
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    print("\nEvaluation results on the stratified holdout set:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved labelled test data to {ROOT / 'test_data.csv'}")
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()