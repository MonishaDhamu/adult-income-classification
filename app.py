from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "income"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Adult Income Classifier", page_icon="📊", layout="wide")
st.title("Adult Income Classification")
st.write(
    "Compare five saved classifiers on labeled Adult Income test data. "
    "The app evaluates uploaded rows and never retrains models."
)

available_models = {
    name: MODEL_DIR / filename
    for name, filename in MODEL_FILES.items()
    if (MODEL_DIR / filename).is_file()
}
if not available_models:
    st.error("No saved models found. Run `python train_models.py` from this project folder first.")
    st.stop()

uploaded = st.file_uploader(
    "Upload labeled test data (CSV)",
    type=["csv"],
    help="Include all 14 input columns and the income target column.",
)
if uploaded is not None:
    try:
        test_data = pd.read_csv(uploaded, skipinitialspace=True, na_values="?")
    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        st.stop()
    source_label = uploaded.name
else:
    bundled_test = ROOT / "test_data.csv"
    if not bundled_test.exists():
        st.info("Upload a labeled test CSV, or run the training script to create test_data.csv.")
        st.stop()
    test_data = pd.read_csv(bundled_test, skipinitialspace=True, na_values="?")
    source_label = "Bundled test_data.csv"

if TARGET not in test_data.columns:
    st.error(f"The CSV must contain the `{TARGET}` target column.")
    st.stop()

target_labels = test_data[TARGET].astype(str).str.strip().str.rstrip(".")
y_true = target_labels.map({"<=50K": 0, ">50K": 1})
if y_true.isna().any():
    invalid_labels = target_labels[y_true.isna()].drop_duplicates().tolist()
    st.error(f"Unexpected income labels: {invalid_labels}")
    st.stop()
y_true = y_true.astype(int)
X = test_data.drop(columns=[TARGET])


def evaluate_model(model_path: Path):
    model = joblib.load(model_path)
    expected_features = list(model.feature_names_in_)
    missing_features = [name for name in expected_features if name not in X.columns]
    if missing_features:
        raise ValueError("Missing input columns: " + ", ".join(missing_features))
    model_features = X.loc[:, expected_features]
    predictions = np.asarray(model.predict(model_features), dtype=int)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(model_features)[:, 1]
    else:
        scores = model.decision_function(model_features)
    auc = roc_auc_score(y_true, scores) if y_true.nunique() == 2 else np.nan
    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": auc,
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predictions),
    }
    return predictions, metrics


st.caption(f"Data: {source_label} · {len(test_data):,} rows")
comparison_rows = []
model_errors = {}
for model_name, model_path in available_models.items():
    try:
        _, model_metrics = evaluate_model(model_path)
        comparison_rows.append({"Model": model_name, **model_metrics})
    except Exception as error:
        model_errors[model_name] = str(error)

if model_errors:
    for name, message in model_errors.items():
        st.warning(f"{name} could not be evaluated: {message}")
if not comparison_rows:
    st.error("None of the saved models could evaluate this CSV.")
    st.stop()

st.subheader("Model comparison")
comparison = pd.DataFrame(comparison_rows).sort_values("F1", ascending=False)
st.dataframe(
    comparison.style.format({column: "{:.4f}" for column in comparison.columns if column != "Model"}),
    use_container_width=True,
    hide_index=True,
)

selected_model = st.selectbox("Inspect a model", list(available_models))
try:
    predictions, selected_metrics = evaluate_model(available_models[selected_model])
except Exception as error:
    st.error(f"Could not evaluate {selected_model}: {error}")
    st.stop()

st.subheader(f"{selected_model} details")
metric_columns = st.columns(6)
for column, metric_name in zip(metric_columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    value = selected_metrics[metric_name]
    column.metric(metric_name, "N/A" if pd.isna(value) else f"{value:.4f}")

left, right = st.columns([1, 1])
with left:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=["<=50K", ">50K"]).plot(
        ax=axis, colorbar=False, cmap="Blues", values_format="d"
    )
    axis.set_title(selected_model)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
with right:
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["<=50K", ">50K"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

predictions_data = X.copy()
predictions_data["Actual income"] = y_true.map({0: "<=50K", 1: ">50K"}).values
predictions_data["Predicted income"] = pd.Series(predictions, index=X.index).map({0: "<=50K", 1: ">50K"})
st.download_button(
    "Download predictions",
    predictions_data.to_csv(index=False),
    file_name="adult_income_predictions.csv",
    mime="text/csv",
)