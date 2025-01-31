import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from numpy import linspace
from sklearn.pipeline import Pipeline


# **Feature Selection (Optimized)**
SELECTED_FEATURES = [
    "Tenure in Months", "Monthly Charge", "Total Charges", "CLTV",
    "Contract_One Year", "Contract_Two Year",
    "Online Security", "Streaming TV",
    "Payment Method_Credit Card", "Payment Method_Mailed Check",
    "Paperless Billing", "RevenuePerMonth"
]

TARGET_COLUMN = "Churn"
POST_CHURN_COLUMNS = ["Churn Category", "Churn Reason", "Churn Score", "Customer Status"]

def log_transform(X):
    """Log transform skewed numeric features."""
    return np.log1p(X)

def load_and_combine(train_path, val_path):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    df_train.drop(columns=POST_CHURN_COLUMNS, errors="ignore", inplace=True)
    df_val.drop(columns=POST_CHURN_COLUMNS, errors="ignore", inplace=True)

    df_train.columns = df_train.columns.str.strip()
    df_val.columns = df_val.columns.str.strip()

    df_train = df_train[SELECTED_FEATURES + [TARGET_COLUMN]]
    df_val = df_val[SELECTED_FEATURES + [TARGET_COLUMN]]

    df_combined = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    return df_train, df_val, df_combined

def build_preprocessing_pipeline():
    """
    Builds a preprocessing pipeline.
    """
    numeric_features = ["Tenure in Months", "Monthly Charge", "Total Charges", "CLTV", "RevenuePerMonth"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("log_transform", FunctionTransformer(log_transform)),  # Apply log transformation
                ("binning", KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")),  # Bin numeric features
                ("scaler", StandardScaler())
            ]), numeric_features)
        ]
    )

    return preprocessor

def preprocess_and_train(X_train, y_train):
    """
    Applies preprocessing and trains Logistic Regression.
    Uses class weights instead of undersampling.
    """
    preprocessor = build_preprocessing_pipeline()
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")  # Use balanced weights

    model.fit(X_train_preprocessed, y_train)

    return preprocessor, model

def crossval_threshold_tuning(df_combined, n_splits=5):
    """
    Finds the best decision threshold using cross-validation.
    """
    X = df_combined.drop(TARGET_COLUMN, axis=1)
    y = df_combined[TARGET_COLUMN].values

    preprocessor = build_preprocessing_pipeline()
    X_preprocessed = preprocessor.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true, all_y_proba = [], []

    for train_idx, val_idx in skf.split(X_preprocessed, y):
        X_train_fold, X_val_fold = X_preprocessed[train_idx], X_preprocessed[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_val_proba_fold = model.predict_proba(X_val_fold)[:, 1]

        all_y_true.extend(y_val_fold)
        all_y_proba.extend(y_val_proba_fold)

    all_y_true, all_y_proba = np.array(all_y_true), np.array(all_y_proba)

    thresholds = linspace(0, 1, 101)
    best_threshold, best_f1 = 0.70, -1  # Adjusted threshold for better precision

    for th in thresholds:
        preds = (all_y_proba >= th).astype(int)
        curr_f1 = f1_score(all_y_true, preds)
        if curr_f1 > best_f1:
            best_f1, best_threshold = curr_f1, th

    return best_threshold, best_f1

def final_training_and_evaluation(train_path, val_path, test_path):
    df_train, df_val, df_combined = load_and_combine(train_path, val_path)
    df_test = pd.read_csv(test_path).drop(columns=POST_CHURN_COLUMNS, errors="ignore")
    df_test.columns = df_test.columns.str.strip()
    df_test = df_test[SELECTED_FEATURES + [TARGET_COLUMN]]

    best_threshold, best_f1 = crossval_threshold_tuning(df_combined)
    print(f"Best threshold: {best_threshold:.2f} | F1 Score: {best_f1:.3f}")

    preprocessor, model = preprocess_and_train(df_combined.drop(TARGET_COLUMN, axis=1), df_combined[TARGET_COLUMN])

    X_test, y_test = df_test.drop(TARGET_COLUMN, axis=1), df_test[TARGET_COLUMN].values
    X_test_preprocessed = preprocessor.transform(X_test)

    y_test_proba = model.predict_proba(X_test_preprocessed)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print("\n=== Final Test Set Metrics ===")
    print(f"Accuracy: {test_acc:.3f} | Precision: {test_prec:.3f} | Recall: {test_rec:.3f}")
    print(f"F1 Score: {test_f1:.3f} | ROC AUC: {test_auc:.3f}")

    # ✅ **Save the model and threshold**
    joblib.dump({"preprocessor": preprocessor, "model": model}, "Models/final_best_model.pkl")
    joblib.dump({"threshold": best_threshold}, "Models/final_threshold.pkl")

    print("\n✅ Model saved successfully: models/final_best_model.pkl")
    print("✅ Threshold saved successfully: models/final_threshold.pkl")


if __name__ == "__main__":
    final_training_and_evaluation(
        "/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/train_ready.csv",
        "/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/val_ready.csv",
        "/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/test_ready.csv"
    )
