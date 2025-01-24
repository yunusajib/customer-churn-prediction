import pandas as pd
import numpy as np
import joblib

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# For threshold tuning
from numpy import linspace

# Columns that might leak churn info (post-churn) - adjust as needed
POST_CHURN_COLUMNS = [
    "Churn Category",
    "Churn Reason",
    "Churn Score",
    "Customer Status",
    # Add anything else known only after churn
]

def load_and_combine(train_path, val_path):
    """
    Loads train.csv and val.csv, drops post-churn columns,
    combines them for cross-validation & threshold tuning.
    Returns combined DataFrame plus separate DataFrames
    for the original train and val in case needed.
    """
    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)

    # Drop columns that cause data leakage if they exist
    df_train.drop(columns=POST_CHURN_COLUMNS, errors='ignore', inplace=True)
    df_val.drop(columns=POST_CHURN_COLUMNS, errors='ignore', inplace=True)

    # Combine for cross-validation & threshold tuning
    df_combined = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    return df_train, df_val, df_combined

def build_pipeline(X):
    """
    Builds a ColumnTransformer + LogisticRegression pipeline.
    Automatically detects numeric vs. categorical columns.
    """
    # Identify numeric vs. categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    # Use class_weight='balanced' to address 70/30 imbalance
    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline

def crossval_threshold_tuning(df_combined, n_splits=5):
    """
    1. Perform Stratified K-Fold cross-validation on df_combined.
    2. For each fold, train the pipeline, then collect predicted probabilities.
    3. Evaluate multiple thresholds to find the best threshold (e.g., maximizing F1).
    4. Return the best threshold, plus average metrics across folds.
    """
    # Separate features and target
    X = df_combined.drop("Churn", axis=1)
    y = df_combined["Churn"].values

    pipeline = build_pipeline(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # We'll keep track of the best threshold found across folds
    # A simpler approach: we can gather all probabilities and labels across all folds,
    # then find the single best threshold. Alternatively, we do it fold by fold.
    all_y_true = []
    all_y_proba = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        pipeline.fit(X_train_fold, y_train_fold)
        y_val_proba_fold = pipeline.predict_proba(X_val_fold)[:, 1]

        # Collect for global threshold search
        all_y_true.extend(y_val_fold)
        all_y_proba.extend(y_val_proba_fold)

    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)

    # Now we find the best threshold on combined out-of-fold predictions
    thresholds = linspace(0, 1, 101)  # 0.00, 0.01, 0.02, ..., 1.00
    best_threshold = 0.5
    best_f1 = -1

    for th in thresholds:
        preds = (all_y_proba >= th).astype(int)
        curr_f1 = f1_score(all_y_true, preds)
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_threshold = th

    # Optional: you can also track precision, recall, etc.
    return best_threshold, best_f1

def final_training_and_evaluation(train_path, val_path, test_path):
    """
    Full pipeline:
      1) Load and drop post-churn columns from train/val/test
      2) Combine train+val for cross-validation and threshold tuning
      3) Retrain pipeline on entire (train+val) with best threshold
      4) Evaluate on test
      5) Save pipeline and best threshold
    """
    # --------------------------
    # 1. Load Data
    # --------------------------
    df_train, df_val, df_combined = load_and_combine(train_path, val_path)

    df_test = pd.read_csv(test_path)
    # Drop post-churn columns in test as well
    df_test.drop(columns=POST_CHURN_COLUMNS, errors='ignore', inplace=True)

    # --------------------------
    # 2. Find Best Threshold Using Crossval on (train+val)
    # --------------------------
    best_threshold, best_f1 = crossval_threshold_tuning(df_combined)
    print(f"Best threshold from cross-validation: {best_threshold:.2f}")
    print(f"Corresponding F1 at that threshold:   {best_f1:.3f}")

    # --------------------------
    # 3. Retrain on (train+val) Entirely
    # --------------------------
    pipeline = build_pipeline(df_combined.drop("Churn", axis=1))
    pipeline.fit(df_combined.drop("Churn", axis=1), df_combined["Churn"])

    # --------------------------
    # 4. Evaluate on Test
    # --------------------------
    X_test = df_test.drop("Churn", axis=1)
    y_test = df_test["Churn"].values

    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    # Use best_threshold to generate final predictions
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    # Metrics
    test_acc  = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec  = recall_score(y_test, y_test_pred)
    test_f1   = f1_score(y_test, y_test_pred)
    test_auc  = roc_auc_score(y_test, y_test_proba)

    print("\n=== Final Test Set Metrics (using best threshold) ===")
    print(f"Accuracy:  {test_acc:.3f}")
    print(f"Precision: {test_prec:.3f}")
    print(f"Recall:    {test_rec:.3f}")
    print(f"F1 Score:  {test_f1:.3f}")
    print(f"ROC AUC:   {test_auc:.3f}")

    # --------------------------
    # 5. Save Pipeline and Best Threshold
    # --------------------------
    joblib.dump(pipeline, "models/final_logreg_pipeline.pkl")
    # Save threshold in a small file or dict
    joblib.dump({"threshold": best_threshold}, "models/final_threshold.pkl")

    print("\nModel pipeline saved to models/final_logreg_pipeline.pkl")
    print(f"Best threshold saved to models/final_threshold.pkl = {best_threshold:.2f}")

if __name__ == "__main__":
    train_csv = "/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/train_ready.csv"
    val_csv   = "/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/val_ready.csv"
    test_csv  = "/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/test_ready.csv"


    final_training_and_evaluation(train_csv, val_csv, test_csv)

