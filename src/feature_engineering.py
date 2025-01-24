import pandas as pd
import numpy as np

def best_feature_engineering(input_csv: str, output_csv: str):
    """
    Loads the dataset, applies data cleaning & feature engineering,
    and saves a processed CSV for modeling.

    Steps:
      1. Handle 'Churn' & drop or transform post-churn columns
      2. Fill missing values in numeric/categorical columns
      3. Drop or transform columns that are duplicates or less useful
      4. Create new derived features (if desired)
      5. Perform final encoding (as needed)
      6. Save final cleaned dataset
    """

    # 1. Load Data
    df = pd.read_csv(input_csv)

    # -----------------------------------------------------------------------------
    # 2. Convert 'Churn' to 0/1
    #    - If you see "Churned" or "Stayed" in 'Churn', map "Churned" => 1, "Stayed" => 0.
    # -----------------------------------------------------------------------------
    if "Churn" in df.columns:
        unique_churn_vals = df["Churn"].dropna().unique().tolist()
        # Example scenario: if it includes 'Churned' or 'Stayed'
        if "Churned" in unique_churn_vals or "Stayed" in unique_churn_vals:
            df["Churn"] = df["Churn"].map({"Churned": 1, "Stayed": 0})
        # If it's already True/False or 1/0, you could handle that here
        # e.g., df["Churn"] = df["Churn"].astype(int) for True/False
    else:
        raise ValueError("No 'Churn' column found in dataset. Cannot proceed.")

    # -----------------------------------------------------------------------------
    # 3. Drop or handle columns that might be post-churn info or duplicates
    # -----------------------------------------------------------------------------
    # 'Churn Category' and 'Churn Reason' typically describe why the customer
    # left AFTER they've already churned—often not predictive.
    drop_cols = []
    if "Churn Category" in df.columns:
        drop_cols.append("Churn Category")
    if "Churn Reason" in df.columns:
        drop_cols.append("Churn Reason")

    # 'Churn Score' might also be a derived metric assigned after churn is observed;
    # you can consider dropping it if it's post-churn. If you believe it's a
    # marketing metric assigned before churn, you could keep it.
    if "Churn Score" in df.columns:
        drop_cols.append("Churn Score")

    # 'Customer Status' often duplicates info in 'Churn' ("Stayed"/"Churned").
    if "Customer Status" in df.columns:
        drop_cols.append("Customer Status")

    # 'Lat Long' is a combined string, and we already have 'Latitude' / 'Longitude'.
    if "Lat Long" in df.columns:
        drop_cols.append("Lat Long")

    # If you prefer not to use 'Customer ID' in modeling, drop it:
    if "Customer ID" in df.columns:
        drop_cols.append("Customer ID")

    # If you don’t need geographic info at a fine level:
    #   - Possibly drop 'Zip Code', or
    #   - Convert (Latitude, Longitude) to region-based features (optional)
    # For demonstration, let's drop 'Zip Code' to reduce columns:
    if "Zip Code" in df.columns:
        drop_cols.append("Zip Code")

    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # -----------------------------------------------------------------------------
    # 4. Handle Missing Values
    #    - Fill numeric columns with mean or median
    #    - Fill object/categorical columns with "Unknown" or drop them if too sparse
    # -----------------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols  = df.select_dtypes(include=[object]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    for col in object_cols:
        if df[col].isnull().any():
            df[col].fillna("Unknown", inplace=True)

    # -----------------------------------------------------------------------------
    # 5. Example Feature Engineering
    # -----------------------------------------------------------------------------
    # (A) Create a feature: "Revenue Per Month" = Total Revenue / Tenure in Months
    if "Total Revenue" in df.columns and "Tenure in Months" in df.columns:
        df["RevenuePerMonth"] = df.apply(
            lambda row: row["Total Revenue"] / row["Tenure in Months"] 
                        if row["Tenure in Months"] > 0 else 0,
            axis=1
        )

    # (B) Create a feature: "ExtraChargesRatio" 
    # = (Total Extra Data Charges + Total Long Distance Charges) / (Total Revenue)
    extra_cols = ["Total Extra Data Charges", "Total Long Distance Charges", "Total Revenue"]
    if all(col in df.columns for col in extra_cols):
        def extra_ratio(row):
            total_extra = row["Total Extra Data Charges"] + row["Total Long Distance Charges"]
            return total_extra / row["Total Revenue"] if row["Total Revenue"] > 0 else 0
        df["ExtraChargesRatio"] = df.apply(extra_ratio, axis=1)

    # (C) If "Gender_Male", "Partner_1", "Dependents_1", "Contract_One Year", "Contract_Two Year"
    #     are already one-hot/binary columns, no further encoding needed for them.
    #     But for columns like "Internet Type" or "Offer", we might do one-hot encoding.
    cat_cols_for_encoding = []
    # Example: 'Internet Type' might be DSL, Fiber Optic, Cable, Unknown
    if "Internet Type" in df.columns:
        cat_cols_for_encoding.append("Internet Type")
    # Example: 'Offer' might be Offer A, Offer B, or blank
    if "Offer" in df.columns:
        cat_cols_for_encoding.append("Offer")
    # Example: 'Payment Method' might be Bank Withdrawal, Credit Card, ...
    if "Payment Method" in df.columns:
        cat_cols_for_encoding.append("Payment Method")

    # Add any other columns you deem categorical
    cat_cols_for_encoding = [c for c in cat_cols_for_encoding if c in df.columns]

    # One-hot encode these columns
    if len(cat_cols_for_encoding) > 0:
        df = pd.get_dummies(df, columns=cat_cols_for_encoding, drop_first=True)

    # -----------------------------------------------------------------------------
    # 6. Save the Final, Cleaned & Feature-Engineered Dataset
    # -----------------------------------------------------------------------------
    df.to_csv(output_csv, index=False)
    print(f"Processed dataset saved to: {output_csv}")

# ------------------------------------------------------------------------------
# Example usage:
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    input_file = "/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/test_processed.csv"     # Adjust path
    output_file = "/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/test_cleaned.csv"

    best_feature_engineering(input_file, output_file)
