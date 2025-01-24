import pandas as pd
import numpy as np

def preprocess_data(
    train_path: str,
    val_path: str,
    test_path: str,
    out_train_path: str,
    out_val_path: str,
    out_test_path: str
):
    """
    Reads train, val, test data, performs cleaning and preprocessing, 
    ensures consistent encoding, and saves the processed splits.
    """

    # ------------------
    # 1. Load raw splits
    # ------------------
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Create a 'split' column to keep track of which rows belong to train/val/test
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'

    # Combine all into one DataFrame
    combined_df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

    # ----------------------------
    # 2. Example: Handle data types
    # ----------------------------
    # Convert 'TotalCharges' to numeric
    # if 'Total Charges' in combined_df.columns:
    #     combined_df['TotalCharges'] = pd.to_numeric(combined_df['TotalCharges'], errors='coerce')
    #     # Fill missing with mean or any other strategy
    #     combined_df['TotalCharges'].fillna(combined_df['TotalCharges'].mean(), inplace=True)

    # ---------------------
    # 3. Convert 'Churn' to 0/1 
    # ---------------------
    # If you have a 'Churn' column with "Yes"/"No", map them to 1/0
    if 'Churn' in combined_df.columns:
        combined_df['Churn'] = combined_df['Churn'].map({'Yes': 1, 'No': 0})

    # ------------------------------------------
    # 4. Drop columns you don't need (if any)
    # ------------------------------------------
    # Example: 'customerID' might be useless for modeling
    if 'customerID' in combined_df.columns:
        combined_df.drop('customerID', axis=1, inplace=True)

    # -----------------------------------------------------
    # 5. Identify Categorical Columns & Perform One-Hot Encoding
    # -----------------------------------------------------
    # Below is just an example list; adjust based on your dataset
    cat_cols = [
        'Gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
        'PaperlessBilling', 'PaymentMethod'
    ]

    # Only encode columns actually present in the DataFrame
    cat_cols = [col for col in cat_cols if col in combined_df.columns]

    # One-hot encode these columns. drop_first=True prevents dummy variable trap
    combined_df = pd.get_dummies(combined_df, columns=cat_cols, drop_first=True)

    # ------------------------------------
    # 6. Separate Combined Back Into Splits
    # ------------------------------------
    df_train_processed = combined_df[combined_df['split'] == 'train'].copy()
    df_val_processed   = combined_df[combined_df['split'] == 'val'].copy()
    df_test_processed  = combined_df[combined_df['split'] == 'test'].copy()

    # Drop the 'split' column since it's only for internal tracking
    df_train_processed.drop('split', axis=1, inplace=True)
    df_val_processed.drop('split', axis=1, inplace=True)
    df_test_processed.drop('split', axis=1, inplace=True)

    # ----------------------------------------
    # 7. Save the Processed Splits to CSV
    # ----------------------------------------
    df_train_processed.to_csv(out_train_path, index=False)
    df_val_processed.to_csv(out_val_path, index=False)
    df_test_processed.to_csv(out_test_path, index=False)

if __name__ == "__main__":
    # Example usage with placeholder file paths:
    preprocess_data(
        train_path="/Users/yunusajib/Desktop/customer-churn-prediction/Data/Raw/Telco_Churn/train.csv",
        val_path="/Users/yunusajib/Desktop/customer-churn-prediction/Data/Raw/Telco_Churn/validation.csv",
        test_path="/Users/yunusajib/Desktop/customer-churn-prediction/Data/Raw/Telco_Churn/test.csv",
        out_train_path="/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/train_processed.csv",
        out_val_path="/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/val_processed.csv",
        out_test_path="/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/test_processed.csv"
    )
