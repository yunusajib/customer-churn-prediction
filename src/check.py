import pandas as pd
train = pd.read_csv("/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/train_ready.csv")
val   = pd.read_csv("/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/val_ready.csv")
test  = pd.read_csv("/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/val_ready.csv")

train_val = pd.concat([train, val], ignore_index=True)

cat_cols = train_val.select_dtypes(exclude=[float, int]).columns.tolist()

for col in cat_cols:
    train_val_unique = set(train_val[col].dropna().unique())
    test_unique = set(test[col].dropna().unique())
    diff = test_unique - train_val_unique

    if diff:
        print(f"{col} has unknown categories in test: {diff}")
