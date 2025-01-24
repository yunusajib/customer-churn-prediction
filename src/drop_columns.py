import pandas as pd

# Update the file path to where your CSV actually is
csv_path = "/Users/yunusajib/Desktop/customer-churn-prediction/Data/processed/test_cleaned.csv"
columns_to_drop = [
    "Churn Category",
    "Churn Reason",
    "Churn Score",
    "Customer Status",
    "Customer ID",
    "Lat Long",
    # ...any others you don't want
]

df = pd.read_csv(csv_path)
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Save the cleaned dataset
df.to_csv("data/processed/test_ready.csv", index=False)
print("Dropped columns and saved to data/processed/training_ready.csv")
