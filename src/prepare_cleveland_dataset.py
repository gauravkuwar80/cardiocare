import pandas as pd

# Load original dataset
df = pd.read_csv("data/raw/heart_disease.csv")

# Convert original target into binary classification
df["target"] = (df["num"] > 0).astype(int)

# Remove original multiclass target column
df = df.drop(columns=["num"])

# Save as a separate processed dataset
output_path = "data/processed/heart_cleveland_cleaned.csv"
df.to_csv(output_path, index=False)

print("Cleveland dataset prepared successfully!")
print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())