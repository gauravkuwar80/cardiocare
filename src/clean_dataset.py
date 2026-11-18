
import pandas as pd

# Load original Kaggle dataset
input_path = "data/raw/heart.csv"
output_path = "data/processed/heart_cleaned.csv"

df = pd.read_csv(input_path)

# Remove exact duplicate rows
df_cleaned = df.drop_duplicates().reset_index(drop=True)

# Save cleaned dataset
df_cleaned.to_csv(output_path, index=False)

print("Dataset cleaning completed!")
print("Original shape:", df.shape)
print("Cleaned shape:", df_cleaned.shape)
print("Duplicates removed:", len(df) - len(df_cleaned))
print("\nTarget distribution:")
print(df_cleaned["target"].value_counts())
