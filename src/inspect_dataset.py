
import pandas as pd
from pathlib import Path

# Load raw dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET INFORMATION ==========")
print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["num"].value_counts().sort_index())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())
