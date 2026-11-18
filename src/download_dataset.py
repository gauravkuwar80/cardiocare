
from ucimlrepo import fetch_ucirepo
from pathlib import Path

# Fetch the official UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Combine input features and target
features = heart_disease.data.features
target = heart_disease.data.targets

dataset = features.copy()
dataset["num"] = target["num"]

# Define project root and output directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save dataset
output_path = OUTPUT_DIR / "heart_disease.csv"
dataset.to_csv(output_path, index=False)

print("Dataset downloaded successfully!")
print(f"Dataset shape: {dataset.shape}")
print(f"Saved at: {output_path}")
print("\nFirst 5 rows:")
print(dataset.head())
