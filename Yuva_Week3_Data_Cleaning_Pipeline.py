"""
Yuva Week 3 - Data Cleaning and Preprocessing Pipeline
IMPORTANT: The rows below are fictional practice data, not real customer research.
Run:
  py -m pip install pandas numpy
  py Yuva_Week3_Data_Cleaning_Pipeline.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("week3_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Fictional practice data. Replace with a properly licensed public dataset for real analysis.
raw_data = [
    {"product_id": "P001", "product_name": "Hydrating Face Cream", "category": " Skin Care ", "price": "₹499", "rating": "4.5"},
    {"product_id": "P002", "product_name": "Gentle Cleanser", "category": "skin-care", "price": "399", "rating": "4.2"},
    {"product_id": "P003", "product_name": "Herbal Shampoo", "category": "Hair Care", "price": "₹299", "rating": "4.0"},
    {"product_id": "P004", "product_name": "Vitamin C Serum", "category": "Skin Care", "price": "999", "rating": "5.2"},
    {"product_id": "P004", "product_name": "Vitamin C Serum", "category": "Skin Care", "price": "999", "rating": "5.2"},
    {"product_id": "P005", "product_name": "Body Lotion", "category": "Body care", "price": None, "rating": "4.1"},
    {"product_id": "P006", "product_name": "Daily Sunscreen", "category": "Sun Care", "price": "-100", "rating": None},
    {"product_id": "P007", "product_name": "Face Mask", "category": "skin care", "price": "1,299", "rating": "3.9"},
    {"product_id": "P008", "product_name": "Hair Oil", "category": "Hair Care ", "price": "249", "rating": "4.3"},
    {"product_id": "P009", "product_name": "", "category": "Body Care", "price": "199", "rating": "4.0"},
]
df_raw = pd.DataFrame(raw_data)
df_raw.to_csv(OUTPUT_DIR / "week3_raw_sample.csv", index=False)

print("RAW DATA PROFILE")
print("Shape:", df_raw.shape)
print("Missing values:\n", df_raw.isna().sum())
print("Exact duplicate rows:", int(df_raw.duplicated().sum()))

df = df_raw.copy()
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
for col in ["product_id", "product_name", "category"]:
    df[col] = df[col].astype("string").str.strip()

category_map = {
    "skin-care": "Skin Care", "skin care": "Skin Care",
    "hair care": "Hair Care", "body care": "Body Care", "sun care": "Sun Care"
}
df["category"] = df["category"].str.lower().map(category_map).fillna(df["category"])

df["price"] = (df["price"].astype("string")
               .str.replace("₹", "", regex=False)
               .str.replace(",", "", regex=False).str.strip())
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

before = len(df)
df = df.drop_duplicates()
duplicates_removed = before - len(df)

df["invalid_price"] = df["price"].lt(0)
df["invalid_rating"] = df["rating"].lt(0) | df["rating"].gt(5)
df["missing_product_name"] = df["product_name"].isna() | df["product_name"].eq("")
df.loc[df["invalid_price"], "price"] = np.nan
df.loc[df["invalid_rating"], "rating"] = np.nan

valid_prices = df["price"].dropna()
if len(valid_prices) >= 4:
    q1, q3 = valid_prices.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["price_outlier_flag"] = df["price"].lt(lower) | df["price"].gt(upper)
else:
    df["price_outlier_flag"] = False

quality = pd.DataFrame({
    "column": df.columns,
    "missing_values": [int(df[c].isna().sum()) for c in df.columns],
    "data_type": [str(df[c].dtype) for c in df.columns],
})
quality.to_csv(OUTPUT_DIR / "week3_quality_summary.csv", index=False)
df.to_csv(OUTPUT_DIR / "week3_cleaned_sample.csv", index=False)

log = [
    f"Initial row count: {len(df_raw)}",
    f"Exact duplicate rows removed: {duplicates_removed}",
    f"Rows after exact duplicate removal: {len(df)}",
    "Text was trimmed and category labels standardized.",
    "Price currency symbols and separators were removed before numeric conversion.",
    "Negative prices and ratings outside 0-5 were flagged and set to missing.",
    "Possible price outliers were flagged with the IQR method, not automatically deleted.",
    "Missing values were reported, not automatically filled.",
    "All rows are fictional practice data and do not represent real products or customers."
]
(OUTPUT_DIR / "week3_cleaning_log.txt").write_text("\n".join(log), encoding="utf-8")

print("\nCLEANING SUMMARY")
print("Exact duplicates removed:", duplicates_removed)
print("Cleaned shape:", df.shape)
print("Category counts:\n", df["category"].value_counts(dropna=False))
print("Output saved in:", OUTPUT_DIR.resolve())
print("Done. Review outputs before using a real dataset.")
