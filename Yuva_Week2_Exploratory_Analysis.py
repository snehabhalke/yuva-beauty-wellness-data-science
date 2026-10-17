"""
YUVA Data Science with Python Analyst Internship
Week 2: Data Identification, Sourcing Strategy & Initial Analysis Planning

IMPORTANT:
- This script uses invented sample data for learning.
- It does NOT contain real customer records.
- Later, replace the sample data with a legally usable public CSV dataset.
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("week2_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Create a tiny sample dataset so the script works immediately.
# These rows are fictional examples for practice only.
data = {
    "product_name": [
        "Daily Sunscreen", "Gentle Cleanser", "Hydrating Moisturiser",
        "Herbal Shampoo", "Tinted Lip Balm", "Vitamin C Serum",
        "Daily Sunscreen", "Gentle Cleanser"
    ],
    "category": [
        "Skincare", "Skincare", "Skincare", "Haircare",
        "Makeup", "Skincare", "Skincare", "Skincare"
    ],
    "brand": [
        "Brand A", "Brand B", "Brand A", "Brand C",
        "Brand B", "Brand C", "Brand A", "Brand B"
    ],
    "price_inr": [499, 299, 599, 249, 199, 699, 499, 299],
    "rating": [4.5, 4.2, 4.6, 3.9, 4.1, 4.4, 4.5, None]
}

df = pd.DataFrame(data)

print("\n=== 1. First five rows ===")
print(df.head())

print("\n=== 2. Dataset size ===")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n=== 3. Column information ===")
print(df.info())

# 2. Clean the data.
df["category"] = df["category"].astype("string").str.strip().str.title()
df["brand"] = df["brand"].astype("string").str.strip()
df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Count missing values before deciding what to do with them.
print("\n=== 4. Missing values in each column ===")
print(df.isna().sum())

# Remove exact duplicate rows. Keep this step only when duplicates are truly unwanted.
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"\nExact duplicate rows removed: {before - after}")

# Flag suspicious values instead of silently changing them.
invalid_price_count = (df["price_inr"] < 0).sum()
invalid_rating_count = ((df["rating"] < 0) | (df["rating"] > 5)).sum()
print(f"Rows with negative prices: {invalid_price_count}")
print(f"Rows with ratings outside 0–5: {invalid_rating_count}")

# Keep only records with a valid, non-negative price for price summaries.
price_data = df[df["price_inr"].notna() & (df["price_inr"] >= 0)]

# 3. Initial exploratory analysis.
print("\n=== 5. Number of products by category ===")
print(df["category"].value_counts(dropna=False))

print("\n=== 6. Average price by category (INR) ===")
print(price_data.groupby("category")["price_inr"].mean().round(2))

print("\n=== 7. Average rating by category ===")
print(df.groupby("category")["rating"].mean().round(2))

print("\n=== 8. Overall numeric summary ===")
print(df[["price_inr", "rating"]].describe().round(2))

# 4. Save the cleaned sample data and category summaries.
cleaned_file = OUTPUT_DIR / "week2_cleaned_sample.csv"
df.to_csv(cleaned_file, index=False)

category_summary = (
    df.groupby("category", dropna=False)
      .agg(
          product_count=("product_name", "count"),
          average_price_inr=("price_inr", "mean"),
          average_rating=("rating", "mean")
      )
      .round(2)
      .reset_index()
)
summary_file = OUTPUT_DIR / "week2_category_summary.csv"
category_summary.to_csv(summary_file, index=False)

print("\n=== 9. Files saved ===")
print(f"Cleaned sample: {cleaned_file}")
print(f"Category summary: {summary_file}")

print("\nNEXT STEP:")
print("Download a public beauty product dataset only after checking its license.")
print("Then replace the sample-data section with:")
print('real_df = pd.read_csv("your_dataset.csv")')
print("Inspect its columns before adapting the cleaning and analysis steps.")
