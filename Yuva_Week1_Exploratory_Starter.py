"""
Yuva Data Science with Python Analyst Internship
Week 1: project setup and a small synthetic-data EDA demonstration.

IMPORTANT:
- This script uses invented example rows for learning only.
- It is not real customer data and must not be presented as research findings.
- Later, replace the sample data with a permitted, documented dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Small synthetic example so the script runs without downloading a dataset.
data = {
    "product_name": [
        "Gentle Cleanser", "Hydrating Cream", "Daily Sunscreen",
        "Hair Oil", "Lip Balm", "Face Serum"
    ],
    "category": [
        "Skincare", "Skincare", "Skincare",
        "Haircare", "Makeup", "Skincare"
    ],
    "price": [250, 450, 399, 220, 99, 599],
    "rating": [4.2, 4.5, 4.4, 4.0, 3.9, 4.6],
}

df = pd.DataFrame(data)

print("\n--- Example dataset (synthetic) ---")
print(df)

print("\n--- Dataset shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values by column ---")
print(df.isna().sum())

print("\n--- Summary statistics ---")
print(df[["price", "rating"]].describe())

print("\n--- Number of example products by category ---")
print(df["category"].value_counts())

# Simple chart of example product counts by category.
df["category"].value_counts().plot(kind="bar")
plt.title("Example Product Count by Category (Synthetic Data)")
plt.xlabel("Category")
plt.ylabel("Number of example products")
plt.tight_layout()
plt.show()

# Simple price-rating association; a tiny synthetic sample is not enough
# for a meaningful business conclusion.
print("\nReminder: these invented rows are for learning only.")
print("Do not draw real-world conclusions from this sample.")
