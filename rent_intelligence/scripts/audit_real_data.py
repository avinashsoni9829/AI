import sqlite3
import pandas as pd

DB_PATH = "rent_intelligence/data/rent_posts.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM rental_posts", conn)
conn.close()

print("Total rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatus counts:")
print(df["status"].value_counts(dropna=False))

if "listing_type" in df.columns:
    print("\nListing type counts:")
    print(df["listing_type"].value_counts(dropna=False))
else:
    print("\nListing type counts: (no listing_type column)")

print("\nConfidence summary:")
print(df["confidence"].describe())

print("\nSample usable rows:")
usable = df[
    df["rent"].notnull() &
    df["area"].notnull() &
    df["bhk"].notnull() &
    (df["confidence"] >= 0.6)
]

cols = ["area", "bhk", "rent", "deposit", "furnishing", "property_type", "confidence"]
if "listing_type" in usable.columns:
    cols.append("listing_type")
print(usable[cols].head(20))
print("\nUsable rows:", len(usable))

if "listing_type" in usable.columns:
    print("\nUsable rows by listing type:")
    print(usable["listing_type"].value_counts(dropna=False))