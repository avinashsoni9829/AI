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

print("\nConfidence summary:")
print(df["confidence"].describe())

print("\nSample usable rows:")
usable = df[
    df["rent"].notnull() &
    df["area"].notnull() &
    df["bhk"].notnull() &
    (df["confidence"] >= 0.6)
]

print(usable[["area", "bhk", "rent", "deposit", "furnishing", "property_type", "confidence"]].head(20))
print("\nUsable rows:", len(usable))