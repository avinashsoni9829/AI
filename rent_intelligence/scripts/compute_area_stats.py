import pandas as pd

df = pd.read_csv("rent_intelligence/data/rental_listings.csv")

df["rent_per_sqft"] = df["rent"] / df["size_sqft"]


# outlier management 

cleaned = []

for city, city_df in df.groupby("city"):
    q1 = city_df["rent"].quantile(0.25)
    q3 = city_df["rent"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    city_clean = city_df[
        (city_df["rent"] >= lower) &
        (city_df["rent"] <= upper)
    ]

    cleaned.append(city_clean)


df = pd.concat(cleaned)


area_stats = df.groupby(["city", "area"]).agg(
    avg_rent=("rent", "mean"),
    median_rent=("rent", "median"),
    min_rent=("rent", "min"),
    max_rent=("rent", "max"),
    listing_count=("rent", "count"),
    avg_rent_per_sqft=("rent_per_sqft", "mean"),
    lat=("lat", "mean"),
    lng=("lng", "mean")
).reset_index()

def classify_tier(avg_rent):
    if avg_rent >= 70000:
        return "premium"
    elif avg_rent >= 35000:
        return "mid"
    return "budget"

area_stats["rent_tier"] = area_stats["avg_rent"].apply(classify_tier)

area_stats.to_csv("rent_intelligence/data/area_stats.csv", index=False)

print("Area stats generated ✅")
print(area_stats.head())