import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="City Rent Intelligence API")

LISTINGS_PATH = "rent_intelligence/data/rental_listings.csv"
STATS_PATH = "rent_intelligence/data/area_stats.csv"

listings_df = pd.read_csv(LISTINGS_PATH)
stats_df = pd.read_csv(STATS_PATH)

@app.get("/")
def home():
    return {
        "message": "City Rent Intelligence API is running"
    }

@app.get("/areas")
def get_areas(city: str):
    city_stats = stats_df[stats_df["city"].str.lower() == city.lower()]

    if city_stats.empty:
        raise HTTPException(status_code=404, detail="City not found")

    return {
        "city": city,
        "areas": sorted(city_stats["area"].unique().tolist())
    }

@app.get("/area-insights")
def get_area_insights(city: str, area: str):
    result = stats_df[
        (stats_df["city"].str.lower() == city.lower()) &
        (stats_df["area"].str.lower() == area.lower())
    ]

    if result.empty:
        raise HTTPException(status_code=404, detail="Area not found")

    row = result.iloc[0]

    return {
        "city": row["city"],
        "area": row["area"],
        "avg_rent": round(row["avg_rent"], 2),
        "median_rent": round(row["median_rent"], 2),
        "min_rent": round(row["min_rent"], 2),
        "max_rent": round(row["max_rent"], 2),
        "listing_count": int(row["listing_count"]),
        "avg_rent_per_sqft": round(row["avg_rent_per_sqft"], 2),
        "rent_tier": row["rent_tier"],
        "lat": round(row["lat"], 6),
        "lng": round(row["lng"], 6)
    }

@app.get("/map-data")
def get_map_data(city: str):
    city_stats = stats_df[stats_df["city"].str.lower() == city.lower()]

    if city_stats.empty:
        raise HTTPException(status_code=404, detail="City not found")

    data = []

    for _, row in city_stats.iterrows():
        data.append({
            "area": row["area"],
            "lat": round(row["lat"], 6),
            "lng": round(row["lng"], 6),
            "avg_rent": round(row["avg_rent"], 2),
            "median_rent": round(row["median_rent"], 2),
            "avg_rent_per_sqft": round(row["avg_rent_per_sqft"], 2),
            "listing_count": int(row["listing_count"]),
            "rent_tier": row["rent_tier"]
        })

    return {
        "city": city,
        "areas": data
    }