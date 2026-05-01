import pandas as pd
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field, model_validator
import joblib

rent_model = joblib.load("rent_intelligence/data/rent_model.pkl")

class RentPredictionInput(BaseModel):
    city: str
    area: str
    bhk: int = Field(..., ge=1, le=6)
    size_sqft: int = Field(..., ge=300, le=6000)
    bathrooms: int = Field(..., ge=1, le=8)
    property_age: int = Field(..., ge=0, le=100)
    distance_to_metro_km: float = Field(..., ge=0, le=30)
    has_parking: int = Field(..., ge=0, le=1)
    property_type: str
    furnishing: str

    @model_validator(mode="after")
    def validate_property_logic(self):
        min_size_by_bhk = {
            1: 350,
            2: 700,
            3: 1000,
            4: 1500,
            5: 2200,
            6: 3000
        }

        if self.size_sqft < min_size_by_bhk.get(self.bhk, 300):
            raise ValueError("size_sqft is too small for selected BHK")

        if self.bathrooms > self.bhk + 2:
            raise ValueError("bathrooms too high for selected BHK")

        return self
    


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



@app.get("/similar-areas")
def get_similar_areas(city: str, area: str, limit: int = 3):
    city_stats = stats_df[stats_df["city"].str.lower() == city.lower()]

    if city_stats.empty:
        raise HTTPException(status_code=404, detail="City not found")

    target = city_stats[city_stats["area"].str.lower() == area.lower()]

    if target.empty:
        raise HTTPException(status_code=404, detail="Area not found")

    target_row = target.iloc[0]

    comparisons = city_stats[
        city_stats["area"].str.lower() != area.lower()
    ].copy()

    comparisons["rent_diff"] = (
        comparisons["avg_rent"] - target_row["avg_rent"]
    ).abs()

    comparisons["rent_per_sqft_diff"] = (
        comparisons["avg_rent_per_sqft"] - target_row["avg_rent_per_sqft"]
    ).abs()

    comparisons["similarity_score"] = (
        comparisons["rent_diff"] +
        comparisons["rent_per_sqft_diff"] * 1000
    )

    comparisons = comparisons.sort_values("similarity_score").head(limit)

    return {
        "city": city,
        "area": area,
        "similar_areas": [
            {
                "area": row["area"],
                "avg_rent": round(row["avg_rent"], 2),
                "median_rent": round(row["median_rent"], 2),
                "avg_rent_per_sqft": round(row["avg_rent_per_sqft"], 2),
                "listing_count": int(row["listing_count"]),
                "rent_tier": row["rent_tier"],
                "similarity_score": round(row["similarity_score"], 2)
            }
            for _, row in comparisons.iterrows()
        ]
    }


@app.post("/predict-rent")
def predict_rent(data: RentPredictionInput):
    input_df = pd.DataFrame([{
        "city": data.city,
        "area": data.area,
        "bhk": data.bhk,
        "size_sqft": data.size_sqft,
        "bathrooms": data.bathrooms,
        "property_age": data.property_age,
        "distance_to_metro_km": data.distance_to_metro_km,
        "has_parking": data.has_parking,
        "property_type": data.property_type,
        "furnishing": data.furnishing
    }])

    input_df["age_bucket"] = pd.cut(
        input_df["property_age"],
        bins=[0, 5, 15, 50],
        labels=["new", "mid", "old"],
        include_lowest=True
    )

    input_df["distance_bucket"] = pd.cut(
        input_df["distance_to_metro_km"],
        bins=[0, 1, 3, 10],
        labels=["near", "medium", "far"],
        include_lowest=True
    )

    prediction = rent_model.predict(input_df)[0]

    return {
        "city": data.city,
        "area": data.area,
        "predicted_rent": round(float(prediction), 2)
    }