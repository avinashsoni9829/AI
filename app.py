import pandas as pd
import joblib
import shap

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

app = FastAPI(title="House Price Prediction API V3")

pipeline = joblib.load("models/house_price_pipeline_v5.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]
feature_names = list(preprocessor.get_feature_names_out())

explainer = shap.Explainer(model)

class HouseInput(BaseModel):
    size_sqft: int = Field(..., ge=200, le=10000)
    bedrooms: int = Field(..., ge=1, le=10)
    bathrooms: int = Field(..., ge=1, le=10)
    location: str
    property_age: int = Field(..., ge=0, le=100)
    has_parking: int = Field(..., ge=0, le=1)
    distance_to_metro_km: float = Field(..., ge=0, le=20)

    @model_validator(mode="after")
    def validate_house_logic(self):
        if self.size_sqft < self.bedrooms * 250:
            raise ValueError("size_sqft is too small for number of bedrooms")
        return self

@app.get("/")
def home():
    return {"message": "House Price Prediction API with Explainability"}

@app.post("/predict")
def predict_price(data: HouseInput):
    raw_df = pd.DataFrame([{
        "size_sqft": data.size_sqft,
        "bedrooms": data.bedrooms,
        "bathrooms": data.bathrooms,
        "location": data.location,
        "property_age": data.property_age,
        "has_parking": data.has_parking,
        "distance_to_metro_km": data.distance_to_metro_km
    }])

    raw_df["size_per_bedroom"] = raw_df["size_sqft"] / raw_df["bedrooms"]

    raw_df["age_bucket"] = pd.cut(
        raw_df["property_age"],
        bins=[0, 5, 15, 50],
        labels=["new", "mid", "old"]
    )

    raw_df["distance_bucket"] = pd.cut(
        raw_df["distance_to_metro_km"],
        bins=[0, 1, 3, 10],
        labels=["near", "medium", "far"]
    )

    prediction = pipeline.predict(raw_df)[0]

    processed = preprocessor.transform(raw_df)
    shap_values = explainer(processed)

    impacts = shap_values.values[0]

    reasons = []
    for feature, impact in zip(feature_names, impacts):
        reasons.append({
            "feature": feature,
            "impact": round(float(impact), 2)
        })

    reasons = sorted(
        reasons,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )[:5]

    return {
        "predicted_price_lakhs": round(float(prediction), 2),
        "top_reasons": reasons
    }