import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title = "House prediction api")
model = joblib.load("models/model.pkl")
pipeline = joblib.load("models/house_price_pipeline.pkl")

class HouseInput(BaseModel):
    size_sqft: int 
    bedrooms: int
    location_score: int


class HouseInputV2(BaseModel):
    size_sqft: int
    bedrooms: int
    bathrooms: int
    location: str
    property_age: int
    has_parking: int
    distance_to_metro_km: float

@app.get("/")
def home():
    return {"message" : "House price prediction"}

@app.post("/predict")
def predict_price(data: HouseInput):
    input_df = pd.DataFrame(
        [[data.size_sqft,data.bedrooms,data.location_score]],
        columns=["size_sqft","bedrooms","location_score"]
    )
    prediction = model.predict(input_df)[0]
    return{
        "size_sqft": data.size_sqft,
        "bedrooms": data.bedrooms,
        "location_score":data.location_score,
        "predicted_price_lakhs":round(prediction,2)
    }

@app.post("/predictv2")
def predict_price(data: HouseInputV2):
    input_df = pd.DataFrame([{
        "size_sqft": data.size_sqft,
        "bedrooms": data.bedrooms,
        "bathrooms": data.bathrooms,
        "location": data.location,
        "property_age": data.property_age,
        "has_parking": data.has_parking,
        "distance_to_metro_km": data.distance_to_metro_km
    }])

    prediction = pipeline.predict(input_df)[0]

    return {
        "predicted_price_lakhs": round(prediction, 2)
    }
