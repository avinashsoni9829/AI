import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title = "House prediction api")
model = joblib.load("models/model.pkl")

class HouseInput(BaseModel):
    size_sqft: int 
    bedrooms: int
    location_score: int 

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
