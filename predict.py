import pandas as pd
import joblib

# Load model
model = joblib.load("models/model.pkl")

# New input
new_house = pd.DataFrame(
    [[1400, 3, 4]],
    columns=["size_sqft", "bedrooms", "location_score"]
)

pred = model.predict(new_house)

print("Predicted Price:", pred[0], "lakhs")