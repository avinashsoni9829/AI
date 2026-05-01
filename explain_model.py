import pandas as pd
import shap
import joblib

df = pd.read_csv("data/house_data_big.csv")

df["size_per_bedroom"] = df["size_sqft"] / df["bedrooms"]

df["age_bucket"] = pd.cut(
    df["property_age"],
    bins=[0, 5, 15, 50],
    labels=["new", "mid", "old"]
)

df["distance_bucket"] = pd.cut(
    df["distance_to_metro_km"],
    bins=[0, 1, 3, 10],
    labels=["near", "medium", "far"]
)

X = df.drop("price", axis=1)

pipeline = joblib.load("models/house_price_pipeline_v5.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

X_processed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

explainer = shap.Explainer(model)
shap_values = explainer(X_processed)

shap_values.feature_names = list(feature_names)

# Single prediction explanation
shap.plots.waterfall(shap_values[0])

# Global model explanation
shap.plots.bar(shap_values)