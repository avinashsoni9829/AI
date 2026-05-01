import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor

df = pd.read_csv("rent_intelligence/data/rental_listings.csv")

df["rent_per_sqft"] = df["rent"] / df["size_sqft"]

df["age_bucket"] = pd.cut(
    df["property_age"],
    bins=[0, 5, 15, 50],
    labels=["new", "mid", "old"],
    include_lowest=True
)

df["distance_bucket"] = pd.cut(
    df["distance_to_metro_km"],
    bins=[0, 1, 3, 10],
    labels=["near", "medium", "far"],
    include_lowest=True
)

X = df.drop(["rent", "rent_per_sqft", "lat", "lng"], axis=1)
y = df["rent"]


numeric_features = [
    "bhk",
    "size_sqft",
    "bathrooms",
    "property_age",
    "distance_to_metro_km"
]

categorical_features = [
    "city",
    "area",
    "property_type",
    "furnishing",
    "has_parking",
    "age_bucket",
    "distance_bucket"
]



numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("Rent Model Performance")
print("MAE:", mae)
print("R2:", r2)

joblib.dump(pipeline, "rent_intelligence/data/rent_model.pkl")





