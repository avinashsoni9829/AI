import sqlite3
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

DB_PATH = "rent_intelligence/data/rent_posts.db"
MODEL_PATH = "rent_intelligence/data/real_rent_model.pkl"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM rental_posts", conn)
conn.close()

# Only usable parsed rows
df = df[
    df["rent"].notnull() &
    df["area"].notnull() &
    df["bhk"].notnull() &
    (df["confidence"] >= 0.6)
].copy()

# Type cleanup
df["rent"] = df["rent"].astype(float)
df["bhk"] = df["bhk"].astype(int)

# Remove unrealistic rent outliers for now
df = df[(df["rent"] >= 5000) & (df["rent"] <= 200000)]

# Basic fallback columns
df["city"] = df["city"].fillna("Bangalore")
df["deposit"] = df["deposit"].fillna(0)
df["furnishing"] = df["furnishing"].fillna("Unknown")
df["property_type"] = df["property_type"].fillna("Apartment")
df["gender_preference"] = df["gender_preference"].fillna("Any")

print("Training rows:", len(df))

if len(df) < 30:
    raise ValueError(
        "Not enough real data yet. Need at least 30 usable rows, ideally 200+."
    )

X = df[[
    "city",
    "area",
    "bhk",
    "deposit",
    "furnishing",
    "property_type",
    "gender_preference"
]]

y = df["rent"]

numeric_features = [
    "bhk",
    "deposit"
]

categorical_features = [
    "city",
    "area",
    "furnishing",
    "property_type",
    "gender_preference"
]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
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

print("\nReal Rent Model Performance")
print("MAE:", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))

joblib.dump(pipeline, MODEL_PATH)

print("\nReal rent model saved:", MODEL_PATH)