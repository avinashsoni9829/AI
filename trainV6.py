# ✅ feature engineering
# ✅ train/validation/test split
# ✅ monotonic constraints
# ✅ early stopping
# ✅ final evaluation
# ✅ feature importance
# ✅ model bundle save


import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor


df = pd.read_csv("data/house_data_big.csv")


# ✅ feature engineering


df["age_bucket"] = pd.cut(
    df["property_age"],
    bins=[0, 5, 15, 50],
    labels=["new", "mid", "old"],
    include_lowest=True
)

df["distance_bucket"] = pd.cut(
    df["distance_to_metro_km"],
    bins=[0, 1, 3, 20],
    labels=["near", "medium", "far"],
    include_lowest=True
)




X = df.drop("price", axis=1)
y = df["price"]


numeric_features = [
    "size_sqft",
    "bedrooms",
    "bathrooms",
    "property_age",
    "distance_to_metro_km"
]

categorical_features = [
    "location",
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





# ✅ train/validation/test split


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)


X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

feature_names = list(preprocessor.get_feature_names_out())



# -----------------------------
# 7. Monotonic constraints
# -----------------------------
# Numeric feature order:
# size_sqft              -> +1
# bedrooms               -> +1
# bathrooms              -> +1
# property_age           -> -1
# distance_to_metro_km   -> -1
#
# Categorical one-hot features -> 0


numeric_constraints = [1, 1, 1, -1, -1]
categorical_count = len(feature_names) - len(numeric_constraints)
categorical_constraints = [0] * categorical_count

monotone_constraints = tuple(numeric_constraints + categorical_constraints)

print("Feature names:")
for name, constraint in zip(feature_names, monotone_constraints):
    print(name, "constraint:", constraint)


model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="mae",
    monotone_constraints=monotone_constraints,
    early_stopping_rounds=30,
    random_state=42
)

model.fit(
    X_train_processed,
    y_train,
    eval_set=[(X_val_processed, y_val)],
    verbose=False
)


preds = model.predict(X_test_processed)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nFinal Test Performance")
print("MAE:", mae)
print("R2:", r2)
print("Best iteration:", model.best_iteration)


print("\nFeature Importances:")
for name, importance in zip(feature_names, model.feature_importances_):
    print(name, importance)


model_bundle = {
    "preprocessor": preprocessor,
    "model": model,
    "feature_names": feature_names,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

joblib.dump(model_bundle, "models/house_price_model_v6.pkl")

print("\nModel v6 saved ✅")



