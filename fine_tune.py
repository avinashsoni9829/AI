import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("data/house_data_big.csv")


# features
df["age_bucket"] = pd.cut(df["property_age"], bins=[0,5,15,50], labels=["new","mid","old"])
df["distance_bucket"] = pd.cut(df["distance_to_metro_km"], bins=[0,1,3,10], labels=["near","medium","far"])


X = df.drop("price", axis=1)
y = df["price"]

numeric_features = [
    "size_sqft", "bedrooms", "bathrooms",
    "property_age", "distance_to_metro_km"
]

categorical_features = [
    "location", "has_parking", "age_bucket", "distance_bucket"
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



pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(random_state=42))
])


param_dist = {
    "model__n_estimators": [200, 300, 400, 500, 700],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__subsample": [0.6, 0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0]
}


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,   # try 20 random combinations
    scoring="neg_mean_absolute_error",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)



search.fit(X_train, y_train)

print("\nBest Params:")
print(search.best_params_)


best_model = search.best_estimator_

preds = best_model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nFinal Performance:")
print("MAE:", mae)
print("R2:", r2)

joblib.dump(best_model, "models/xgb_tuned.pkl")
print("\nTuned model saved ✅")


