import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error , r2_score

df = pd.read_csv("data/house_data_v2.csv")

X = df.drop("price",axis = 1)
y = df["price"]

numeric_features = [
     "size_sqft", "bedrooms", "bathrooms",
    "property_age", "distance_to_metro_km"
]


categorical_features = [
    "location", "has_parking"
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


models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


best_model = None
best_mae = float("inf")



for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n{name}")
    print("MAE:", mae)
    print("R2:", r2)

    if mae < best_mae:
        best_mae = mae
        best_model = pipeline

print("best model is :")
print(best_model)

##joblib.dump(best_model,"models/house_price_pipeline_v3.pkl")



model = best_model.named_steps["model"]
preprocessor = best_model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()
coefficients = model.coef_

for name, coef in zip(feature_names, coefficients):
    print(name, coef)




