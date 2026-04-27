import pandas as pd 
import joblib


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error,r2_score


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

## load data 

df = pd.read_csv("data/house_data_v2.csv")

X = df.drop("price",axis=1)
y = df["price"]

## column split 

numeric_features = [
    "size_sqft",
    "bedrooms",
    "bathrooms",
    "property_age",
    "distance_to_metro_km"
]

categorical_features = [
    "location",
    "has_parking"
]

## Numeric pipeline 
numeric_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

# categorical pipleline 

categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])


#combine 

preprocessor = ColumnTransformer([
   ("num", numeric_pipeline, numeric_features),
   ("cat",categorical_pipeline,categorical_features)
])

#models 

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state= 42)
}

#split 

X_train,X_test , y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state= 42
)

best_model = None
best_score = float("inf")

# train > compare 
for name,model in models.items():
    pipeline  = Pipeline([
        ("preprocessor",preprocessor),
        ("model",model)
    ])
    pipeline.fit(X_train,y_train)
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test,preds)
    r2 = r2_score(y_test,preds)

    print(f"\n{name}")
    print("MAE:", mae)
    print("R2:", r2)

    if mae < best_score:
        best_score = mae
        best_model = pipeline


joblib.dump(best_model,"models/house_price_pipeline.pkl")
