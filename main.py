import pandas as pd

## ML model jo relation find karta hai input aur output ke beech
from sklearn.linear_model import LinearRegression

## Data ko 2 parts mein todta hai: Training data, Test data
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, r2_score

## mae  - average mistake
## r2 score : model pattern recognition
## close to 1 = good, 0 = weak, < 0 -> bad

data = {
    "size_sqft": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000],
    "bedrooms": [1, 2, 2, 3, 3, 4, 4, 4, 5, 5],
    "location_score": [3, 4, 5, 4, 5, 6, 6, 7, 8, 9],
    "price": [40, 50, 65, 80, 100, 120, 135, 155, 180, 210]
}

df = pd.DataFrame(data)

## AI models mostly table format data pe kaam karte hain
## data dictionary ko table/DataFrame mein convert kiya

X = df[["size_sqft", "bedrooms", "location_score"]]  ## features/input
y = df["price"]  ## label/output

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

## Model internally equation bana raha hai:
## price = a*(size) + b*(bedrooms) + c*(location) + d

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Actual Prices:", list(y_test))
print("Predicted Prices:", list(y_pred))
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

## Important:
## new_house ko DataFrame bana rahe hain taaki warning na aaye
## kyunki model feature names ke saath train hua tha

new_house = pd.DataFrame(
    [[1400, 3, 4]],
    columns=["size_sqft", "bedrooms", "location_score"]
)

pred = model.predict(new_house)

print("Predicted Price (lakhs):", pred[0])

## Feature Importance : some features are strong and some are weak
## Important rule in ML : Data points >> Features

"""
1️⃣ Underfitting

Kam features
Model simple
Prediction weak

2️⃣ Overfitting

Zyada features
Model complex
Training strong
Real world weak

ML rule:

If:

Features = 100
Data points = 5

Model memorization karega, learning nahi.

Learning tab hoti hai jab model pattern samjhe,
sirf data yaad na kare.

"""