import pandas as pd

## ML model jo relation find karta hai input aur output ke beech
from sklearn.linear_model import LinearRegression

data = {
    "size_sqft": [800, 1000, 1200, 1500, 1800],
    "bedrooms": [1, 2, 2, 3, 3],
    "location_score": [3, 4, 5, 4, 5],
    "price": [40, 50, 65, 80, 100]
}


df = pd.DataFrame(data)
##  AI models tables mein kaam karte hain.
##  data is converted into tables 



X = df[["size_sqft", "bedrooms", "location_score"]] ## features 
y = df["price"]  ## label  

model = LinearRegression()

model.fit(X,y) 
## Model internally equation bana raha hai: price = a*(size) + b*(bedrooms) + c*(location) + d

new_house = [[1400, 3, 4]]
pred = model.predict(new_house)




print("Predicted Price (lakhs):", pred[0])


## Feature Importance :  some features are strong and some are weak
## important rule in ML : Data points >> Features

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

ML rule : 

If:

Features = 100

Data points = 5

Model memorization karega, learning nahi.

Learning tab hoti hai jab model pattern samjhe, sirf data yaad na kare.


"""