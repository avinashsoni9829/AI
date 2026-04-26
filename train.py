import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import joblib

# load the data 
df = pd.read_csv("data/data.csv")

X = df[["size_sqft","bedrooms","location_score"]]
y = df["price"]

# split 
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# train 
model = LinearRegression()
model.fit(X_train,y_train)

# evaluate 
y_pred = model.predict(X_test)

print("MAE : ", mean_absolute_error(y_test,y_pred))
print("R2 : ",r2_score(y_test,y_pred))

# save the model 
joblib.dump(model,"models/model.pkl")