import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000

data = []

locations = ["Bangalore", "Delhi", "Mumbai"]

for _ in range(rows):
    size = np.random.randint(500, 3000)
    bedrooms = np.random.randint(1, 6)
    bathrooms = np.random.randint(1, 5)
    location = np.random.choice(locations)
    age = np.random.randint(0, 20)
    parking = np.random.choice([0, 1])
    distance = np.round(np.random.uniform(0.2, 10), 2)

    # 🧠 price logic (important)
    base_price = size * 0.05

    # location impact
    if location == "Mumbai":
        base_price *= 1.5
    elif location == "Delhi":
        base_price *= 1.2

    # age effect
    base_price -= age * 0.5

    # distance effect
    base_price -= distance * 1.5

    # parking bonus
    if parking == 1:
        base_price += 5

    # bedrooms bonus
    base_price += bedrooms * 3

    # noise (realism)
    noise = np.random.normal(0, 5)

    final_price = base_price + noise

    data.append([
        size, bedrooms, bathrooms,
        location, age, parking,
        distance, round(final_price, 2)
    ])

df = pd.DataFrame(data, columns=[
    "size_sqft", "bedrooms", "bathrooms",
    "location", "property_age",
    "has_parking", "distance_to_metro_km",
    "price"
])

df.to_csv("data/house_data_big.csv", index=False)

print("Dataset generated ✅")