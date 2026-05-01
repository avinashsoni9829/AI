import pandas as pd
import numpy as np

np.random.seed(42)

rows = 3000
data = []

locations = {
    "Bangalore": 1.0,
    "Delhi": 1.15,
    "Mumbai": 1.45
}

property_types = ["Apartment", "Builder Floor", "Villa"]

size_ranges = {
    1: (450, 850),
    2: (850, 1400),
    3: (1200, 2200),
    4: (1800, 3200),
    5: (2500, 4500)
}

for _ in range(rows):
    bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.22, 0.35, 0.25, 0.13, 0.05])

    min_size, max_size = size_ranges[bedrooms]
    size = np.random.randint(min_size, max_size)

    bathrooms = max(1, bedrooms + np.random.choice([-1, 0, 1], p=[0.15, 0.65, 0.20]))

    location = np.random.choice(list(locations.keys()))
    location_factor = locations[location]

    age = np.random.randint(0, 25)
    parking = np.random.choice([0, 1], p=[0.35, 0.65])
    distance = np.round(np.random.uniform(0.2, 12), 2)

    property_type = np.random.choice(property_types, p=[0.72, 0.20, 0.08])

    # base price per sqft by city
    base_rate = 0.045 * location_factor

    # BHK premium
    bhk_premium = {
        1: 0,
        2: 12,
        3: 28,
        4: 50,
        5: 80
    }[bedrooms]

    # property type premium
    type_premium = {
        "Apartment": 0,
        "Builder Floor": -5,
        "Villa": 45
    }[property_type]

    price = size * base_rate

    price += bhk_premium
    price += type_premium

    # newer property premium
    price -= age * 0.7

    # metro distance penalty
    price -= distance * 1.8

    # parking premium
    if parking == 1:
        price += 7

    # realistic noise
    noise = np.random.normal(0, 6)
    price += noise

    price = max(price, 15)

    data.append([
        size,
        bedrooms,
        bathrooms,
        location,
        property_type,
        age,
        parking,
        distance,
        round(price, 2)
    ])

df = pd.DataFrame(data, columns=[
    "size_sqft",
    "bedrooms",
    "bathrooms",
    "location",
    "property_type",
    "property_age",
    "has_parking",
    "distance_to_metro_km",
    "price"
])

df.to_csv("data/house_data_big.csv", index=False)

print("Better dataset generated ✅")
print(df.head())