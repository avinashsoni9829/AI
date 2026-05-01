import pandas as pd
import numpy as np

np.random.seed(42)

areas = {
    "Bangalore": {
        "Whitefield": {"lat": 12.9698, "lng": 77.7500, "tier": "mid"},
        "Indiranagar": {"lat": 12.9784, "lng": 77.6408, "tier": "premium"},
        "Koramangala": {"lat": 12.9352, "lng": 77.6245, "tier": "premium"},
        "HSR Layout": {"lat": 12.9116, "lng": 77.6389, "tier": "mid"},
        "BTM Layout": {"lat": 12.9166, "lng": 77.6101, "tier": "budget"},
        "Marathahalli": {"lat": 12.9569, "lng": 77.7011, "tier": "mid"},
    },
    "Mumbai": {
        "Andheri": {"lat": 19.1197, "lng": 72.8468, "tier": "premium"},
        "Powai": {"lat": 19.1176, "lng": 72.9060, "tier": "premium"},
        "Thane": {"lat": 19.2183, "lng": 72.9781, "tier": "mid"},
        "Borivali": {"lat": 19.2307, "lng": 72.8567, "tier": "mid"},
        "Dadar": {"lat": 19.0178, "lng": 72.8478, "tier": "premium"},
    },
    "Delhi": {
        "Saket": {"lat": 28.5245, "lng": 77.2066, "tier": "premium"},
        "Dwarka": {"lat": 28.5921, "lng": 77.0460, "tier": "mid"},
        "Rohini": {"lat": 28.7499, "lng": 77.0565, "tier": "budget"},
        "Karol Bagh": {"lat": 28.6514, "lng": 77.1907, "tier": "mid"},
        "Vasant Kunj": {"lat": 28.5200, "lng": 77.1587, "tier": "premium"},
    }
}

tier_multiplier = {
    "budget": 0.85,
    "mid": 1.0,
    "premium": 1.35
}

city_multiplier = {
    "Bangalore": 1.0,
    "Delhi": 1.1,
    "Mumbai": 1.45
}

property_types = ["Apartment", "Builder Floor", "Villa"]
furnishing_types = ["Unfurnished", "Semi Furnished", "Fully Furnished"]

size_ranges = {
    1: (450, 800),
    2: (800, 1300),
    3: (1200, 2000),
    4: (1800, 3000)
}

rows = []

for city, city_areas in areas.items():
    for area, meta in city_areas.items():
        for _ in range(250):
            bhk = np.random.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10])
            size_min, size_max = size_ranges[bhk]
            size_sqft = np.random.randint(size_min, size_max)

            bathrooms = max(1, bhk + np.random.choice([-1, 0, 1], p=[0.15, 0.65, 0.20]))
            property_age = np.random.randint(0, 25)
            distance_to_metro_km = round(np.random.uniform(0.2, 8), 2)
            has_parking = np.random.choice([0, 1], p=[0.35, 0.65])

            property_type = np.random.choice(property_types, p=[0.75, 0.20, 0.05])
            furnishing = np.random.choice(furnishing_types, p=[0.25, 0.50, 0.25])

            base_rate_per_sqft = 28
            rent = size_sqft * base_rate_per_sqft
            rent *= city_multiplier[city]
            rent *= tier_multiplier[meta["tier"]]

            rent += bhk * 2500
            rent -= property_age * 250
            rent -= distance_to_metro_km * 700

            if has_parking:
                rent += 2500

            if furnishing == "Semi Furnished":
                rent += 3000
            elif furnishing == "Fully Furnished":
                rent += 7000

            if property_type == "Villa":
                rent += 25000
            elif property_type == "Builder Floor":
                rent -= 2000

            noise = np.random.normal(0, 5000)
            rent += noise
            rent = max(8000, rent)

            rows.append({
                "city": city,
                "area": area,
                "lat": meta["lat"] + np.random.normal(0, 0.005),
                "lng": meta["lng"] + np.random.normal(0, 0.005),
                "bhk": bhk,
                "size_sqft": size_sqft,
                "bathrooms": bathrooms,
                "property_age": property_age,
                "distance_to_metro_km": distance_to_metro_km,
                "has_parking": has_parking,
                "property_type": property_type,
                "furnishing": furnishing,
                "rent": round(rent, 2)
            })

df = pd.DataFrame(rows)
df.to_csv("rent_intelligence/data/rental_listings.csv", index=False)

print("Rental dataset generated ✅")
print(df.head())
print("Rows:", len(df))