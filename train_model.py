import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Create synthetic dataset
# Features: Soil_Type, Temperature, Humidity, Rainfall_Categorical
# Target: Crop

# Soil Types: 
# 0: Black, 1: Red, 2: Alluvial, 3: Loamy, 4: Sandy

# Rainfall Categories (encoded as integers for simplicity in model, or kept as categories):
# 0: Low (<50mm), 1: Moderate (50-150mm), 2: Heavy (>150mm)

data = {
    'Soil_Type': [],
    'Temperature': [],
    'Humidity': [],
    'Rainfall': [],
    'Crop': []
}

np.random.seed(42)

def add_samples(soil, temp_mean, hum_mean, rain, crop, count=100):
    for _ in range(count):
        data['Soil_Type'].append(soil)
        data['Temperature'].append(np.random.normal(temp_mean, 3.0))
        data['Humidity'].append(np.random.normal(hum_mean, 5.0))
        data['Rainfall'].append(rain)
        data['Crop'].append(crop)

# Generate data based on typical Indian farming conditions
# Cotton: Black soil, High temp, Mod rain
add_samples('Black', 30.0, 60.0, 'Moderate', 'Cotton', 150)
# Soybean: Black/Loamy soil, Mod temp, Mod rain
add_samples('Black', 25.0, 70.0, 'Moderate', 'Soybean', 150)
add_samples('Loamy', 25.0, 70.0, 'Moderate', 'Soybean', 100)
# Rice: Alluvial/Red soil, High temp, High humidity, Heavy rain
add_samples('Alluvial', 28.0, 85.0, 'Heavy', 'Rice', 200)
add_samples('Red', 28.0, 85.0, 'Heavy', 'Rice', 100)
# Wheat: Alluvial/Loamy soil, Low temp, Mod humidity, Low/Mod rain
add_samples('Alluvial', 18.0, 50.0, 'Low', 'Wheat', 150)
add_samples('Loamy', 20.0, 55.0, 'Moderate', 'Wheat', 100)
# Groundnut: Red/Sandy soil, Mod/High temp, Mod rain
add_samples('Red', 28.0, 60.0, 'Moderate', 'Groundnut', 150)
add_samples('Sandy', 30.0, 50.0, 'Low', 'Groundnut', 100)
# Sugarcane: Alluvial/Black, High temp, High humidity, Heavy rain (irrigation)
add_samples('Alluvial', 32.0, 75.0, 'Heavy', 'Sugarcane', 150)
add_samples('Black', 30.0, 70.0, 'Heavy', 'Sugarcane', 100)
# Maize: Loamy/Red, Mod temp, Mod rain
add_samples('Loamy', 26.0, 65.0, 'Moderate', 'Maize', 150)
add_samples('Red', 27.0, 60.0, 'Moderate', 'Maize', 100)
# Bajra (Pearl Millet): Sandy, High Temp, Low Humidity, Low rain
add_samples('Sandy', 32.0, 40.0, 'Low', 'Bajra', 150)
add_samples('Red', 30.0, 45.0, 'Low', 'Bajra', 100)

df = pd.DataFrame(data)

# Encode categorical variables
soil_encoder = LabelEncoder()
rain_encoder = LabelEncoder()

df['Soil_Type_Encoded'] = soil_encoder.fit_transform(df['Soil_Type'])
df['Rainfall_Encoded'] = rain_encoder.fit_transform(df['Rainfall'])

X = df[['Soil_Type_Encoded', 'Temperature', 'Humidity', 'Rainfall_Encoded']]
y = df['Crop']

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('soil_encoder.pkl', 'wb') as f:
    pickle.dump(soil_encoder, f)

with open('rain_encoder.pkl', 'wb') as f:
    pickle.dump(rain_encoder, f)

print(f"Model trained with accuracy: {model.score(X, y)*100:.2f}%")
print("Saved crop_model.pkl, soil_encoder.pkl, rain_encoder.pkl in current directory.")
