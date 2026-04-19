import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Synthetic Dataset for Linear Regression Yield Prediction
# Variables affecting yield: Farm Size (Acres), Rainfall (mm/year), Temperature (C), Crop Type
# Target: Yield (Tonnes)

np.random.seed(42)

crops = ['Wheat', 'Rice', 'Cotton', 'Sugarcane', 'Soybean']
data = {'Crop': [], 'Farm_Size_Acres': [], 'Rainfall_mm': [], 'Temperature': [], 'Yield_Tonnes': []}

# Generate 500 samples
for _ in range(500):
    crop = np.random.choice(crops)
    farm_size = np.random.uniform(1.0, 50.0) # 1 to 50 acres
    rainfall = np.random.uniform(300, 2000) # 300mm to 2000mm
    temp = np.random.uniform(20.0, 40.0) # 20C to 40C
    
    # Base yield per acre varies by crop roughly
    # Wheat: ~1.5 tonnes/acre, Rice: ~2 tonnes/acre, Cotton: ~0.5 tonnes/acre
    # Sugarcane: ~30 tonnes/acre, Soybean: ~1.2 tonnes/acre
    base_yield = {
        'Wheat': 1.5,
        'Rice': 2.0,
        'Cotton': 0.5,
        'Sugarcane': 30.0,
        'Soybean': 1.2
    }[crop]
    
    # Add some correlation with rainfall and temp (just simple synthetic logic)
    # E.g., Rice likes high rain, Wheat likes moderate temp
    weather_multiplier = 1.0
    if crop == 'Rice' and rainfall > 1000:
        weather_multiplier += 0.2
    if crop == 'Wheat' and temp < 30:
        weather_multiplier += 0.2
        
    # Introduce some random noise to make the regression realistic
    noise = np.random.normal(0, base_yield * 0.1)
    
    total_yield = (base_yield * farm_size * weather_multiplier) + (noise * farm_size)
    total_yield = max(0.1, total_yield) # Ensure positive yield
    
    data['Crop'].append(crop)
    data['Farm_Size_Acres'].append(farm_size)
    data['Rainfall_mm'].append(rainfall)
    data['Temperature'].append(temp)
    data['Yield_Tonnes'].append(total_yield)

df = pd.DataFrame(data)

# Encode Categorical Variables
crop_encoder = LabelEncoder()
df['Crop_Encoded'] = crop_encoder.fit_transform(df['Crop'])

X = df[['Crop_Encoded', 'Farm_Size_Acres', 'Rainfall_mm', 'Temperature']]
y = df['Yield_Tonnes']

# Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Save Models
with open('yield_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('yield_encoder.pkl', 'wb') as f:
    pickle.dump(crop_encoder, f)

print(f"Linear Regression R^2 Score: {model.score(X, y):.4f}")
print("Saved yield_model.pkl and yield_encoder.pkl")
