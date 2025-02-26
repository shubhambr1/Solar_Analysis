import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Function to take dynamic inputs
def get_input(prompt, default_value):
    user_input = input(f"{prompt} (default {default_value}): ")
    return default_value if user_input.strip() == '' else float(user_input)

# Constants (input from the user)
panel_width = get_input("Enter panel width in meters", 1.0)
panel_height = get_input("Enter panel height in meters", 1.0)
reflectivity = get_input("Enter panel reflectivity (0 to 1)", 0.2)
num_rays = 10000  # Number of sunlight rays
sun_intensity = 10000  # W/m² (Solar constant)

# Economic parameters
panel_lifetime = get_input("Enter panel lifetime in years", 25)
electricity_rate = get_input("Enter electricity rate in $/kWh", 0.12)
storage_cost = get_input("Enter storage cost in $/kWh", 0.05)

# Parallel processing
num_cores = 4

# Function to simulate efficiency based on angle (for training the ML model)
def calculate_efficiency(angle, reflectivity, intensity):
    return max(0, np.cos(np.radians(angle))) * (1 - reflectivity) * (intensity / sun_intensity)

# Generate synthetic training data
np.random.seed(42)
train_angles = np.random.uniform(0, 90, 5000)  # Random angles for training
train_reflectivity = np.random.uniform(0.1, 0.5, 5000)  # Vary reflectivity
train_intensity = np.random.uniform(8000, 12000, 5000)  # Vary sun intensity

# Compute actual efficiency
train_efficiency = np.array([
    calculate_efficiency(a, r, i) for a, r, i in zip(train_angles, train_reflectivity, train_intensity)
])

# Prepare training dataset
X_train = np.column_stack((train_angles, train_reflectivity, train_intensity))
y_train = train_efficiency

# Train an ML Model (Random Forest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Test model performance
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
y_pred = rf_model.predict(X_test_split)

# Evaluate model accuracy
mae = mean_absolute_error(y_test_split, y_pred)
r2 = r2_score(y_test_split, y_pred)
print(f"ML Model Performance: MAE = {mae:.4f}, R² = {r2:.4f}")

# Generate real simulation rays
angles = np.random.uniform(0, 90, num_rays)  # Random angles of incidence
x_positions = np.random.uniform(0, panel_width, num_rays)  # Random x positions

# Predict efficiency using ML model
X_test_real = np.column_stack((angles, np.full(num_rays, reflectivity), np.full(num_rays, sun_intensity)))
predicted_efficiency = rf_model.predict(X_test_real)

# Compute absorbed energy based on ML model predictions
absorbed_energy_per_ray = predicted_efficiency * sun_intensity

# Total energy calculations
total_absorbed_energy = np.mean(absorbed_energy_per_ray)
efficiency_percentage = total_absorbed_energy * 100 / sun_intensity

# Economic analysis
total_energy_per_year = total_absorbed_energy * 365
annual_electricity_revenue = total_energy_per_year * electricity_rate / 1000
annual_storage_cost = total_energy_per_year * storage_cost / 1000

# Visualization of ML predictions
plt.figure(figsize=(10, 6))
plt.scatter(x_positions, angles, c=absorbed_energy_per_ray, cmap='plasma', s=5)
plt.colorbar(label='ML Predicted Absorption Efficiency')
plt.title('Solar Panel Efficiency Prediction using Machine Learning')
plt.xlabel('Panel Width (m)')
plt.ylabel('Angle of Incidence (degrees)')
plt.grid(True)
plt.show()

# Results
print(f"Panel Efficiency (ML Prediction): {efficiency_percentage:.2f}%")
print(f"Annual Electricity Revenue: ${annual_electricity_revenue:.2f}")
print(f"Annual Storage Cost: ${annual_storage_cost:.2f}")

# Lifetime economic estimate
total_revenue_over_lifetime = annual_electricity_revenue * panel_lifetime
total_storage_cost_over_lifetime = annual_storage_cost * panel_lifetime

print(f"Total Revenue over {panel_lifetime} years: ${total_revenue_over_lifetime:.2f}")
print(f"Total Storage Cost over {panel_lifetime} years: ${total_storage_cost_over_lifetime:.2f}")
