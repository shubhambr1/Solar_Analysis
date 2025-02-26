import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def calculate_efficiency(angle, reflectivity, intensity):
    return max(0, np.cos(np.radians(angle))) * (1 - reflectivity) * (intensity / 10000)

def train_ml_model():
    np.random.seed(42)
    train_angles = np.random.uniform(0, 90, 5000)
    train_reflectivity = np.random.uniform(0.1, 0.5, 5000)
    train_intensity = np.random.uniform(8000, 12000, 5000)
    
    train_efficiency = np.array([calculate_efficiency(a, r, i) for a, r, i in zip(train_angles, train_reflectivity, train_intensity)])
    X_train = np.column_stack((train_angles, train_reflectivity, train_intensity))
    y_train = train_efficiency
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model
