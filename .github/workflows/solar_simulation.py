import numpy as np
from solar_ml_model import train_ml_model

def simulate_solar_panel(panel_width=1.0, reflectivity=0.2, num_rays=10000, sun_intensity=10000):
    rf_model = train_ml_model()
    angles = np.random.uniform(0, 90, num_rays)
    x_positions = np.random.uniform(0, panel_width, num_rays)
    X_test_real = np.column_stack((angles, np.full(num_rays, reflectivity), np.full(num_rays, sun_intensity)))
    predicted_efficiency = rf_model.predict(X_test_real)
    return np.mean(predicted_efficiency * sun_intensity)
