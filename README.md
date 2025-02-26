Solar Panel Efficiency & Economic Analysis

 Overview
This project consists of multiple Python packages designed to simulate solar panel efficiency using machine learning, optimize panel angles for maximum energy absorption, visualize efficiency trends, and perform economic analysis of solar panel investments.

 Features
Machine Learning Model**: Uses Random Forest to predict solar panel efficiency based on angle, reflectivity, and sunlight intensity.
Solar Simulation**: Simulates solar energy absorption for a given panel setup.
Angle Optimization**: Calculates optimal tilt angles for panels at different times of the day.
Visualization**: Plots efficiency trends and optimal angles.
Economic Analysis**: Estimates revenue, costs, and profit over the panel's lifetime.

Package Breakdown

 1. `solar_ml_model.py`
- Trains a Random Forest model to predict panel efficiency based on angle, reflectivity, and intensity.
- Function: `train_ml_model()`

 2. `solar_simulation.py`
- Uses the trained ML model to simulate energy absorption.
- Function: `simulate_solar_panel(panel_width, reflectivity, num_rays, sun_intensity)`

 3. `solar_angles.py`
- Provides functions to calculate the solar angle and optimal panel tilt.
- Functions: `get_solar_angle(hour)`, `get_optimal_panel_angle(hour)`

 4. `solar_visualization.py`
- Generates plots to visualize efficiency trends and tilt optimization.
- Function: `plot_tilt_and_efficiency()`

 5. `solar_economics.py`
- Performs financial calculations for solar panel investment.
- Function: `economic_analysis(panel_capacity_kW, sun_hours_per_day, energy_price, lifetime, installation_cost, maintenance_cost)`

 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shubhambr1/solar-analysis.git
   cd solar-analysis
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

 Usage
1. Train the ML model:
   ```python
   from solar_ml_model import train_ml_model
   model = train_ml_model()
   ```
2. Simulate solar panel absorption:
   ```python
   from solar_simulation import simulate_solar_panel
   energy = simulate_solar_panel()
   print(f"Total Absorbed Energy: {energy}")
   ```
3. Get optimal panel angle:
   ```python
   from solar_angles import get_optimal_panel_angle
   angle = get_optimal_panel_angle(10)
   print(f"Optimal Panel Angle at 10 AM: {angle}")
   ```
4. Visualize efficiency trends:
   ```python
   from solar_visualization import plot_tilt_and_efficiency
   plot_tilt_and_efficiency()
   ```
5. Perform economic analysis:
   ```python
   from solar_economics import economic_analysis
   revenue, cost, profit = economic_analysis()
   print(f"Total Revenue: {revenue}, Total Cost: {cost}, Profit: {profit}")
   ```

 License
This project is open-source and available under the MIT License.

 Author
[Shubham BR](https://github.com/shubhambr1)

