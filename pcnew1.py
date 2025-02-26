import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define panel parameters
reflectivity = 0.1  # 20% reflectivity

# Define time intervals with 30-minute slots
TIME_SLOTS = {
    "Morning": [(6, 6.5), (6.5, 7), (7, 7.5), (7.5, 8), (8, 8.5), (8.5, 9)],
    "Afternoon": [(12, 12.5), (12.5, 13), (13, 13.5), (13.5, 14), (14, 14.5), (14.5, 15)],
    "Evening": [(16, 16.5), (16.5, 17), (17, 17.5), (17.5, 18), (18, 18.5)]
}

# Function to model solar angles based on time
def get_solar_angle(hour):
    """Returns the solar incidence angle at a given hour."""
    if hour < 12:
        return max(10, 90 - 7.5 * (12 - hour))  # Morning increase
    else:
        return max(10, 90 - 7.5 * (hour - 12))  # Afternoon decrease

# Function to get the optimal tilt angle
def get_optimal_panel_angle(hour):
    """Returns the optimal panel tilt angle (relative to ground) for maximum absorption."""
    solar_incidence_angle = get_solar_angle(hour)
    optimal_tilt_angle = 90 - solar_incidence_angle  # Panel tilt = 90 - solar angle
    return max(0, optimal_tilt_angle)  # Ensure non-negative values

# Simulate maximum solar absorption over time slots
max_efficiency = (1 - reflectivity)  # Constant max efficiency
time_labels = []
optimal_tilt_angles = []
efficiencies = []

for period, slots in TIME_SLOTS.items():
    for start, end in slots:
        hour = (start + end) / 2
        optimal_tilt = get_optimal_panel_angle(hour)

        time_labels.append(f"{int(start)}:{int((start % 1) * 60)}-{int(end)}:{int((end % 1) * 60)}")
        optimal_tilt_angles.append(optimal_tilt)
        efficiencies.append(max_efficiency)  # Constant efficiency

# Visualization
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(time_labels, efficiencies, marker='o', color='b', label='Constant Max Absorption Efficiency')
ax2.plot(time_labels, optimal_tilt_angles, marker='s', color='r', linestyle='--', label='Optimal Panel Tilt')

ax1.set_xlabel("Time of Day")
ax1.set_ylabel("Absorption Efficiency (Constant)", color='b')
ax2.set_ylabel("Optimal Panel Tilt (Degrees)", color='r')

ax1.set_title("Constant Maximum Absorption Efficiency & Optimal Tilt over Time")
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.xticks(rotation=45)
plt.grid()
plt.show()

# Display results in table format
absorbance_table = pd.DataFrame({
    "Time Interval": time_labels,
    "Optimal Panel Tilt (°)": optimal_tilt_angles,
    "Max Absorption Efficiency": efficiencies
})
print(absorbance_table)

panel_width = 1.0  # meters
panel_height = 1.0  # meters
reflectivity = 0.2
installation_cost = 500  # USD
maintenance_cost_per_year = 50  # USD
lifetime_years = 25
energy_price_per_kWh = 0.12  # USD
storage_cost_per_kWh = 0.05  # USD
panel_power_capacity_kW = 1.5  # Maximum output in kW
sun_hours_per_day = 5  # Effective sunlight hours per day
daily_absorbed_energy = panel_power_capacity_kW * sun_hours_per_day * np.mean(efficiencies)
annual_absorbed_energy = daily_absorbed_energy * 365

# Revenue & Costs
annual_revenue = annual_absorbed_energy * energy_price_per_kWh
annual_storage_cost = annual_absorbed_energy * storage_cost_per_kWh
total_revenue_over_lifetime = annual_revenue * lifetime_years
total_storage_cost_over_lifetime = annual_storage_cost * lifetime_years
total_maintenance_cost = maintenance_cost_per_year * lifetime_years
total_cost = installation_cost + total_maintenance_cost + total_storage_cost_over_lifetime
total_profit = total_revenue_over_lifetime - total_cost

print("\n💰 **Economic Analysis** 💰")
print(f"🔹 Total Energy Absorbed Per Year: {annual_absorbed_energy:.2f} kWh")
print(f"🔹 Annual Revenue: ${annual_revenue:.2f}")
print(f"🔹 Annual Storage Cost: ${annual_storage_cost:.2f}")
print(f"🔹 Total Revenue over {lifetime_years} years: ${total_revenue_over_lifetime:.2f}")
print(f"🔹 Total Storage Cost over {lifetime_years} years: ${total_storage_cost_over_lifetime:.2f}")
print(f"🔹 Total Maintenance Cost over {lifetime_years} years: ${total_maintenance_cost:.2f}")
print(f"🔹 Total Installation Cost: ${installation_cost:.2f}")
print(f"💵 **Total Profit over {lifetime_years}: ${total_profit:.2f}**")
# User input for best panel angle
time_input = float(input("Enter the time of day (in hours, e.g., 10 for 10 AM): "))
best_angle = get_optimal_panel_angle(time_input)
print(f"For {time_input}:00, set your solar panel to {best_angle:.2f}° for maximum efficiency.")
