def economic_analysis(panel_capacity_kW=1.5, sun_hours_per_day=5, energy_price=0.12, lifetime=25, installation_cost=500, maintenance_cost=50):
    annual_energy = panel_capacity_kW * sun_hours_per_day * 365
    revenue = annual_energy * energy_price
    total_revenue = revenue * lifetime
    total_cost = installation_cost + (maintenance_cost * lifetime)
    profit = total_revenue - total_cost
    return total_revenue, total_cost, profit
