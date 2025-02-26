def get_solar_angle(hour):
    return max(10, 90 - 7.5 * abs(12 - hour))

def get_optimal_panel_angle(hour):
    return max(0, 90 - get_solar_angle(hour))
