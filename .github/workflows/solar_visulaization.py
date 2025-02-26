import matplotlib.pyplot as plt
import pandas as pd
from solar_angles import get_optimal_panel_angle

def plot_tilt_and_efficiency():
    time_slots = [6, 7, 8, 12, 13, 14, 16, 17, 18]
    optimal_tilt_angles = [get_optimal_panel_angle(hour) for hour in time_slots]
    efficiencies = [0.8] * len(time_slots)
    df = pd.DataFrame({"Time": time_slots, "Tilt Angle": optimal_tilt_angles, "Efficiency": efficiencies})
    df.plot(x="Time", y=["Tilt Angle", "Efficiency"], kind="line", marker="o")
    plt.show()
