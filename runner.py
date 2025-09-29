import numpy as np
import find_peaks
import pandas as pd
from util import column_labels
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math_functions
import plotly.express as px

df1 = pd.read_excel("EELS analysis before biasing.xlsx", header = 0)
df2 = pd.read_excel("DANIEL BEFORE BIASING.xlsx", header = 0)
df = pd.concat([df1, df2], axis = 1)



dfx= np.asarray(df["eV"])
labels = column_labels()
dfy = df.iloc[:, 1:]
dfy = dfy.reindex(columns = labels)

peaker = find_peaks.findPeaks(x_df = dfx, y_df = dfy)
peaker.set_peaks()


first_peaks_x, second_peaks_x = peaker.get_x_window()
first_peaks_y , second_peaks_y = peaker.get_y_window()

integral_ratio_list = []

for key in dfy:
    x0 = np.array(first_peaks_x[key])
    y0 = np.array(first_peaks_y[key])

    # Shift x0 to have the peak at the origin
    x0 = x0 - x0.mean()

    # Numerically integrate the peak
    integral_peak1 = np.trapz(y = y0, x = x0)

    x1 = np.array(second_peaks_x[key])
    y1 = np.array(second_peaks_y[key])

    x1 = x1 - x1.mean()

    integral_peak2 = np.trapz(y=y1, x = x1)

    integral_ratio_list.append(integral_peak2/integral_peak1)

color = ['RdBu_r']
#num_range = [min(integral_ratio_list), max(integral_ratio_list)]
num_range = [1.32, 1.475]
nested_ratio_list = [integral_ratio_list[i:i + 40] for i in range(0, len(integral_ratio_list), 40)]
fig = px.imshow(nested_ratio_list, text_auto=False, origin='upper', color_continuous_scale=color[0], zmin= num_range[0], zmax= num_range[1],  title= 'EELS Before Biasing')
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()


