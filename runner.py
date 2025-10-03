import numpy as np
import find_peaks
import pandas as pd
from util import column_labels, center_axis, fit_and_integrate
import matplotlib.pyplot as plt
from scipy.optismize import curve_fit
import math_functions
import plotly.express as px
from scipy.integrate import quad
from typing import Callable

df1 = pd.read_excel("before biasing.xlsx", header = 0)
df2 = pd.read_excel("before biaising daniel.xlsx", header = 0)
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

    x1 = np.array(second_peaks_x[key])
    y1 = np.array(second_peaks_y[key])

    integral0 = fit_and_integrate(x0, y0, math_functions.sextic, 0.5, 1000)
    integral1 = fit_and_integrate(x1,y1, math_functions.sextic, 0.5, 1000)
    integral_ratio_list.append(integral1/integral0)




"""
Mapping the integral ratio list
"""
color = ['RdBu_r']
num_range = [1.185, 1.485]
nested_ratio_list = [integral_ratio_list[i:i + 40] for i in range(0, len(integral_ratio_list), 40)]
fig = px.imshow(nested_ratio_list, text_auto=False, origin='upper', color_continuous_scale=color[0], zmin= num_range[0], zmax= num_range[1],  title= 'EELS Before Biasing')
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()


