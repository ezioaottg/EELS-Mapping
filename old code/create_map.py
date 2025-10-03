
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


df1 = pd.read_excel("before biasing.xlsx", header = 0)
df2 = pd.read_excel("before biaising daniel.xlsx", header = 0)

df = pd.concat([df1, df2], axis = 1)

x= np.asarray(df["eV"])



labels = [f"{letter}{num}"
          for letter in (chr(c) for c in range(ord('A'), ord('V') + 1))
          for num in range(1, 41)]


peaks_list = {}
df_without_ev = df.iloc[:, 1:]
df_without_ev = df_without_ev.reindex(columns = labels)


for column_name in df_without_ev:
    y = df_without_ev[column_name]
    height = 35
    peaks, properties = find_peaks(y, height=height, distance=20)
    for i in range(150):
        if (len(peaks) > 2):
            height += 0.6
        if(len(peaks) < 2):
            height -= 0.6

        peaks, properties = find_peaks(y, height=height, distance=20)
    

    if (len(peaks) > 2):
        height += 1
        print("TOO MANY PEAKS")
        print(column_name)

    if(len(peaks) < 2):
        height -= 1
        print("TOO FEW PEAKS")
        print(column_name)

    peaks_list[column_name] = peaks




def peak_one(col_num):
    col_key = df_without_ev.columns[col_num]

    refined_peak_1_data = []

    lower_peak = peaks_list[col_key][0]
    lower_range = lower_peak - 8
    upper_range = lower_peak + 8
    
    y = np.array(df_without_ev[col_key][lower_range:upper_range])
    x = np.array(df["eV"][lower_range:upper_range])


    return (x,y)

def peak_two(col_num):
    col_key = df_without_ev.columns[col_num]

    refined_peak_1_data = []


    lower_peak = peaks_list[col_key][1]
    lower_range = lower_peak - 8
    upper_range = lower_peak + 8
    
    y = np.array(df_without_ev[col_key][lower_range:upper_range])
    x = np.array(df["eV"][lower_range:upper_range])


    return (x,y)






def custom_cubic_function(x, a, b, c, d):
    # Define a custom cubic function
    return a * x**3 + b * x**2 + c * x + d


def integrate_area_around_peak(data, fitted_function, params):
    # Unpack x and y data
    x_data, y_data = data

    # Fit the function to get the peak location (x-value of max y)
    x_values = np.linspace(min(x_data), max(x_data), 1000)
    y_values = fitted_function(x_values, *params)
    peak_x = x_values[np.argmax(y_values)]

    # Integrate the fitted function within ±0.5 units around the peak x-value
    integration_range = 0.5
    integrate_func = lambda x: fitted_function(x, *params)
    integral_value, _ = quad(integrate_func, peak_x - integration_range, peak_x + integration_range)

    return integral_value, peak_x


def plot_data_and_fit_different(data, fitted_function, params, col_num, integral_value, peak_x):
    # Unpack x and y data
    x_data, y_data = data

    # Plot original data
    plt.scatter(x_data, y_data, label='Original Data')

    # Generate x values for plotting the fitted function
    x_values = np.linspace(min(x_data) - 1, max(x_data) + 1, 100)

    # Plot fitted function with adjusted parameters
    plt.plot(x_values, fitted_function(x_values, *params), color='red', linestyle='-', label='Fitted Function')

    # Highlight the region within ±0.5 units around the peak x-value
    integration_range = 0.5
    x_shaded = np.linspace(peak_x - integration_range, peak_x + integration_range, 100)
    y_shaded = fitted_function(x_shaded, *params)
    plt.fill_between(x_shaded, y_shaded, alpha=0.3, color='gray')

    # Annotate the plot with the integral value
    plt.text(peak_x - 0.45, max(y_shaded), f'Integral = {integral_value:.2f}',
             fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Original Data and Fitted Function (Column {col_num})')
    plt.legend()
    plt.grid(True)
    plt.show()


# Initialize lists to store integrated areas for each peak
integrated_areas_peak_one = []
integrated_areas_peak_two = []

ratio_list_areas = []

# Input how many datasets you have in total
for col_num in range(880):
    # Example usage:
    # Load data from peak_one for current column
    data1 = peak_one(col_num)

    if data1:  # Check if data1 is not empty
        # Unpack the data for plotting
        x_data1, y_data1 = data1

        # Fit the cubic function to the data
        popt1, _ = curve_fit(custom_cubic_function, x_data1, y_data1)

        # Compute the integrated area for peak_one and get the peak_x
        integral_value1, peak_x1 = integrate_area_around_peak(data1, custom_cubic_function, popt1)
        integrated_areas_peak_one.append(integral_value1)

        # Print the integrated area for peak_one
        # print(f'Integrated area for peak one (column {col_num}): {integral_value1:.3f}')

        # Plot the data and the fitted function with integral value. Can un-comment line below to see fitted function plotted
        # plot_data_and_fit_different(data1, custom_cubic_function, popt1, col_num, integral_value1, peak_x1)

    else:
        # Append None or any default value to integrated_areas_peak_one if data is empty
        integrated_areas_peak_one.append(None)

    # Example usage:
    # Load data from peak_two for current column
    data2 = peak_two(col_num)


    if data2:  # Check if data2 is not empty
        # Unpack the data for plotting
        x_data2, y_data2 = data2

        # Fit the cubic function to the data
        popt2, _ = curve_fit(custom_cubic_function, x_data2, y_data2)

        # Compute the integrated area for peak_two and get the peak_x
        integral_value2, peak_x2 = integrate_area_around_peak(data2, custom_cubic_function, popt2)

        # Check if either integral_value1 or integral_value2 is negative
        if integral_value1 < 0 or integral_value2 < 0:
            integral_value2 = 0

        integrated_areas_peak_two.append(integral_value2)

        # Print the integrated area for peak_two
        # print(f'Integrated area for peak two (column {col_num}): {integral_value2:.3f}')

        # Plot the data and the fitted function with integral value. Can un-comment line below to see fitted function plotted
        #plot_data_and_fit_different(data2, custom_cubic_function, popt2, col_num, integral_value2, peak_x2)

    else:
        # Append None or any default value to integrated_areas_peak_two if data is empty
        integrated_areas_peak_two.append(None)

    # Print the area ratio of peak 2 to peak 1 if both values are available
    if integrated_areas_peak_one[col_num] is not None and integrated_areas_peak_two[col_num] is not None:
        #print(f"Area ratio of peak 2 to peak 1 (column {col_num}): {round(integrated_areas_peak_two[col_num] / integrated_areas_peak_one[col_num], 3)}")
        pass

    ratio_list_areas.append(round(integrated_areas_peak_two[col_num] / integrated_areas_peak_one[col_num], 4))

# Print the accumulated integrated areas
# print(f'Integrated areas for peak one: {integrated_areas_peak_one}')
# print(f'Integrated areas for peak two: {integrated_areas_peak_two}')


before_bias = ratio_list_areas


color = ['RdBu_r']
num_range = [min(ratio_list_areas), max(ratio_list_areas)]
nested_before_bias = [before_bias[i:i + 40] for i in range(0, len(before_bias), 40)]
fig = px.imshow(nested_before_bias, text_auto=False, origin='upper', color_continuous_scale=color[0], zmin= num_range[0], zmax= num_range[1],  title= 'EELS Before Biasing')
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()

