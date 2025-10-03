import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from typing import Callable

def column_labels():
    labels = [f"{letter}{num}"
          for letter in (chr(c) for c in range(ord('A'), ord('V') + 1))
          for num in range(1, 41)]

    return labels

def center_axis(data: np.ndarray):
    return data - data.mean()

def fit_and_integrate(x_data: np.ndarray,
                  y_data: np.ndarray,
                  fitted_function: Callable[[int], int],
                  integration_width: float = 0.5,
                  linspace_num: int = 1000):
    
    x_data = center_axis(x_data)
    popt, _ = curve_fit(fitted_function, x_data, y_data)

    x_fit = np.linspace(np.nanmin(x_data), np.nanmax(x_data))
    y_fit = fitted_function(x_fit, *popt)

    peak_x = x_fit[np.argmax(y_fit)]
    
    integrate_func = lambda x: fitted_function(x, *popt)
    integral, _ = quad(integrate_func, peak_x - integration_width, peak_x + integration_width)
    return integral