import numpy as np
import pandas as pd
from typing import Callable


def cubic(x, a, b, c, d):
    """
    Defines a cubic function to be used in fitting
    """


    return a * x**3 + b * x**2 + c * x + d

def quartic(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def quintic(x,a,b,c,d,e,f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

def sextic(x,a,b,c,d,e,f,g):
    return a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x + g

def numerically_integrate(x_data: np.ndarray,
                          y_data: np.ndarray):
    
    integral = np.trapz(y = y_data, x=x_data)
    return integral

def integrate_fit(x_data: np.ndarray,
                  y_data: np.ndarray,
                  fitted_function: Callable[[int], int],
                  params: list | np.ndarray,
                  linspace_num: int = 50):
    
    x_values = np.linspace(np.nanmin(x_data), np.nanmax(x_data), linspace_num)
    y_values = fitted_function(x_values, *params)


    