import pandas as pd
from scipy.signal import find_peaks
import numpy as np

def return_peaks(data_set: list[float] | np.ndarray[float],
               height: int = 35,
               distance: int = 20,
               tune: int = 150) -> np.ndarray[int]:
    """Returns the two largest peaks in the signal and their index"""
    
    peaks, properties = find_peaks(data_set, height=height, distance=distance)

    for i in range(tune):
        if (len(peaks) > 2):
            height += 0.6
        if(len(peaks) < 2):
            height -= 0.6

        peaks, properties = find_peaks(y, height=height, distance=20)
    

    if (len(peaks) > 2):
        height += 1
        print("TOO MANY PEAKS")
    if(len(peaks) < 2):
        height -= 1
        print("TOO FEW PEAKS")
        
    return peaks

def return_all_peaks(data_frame: pd.DataFrame, **kwargs) -> dict[str, int]:
    """Iterates over the columns of a dataframe to find the peaks. Remove any x-axis before using please"""
    peaks_list = {}
    for column_name in data_frame:
        data = data_frame[column_name]
        peaks_list[column_name] = return_peaks(data, **kwargs)

def get_peak_window():

    


