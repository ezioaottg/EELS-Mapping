import pandas as pd
from scipy.signal import find_peaks
import numpy as np
from numpy.typing import NDArray



class findPeaks():
    def __init__(self, x_df, y_df):
        self.__x_df: pd.DataFrame = x_df
        self.__y_df: pd.DataFrame = y_df
        self.__peaks: dict[str, np.ndarray[int]] = None
        

    def return_peaks(self,data_set: list[float] | np.ndarray[float],
               height: int = 35,
               distance: int = 20,
               tune: int = 180) -> tuple[np.ndarray[int], bool]:
        """Returns the two largest peaks in the signal and their index in the dataset in an array"""
        error = False
        peaks, properties = find_peaks(data_set, height=height, distance=distance)

        for i in range(tune):
            if (len(peaks) > 2):
                height += 0.6
            if(len(peaks) < 2):
                height -= 0.6

            peaks, properties = find_peaks(data_set, height=height, distance=20)
    

        if (len(peaks) > 2):
            height += 1
            print("TOO MANY PEAKS")
            error = True
        if(len(peaks) < 2):
            height -= 1
            print("TOO FEW PEAKS")
            error = True
        
        return peaks, error

    def set_peaks(self, **kwargs) -> None:
        """
        Iterates over the columns of a dataframe to find the peaks of the data in that column.
        sets self.__peaks, a dict, where the keys are the headers of each column in the dataframe and the values are numpy arrays containing the index of where the peaks are located
        """
        peaks_list = {}
        for column_name in self.__y_df:
            data = self.__y_df[column_name]
            peaks_list[column_name], error_bool = self.return_peaks(data, **kwargs)

            if error_bool:
                raise ValueError("Could not find peaks for " + column_name)
    
        self.__peaks = peaks_list


    def get_peak_window_indices(self, window_size: int = 8) -> dict[str, np.ndarray[int]]:
        """
        Iterates over dictionary whose values are numpy arrays of two integers that represents the indices of the two highest peaks in a dataset.
        The function then turns those peaks into a range of values determined by the window-size.

        It returns a similar dictionary as the input dictionary where the keys are the same,
        but the values are now 2D numpy arrays where each row represents the start and end of a peak "window".

        """
    
        peak_windows = {}

        for key, peak in self.__peaks.items():
            lower_peak = peak[0]
            upper_peak = peak[1]

            lower_peak_range = [lower_peak - window_size, lower_peak + window_size]
            upper_peak_range = [upper_peak - window_size, upper_peak + window_size]
            peak_windows[key] = np.asarray([lower_peak_range, upper_peak_range])
    
        return peak_windows


    def get_x_window(self):
        "Converts the indices of the peak windows to the x_values that correspond to those indices"
        "The argument peak just chooses which of the peaks you're trying to get the window of. i.e. peak = 0 for the first, peak = 1 for the second"

        "returns a dictionary with keys being the column names and values a numpy array of the x_values we want."
        peak_windows = self.get_peak_window_indices()
        x0_windows = {}
        x1_windows = {}
        x = self.__x_df
        for key, window in peak_windows.items():
            window_indices = window[0]
            x0_windows[key] = np.array(x[window_indices[0]:window_indices[1]])

        for key, window in peak_windows.items():
            window_indices = window[1]
            x1_windows[key] = np.array(x[window_indices[0]:window_indices[1]])


        return x0_windows, x1_windows

    def get_y_window(self):
        peak_windows = self.get_peak_window_indices()
        y0_windows = {}
        y1_windows = {}

        for key, window in peak_windows.items():
            y = self.__y_df[key]
            window_indices = window[0]
            y0_windows[key] = np.array(y[window_indices[0]:window_indices[1]])

        for key, window in peak_windows.items():
            y = self.__y_df[key]
            window_indices = window[1]
            y1_windows[key] = np.array(y[window_indices[0]:window_indices[1]])

        return y0_windows, y1_windows

    





    


