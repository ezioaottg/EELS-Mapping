import pandas as pd


def column_labels():
    labels = [f"{letter}{num}"
          for letter in (chr(c) for c in range(ord('A'), ord('V') + 1))
          for num in range(1, 41)]

    return