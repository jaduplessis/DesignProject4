import os
import sys

import numpy as np
import pandas as pd

from SlidingWindow import SlidingWindow

# Path to the data
path = os.path.join(os.getcwd(), 'PreProcessing\\USC\\RawData')

# Get all files in the directory
files = os.listdir(path)

window_sizes = [5, 6, 7, 8, 9, 10]

for window in window_sizes:
    # Create directory for window size
    window_path = os.path.join(os.getcwd(), 'PreProcessing\\USC\\Features_' + str(window))
    if not os.path.exists(window_path):
        os.mkdir(window_path)


    # Iterate through all files and create SlidingWindow objects
    for file in files:
        file_path = os.path.join(path, file)

        # Create SlidingWindow object
        SW = SlidingWindow(file_path, window_size=window)

        SW.feature_extraction(window_path=window_path)
    