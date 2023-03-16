import os
import sys

import numpy as np
import pandas as pd

from SlidingWindow import SlidingWindow

# Path to the data
path = os.path.join(os.getcwd(), 'PreProcessing\\USC\\RawData')

# Get all files in the directory
files = os.listdir(path)

# Iterate through all files and create SlidingWindow objects
for file in files:
    file_path = os.path.join(path, file)

    # Create SlidingWindow object
    SW = SlidingWindow(file_path)

    SW.feature_extraction()
    