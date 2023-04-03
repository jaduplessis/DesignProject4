import pandas as pd
import numpy as np


class SlidingWindow:
    def __init__(self, filename, window_size=6, frequency=100):
        self.filename = filename.split('.')[0]
        self.df = pd.read_csv(filename)
        self.window_size = window_size
        self.frequency = frequency
        self.action_labels = {
            '1': "Walking Forward",
            '2': "Walking Left",
            '3': "Walking Right",
            '4': "Walking Upstairs",
            '5': "Walking Downstairs",
            '6': "Running Forward",
            '7': "Jumping Up",
            '8': "Sitting",
            '9': "Standing",
            '10': "Sleeping",
            '11': "Elevator Up",
            '12': "Elevator Down"
        }

    def split_filename(self):
        """ This function splits the filename into its components. """
        # Split filename into components
        file = self.filename.split('/')[-1]
        filename_components = file.split('_a')

        # Get subject number
        subject = filename_components[0].split('\\')[-1]

        # Split activity and trial number
        test_components = filename_components[1].split('t')

        activity = self.action_labels[test_components[0]]
        trial = test_components[1]

        return subject, activity, trial


    def get_windows(self):
        # Calculate number of rows in each window
        window_size_in_rows = self.window_size * self.frequency

        # Calculate number of windows rounded down to nearest integer
        num_windows = int(np.floor(len(self.df) / window_size_in_rows))

        # Loop through and create windows
        windows = []
        window_i = 0
        while window_i < num_windows:
            # Get start and end of window
            start = window_i * window_size_in_rows
            end = start + window_size_in_rows

            # Get window
            window = self.df.iloc[start:end]

            # Add window to list
            windows.append(window)

            # Increment window index
            window_i += 1

        return windows


    def calculate_features(self, data):
        """
        Function to calculate features for a single column of data.
        
        args:
            data: pandas series of data

        returns:
            features: list of features. A row of a dataframe.
        """
        # Calculate features
        F_mean = data.mean() # $ \s
        F_min = data.min() # $ min(x_i) $
        F_max = data.max() # $ max(x_i) $
        F_sum = data.sum() # $ \sum_{i=1}^{n} x_i $
        F_prod = data.prod() # $ \prod_{i=1}^{n} x_i $
        F_std = data.std() # $ \sqrt{\sum_{i=1}^{n} (x_i - \mu)^2 / n} $
        F_mc = np.mean(np.diff(np.sign(data))) # $ \sum_{i=1}^{n-1} \frac{1}{2} |x_i - x_{i+1}| $
        F_zc = np.sum(np.diff(np.sign(data)) != 0) # $ \sum_{i=1}^{n-1} |x_i - x_{i+1}| $
        F_iqr = np.subtract(*np.percentile(data, [75, 25])) # $ \frac{1}{2} (x_{75} - x_{25}) $
        F_skew = data.skew() # $ \frac{\sum_{i=1}^{n} (x_i - \mu)^3 / n}{\sigma^3} $
        F_25 = np.percentile(data, 25) # $ x_{25} $
        F_75 = np.percentile(data, 75) # $ x_{75} $
        F_kr = data.kurtosis() # $ \frac{\sum_{i=1}^{n} (x_i - \mu)^4 / n}{\sigma^4} $
        F_seg = np.sum(np.square(data)) # $ \sum_{i=1}^{n} x_i^2 $
        F_sep = -np.sum(np.square(data) * np.log(np.square(data))) # $ -\sum_{i=1}^{n} x_i^2 \log(x_i^2) $

        # Create list of features
        features = [F_mean, F_min, F_max, F_sum, F_prod, F_std, F_mc, F_zc, F_iqr, F_skew, F_25, F_75, F_kr, F_seg, F_sep]

        return features


    def normalize(self, row):
        """
        Function to normalize a row of data using min-max normalization.
        
        args:
            list: list of feature data
            
        returns:
            list: normalized list of feature data
        """
        # Get min and max of row
        row_min = min(row)
        row_max = max(row)

        # Normalize row
        normalized_row = [(x - row_min) / (row_max - row_min) for x in row]

        return normalized_row

        
    def feature_extraction(self, window_path):
        """ 
        This function converts a large set of time series data into windows of feature extracted data.

        Feature parameters:
            - mean (F_mean)
            - minimum (F_min)
            - maximum (F_max)
            - summation (F_sum)
            - product (F_prod)
            - standard deviation (F_std)
            - mean crossing (F_mc)
            - zero crossing (F_zc)
            - interquartile range (F_iqr)
            - skewness (F_skew)
            - 25th percentile (F_25)
            - 75th percentile (F_75)
            - kurtosis (F_kr)
            - Spectral energy (F_seg)
            - Spectral entropy (F_sep)

        """
        # Determine activity label
        subject, activity, trial = self.split_filename()

        # Create windows
        windows = self.get_windows()

        for index, window in enumerate(windows):
            # Feature options
            feature_types = ['F_mean', 'F_min', 'F_max', 'F_sum', 'F_prod', 'F_std', 'F_mc',
                              'F_zc', 'F_iqr', 'F_skew', 'F_25', 'F_75', 'F_kr', 'F_seg', 'F_sep']

            dimensions =  window.columns # ["acc_x, w/ unit g", "acc_y, w/ unit g", "acc_z, w/ unit g"]

            # Create list to store features and headers
            features = []
            headers = []

            # Create dataframe to store features. For each dimension, there are 15 features.
            # Each window has 3 dimensions, so there are 45 features per window.
            for dimension in dimensions:
                # Get data for dimension
                data = window[dimension]

                # Calculate features and append to list
                features.extend(self.calculate_features(data))

                # Create headers
                for feature in feature_types:
                    headers.append(f'{dimension}_{feature}')


            # Create dataframe
            # ValueError: Shape of passed values is (45, 1), indices imply (45, 45)
            # feature_df = pd.DataFrame(features, columns=headers)
            feature_df = pd.DataFrame(features).T
            feature_df.columns = headers
            
            # Save to csv
            print(f'Saving {subject}_{activity}_{trial}_{index}.csv')
            file_path = f'{window_path}/{subject}_{activity}_{trial}_{index}.csv'

            feature_df.to_csv(file_path, index=False)

        


