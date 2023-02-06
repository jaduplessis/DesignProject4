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
        subject = filename_components[0]

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


    def F_mean(self):
        pass


    def calculate_features(self, data):
        """
        Function to calculate features for a single column of data.
        
        args:
            data: pandas series of data

        returns:
            features: list of features. A row of a dataframe.
        """
        # Calculate features
        F_mean = data.mean()
        F_min = data.min()
        F_max = data.max()
        F_sum = data.sum()
        F_prod = data.prod()
        F_std = data.std()
        F_mc = np.mean(np.diff(np.sign(data)))
        F_zc = np.sum(np.diff(np.sign(data)) != 0)
        F_iqr = np.subtract(*np.percentile(data, [75, 25]))
        F_skew = data.skew()
        F_25 = np.percentile(data, 25)
        F_75 = np.percentile(data, 75)
        F_kr = data.kurtosis()
        F_seg = np.sum(np.square(data))
        F_sep = -np.sum(np.square(data) * np.log(np.square(data)))

        # Create list of features
        features = pd.DataFrame({'F_mean': F_mean, 'F_min': F_min, 'F_max': F_max, 'F_sum': F_sum, 'F_prod': F_prod,
        'F_std': F_std, 'F_mc': F_mc, 'F_zc': F_zc, 'F_iqr': F_iqr, 'F_skew': F_skew, 'F_25': F_25, 'F_75': F_75, 
        'F_kr': F_kr, 'F_seg': F_seg, 'F_sep': F_sep}, index=[0])

        return features


    def normalize(self, row):
        """
        Function to normalize a row of data using min-max normalization.
        
        args:
            row: pandas series of data
            
        returns:
            normalized_row: pandas series of normalized data
        """
        # Calculate min and max
        min = row.min()
        max = row.max()

        # Normalize
        normalized_row = (row - min) / (max - min)
        print(normalized_row)
        return normalized_row       

        
    def feature_extraction(self):
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
            # Create empty dataframe of features as columns
            feature_df = pd.DataFrame(columns=['F_mean', 'F_min', 'F_max', 'F_sum', 'F_prod', 
            'F_std', 'F_mc', 'F_zc', 'F_iqr', 'F_skew', 'F_25', 'F_75', 'F_kr', 'F_seg', 'F_sep'])

            for column in window.columns:

                data = window[column]

                features = self.calculate_features(data)
                # features = self.normalize(features)
                
                feature_df = pd.concat([feature_df, features], ignore_index=True)
                
            # Save to csv
            feature_df.to_csv(f'PreProcessing/USC/Features/{subject}_{activity}_{trial}_{index}.csv', index=False)



        
window = SlidingWindow('PreProcessing/USC/RawData/Subject1_a1t1.csv')
window.feature_extraction()


