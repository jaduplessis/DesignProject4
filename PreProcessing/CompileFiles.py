# '''
# This script compiles the feature extracted data into a single csv file.
# It loads the data from the PreProcessing\USC\Features directory and compiles it into a single csv file.
# It has the option to randomise the rows of the data.
# Alongside the csv file, there is another csv file that maps the rows to Subject, Activity and Window.
# '''

import os
import sys

import numpy as np
import pandas as pd

def compile_files(window_size=7):
    # Path to the data
    path = os.path.join(os.getcwd(), 'PreProcessing\\USC\\Features_' + str(window_size))

    # Get all files in the directory
    files = os.listdir(path)

    # Create empty dataframe for all data
    df = pd.DataFrame()

    # Randomly generate indices in range of number of files
    # labelled_number = num_labels
    # indices = np.random.choice(len(files), labelled_number, replace=False)
    # assert len(indices) == labelled_number

    # Create empty dataframe for mapping data
    df_map = pd.DataFrame()

    # Iterate through all files and load data
    for index, file in enumerate(files):
        file_path = os.path.join(path, file)

        # Load data
        df_temp = pd.read_csv(file_path)

        # Get subject, activity and window
        subject = file.split('_')[0]
        activity = file.split('_')[1]
        window = file.split('_')[2].split('.')[0]

        # Add label column: null for unlabelled data and subject_activity for labelled data
        df_temp['Index'] = index
        df_temp['Subject'] = subject
        df_temp['Label'] = activity

        # Append data to main dataframe
        df = pd.concat([df, df_temp], ignore_index=True)

        # Print progress 
        print('Window Size: ' + str(window_size) + ' | Progress: ' + str(index) + '/' + str(len(files)), end='\r')


    # Save data to csv
    df.to_csv('PreProcessing\\USC\\CompiledData_' + str(window_size) + '.csv', index=False)



if __name__ == '__main__':
    for window_size in ['4', '5', '6', '7', '8', '9', '10']:
        compile_files(window_size)


