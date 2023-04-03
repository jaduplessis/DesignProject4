import pandas as pd
import numpy as np
import os

# Import sklearn modules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.svm import OneClassSVM, SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score

import copy

class OCluDAL():
    def __init__(self, file_path, annotations, damping=0.75, preference=-90):
        self.df_main = pd.read_csv(file_path)
        self.annotations = annotations
        self.data = pd.DataFrame(columns=['Accuracy', 'F1 Score', 'Train Accuracy', 'Number of Annotations', 'damping', 'preference', 'Train_type'])# Concat the results to data df
        self.damping = damping
        self.preference = preference
        self.training_type = 'Random'


    def initialise_data(self, model_type='SVM', indices=None, output_path=None):
        if output_path is not None:
            self.output_path = output_path
        else:
            self.model_type = model_type
            dir = os.listdir('Results')
            id = str(len(dir) + 1)
            self.output_path = f'{self.model_type}_{id}.csv'
        # Get length of data
        df_length = len(self.df_main)
        print(f"Total data: {df_length}")

        if indices is None:
            # Randomly generate indices in range of number of files to be annotated
            indices = np.random.choice(len(self.df_main), self.annotations, replace=False)
 
        assert len(indices) == self.annotations
        print(f"Annotations: {self.annotations}")

        # Create labelled and unlabelled dataframes
        labelled = self.df_main.iloc[indices]
        self.labelled = labelled.drop(['Subject', 'Index'], axis=1)

        unlabelled = self.df_main.drop(indices)
        self.unlabelled = unlabelled.drop(['Subject', 'Index'], axis=1)
        assert len(self.labelled) + len(self.unlabelled) == len(self.df_main)

        # Initialise model
        if model_type == 'SVM':
            self.clf = SVC(kernel='rbf', C=1, probability=True)



    def preprocessing(self):
        # Standardise data
        print('Preprocessing data: Applying StandardScaler')
        scaler = StandardScaler()

        # Apply scaler to data
        self.unlabelled_X_original = scaler.fit_transform(self.unlabelled.drop(['Label'], axis=1))
        self.labelled_X_original = scaler.transform(self.labelled.drop(['Label'], axis=1))

        # Save scaled data to csv
        scaled_df = pd.DataFrame(self.unlabelled_X_original)
        scaled_df.to_csv('scaled_data.csv', index=False)

        self.unlabelled_y_original = self.unlabelled['Label'].values  
        self.labelled_y_original = self.labelled['Label'].values

        # Get unique labels
        self.unique_labels = np.unique(self.labelled_y_original)
        print('Unique labels: ', self.unique_labels)

        # Initialize sets to be used in iterations
        self.labelled_X_new = self.labelled_X_original.copy()
        self.unlabelled_X_new = self.unlabelled_X_original.copy()

        self.labelled_y_new = self.labelled_y_original.copy()
        self.unlabelled_y_new = self.unlabelled_y_original.copy()


    def oracle_annotations(self, indices):
        representative_X = self.unlabelled_X_new[indices]
        representative_y = self.unlabelled_y_new[indices]

        # Update labelled set with newly annotated samples
        self.labelled_X_new = np.vstack([self.labelled_X_new, representative_X])
        self.labelled_y_new = np.hstack([self.labelled_y_new, representative_y])

        assert len(self.labelled_X_new) == len(self.labelled_y_new)

        # Remove representative samples from unlabelled set
        self.unlabelled_X_new = np.delete(self.unlabelled_X_new, indices, axis=0)
        self.unlabelled_y_new = np.delete(self.unlabelled_y_new, indices, axis=0)


    def train_model(self):
        X = self.labelled_X_new.copy()

        # Convert labels to array
        y = self.labelled_y_new.copy()

        clf = self.clf

        # train the SVM model
        clf.fit(X, y)

        self.run_classification(clf)
        return clf


    def BvSB_Sampling(self, probalities, n):
        """
        https://doi.org/10.1109/CVPR.2009.5206627

        Function implements Best vs Second Best sampling strategy. Instead of relying on the entropy score,
        we take a more greedy approach. We consider the difference between the probability values of the 
        two classes having the highest estimated probability value as a measure of uncertainty. Since it is 
        a comparison of the best guess and the second best guess, we refer to it as the 
        Best-versus-Second-Best (BvSB) approach.

        Parameters
        ----------
        probalities : array-like, shape (n_samples, n_classes)
            Probability estimates for each class for each sample.
        n : int
            Number of samples to be selected
        """
        # Get the number of samples and classes
        n_samples, n_classes = probalities.shape

        # Get the indices of the two classes with the highest probability
        # for each sample
        max_indices = np.argmax(probalities, axis=1)
        second_max_indices = np.argsort(probalities, axis=1)[:, -2]

        # Get the probability values of the two classes with the highest probability
        # for each sample
        max_prob = probalities[np.arange(n_samples), max_indices]
        second_max_prob = probalities[np.arange(n_samples), second_max_indices]

        # Calculate the difference between the two classes with the highest probability
        # for each sample
        diff = max_prob - second_max_prob

        # Get the indices of the n samples with the lowest difference
        indices = np.argsort(diff)[:n]

        return indices
    

    def Random_sampling(self, n):
        # Randomly select indices from unlabelled X
        indices = None

        return indices


    def Entropy_Sampling(self, probalities, n):
        """
        Function selects the n-highest entropy probabilities and returns the indices of them
        """
        pass



    def step1(self, max_iter=5):

        self.train_model()
        self.training_type = 'AP'
        # Start iterations
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1    
            print(f"Iteration {iter_count}")
                
            masks = []

            # Novelty detection using OCSVM
            for label in self.unique_labels:
                # Fit OCSVM
                svm = OneClassSVM().fit(self.labelled_X_new[self.labelled_y_new == label])
                novel_mask_i = svm.predict(self.unlabelled_X_new) == -1
                
                masks.append(novel_mask_i)

            novel_mask = np.all(masks, axis=0)
            novel_X = self.unlabelled_X_new[novel_mask]
            print(f"Novelty detected: {len(novel_X)}")

            # Clustering to select representative samples for annotation using Affinity Propagation
            if len(novel_X) > 0:
                ap = AffinityPropagation(damping=self.damping, preference=self.preference).fit(novel_X)
                representative_X = ap.cluster_centers_
                print(f"Representative samples chosen for annotation: {len(representative_X)}")
            else:
                print("No novelty detected. Skipping clustering.")
                break

            # Find row indices of representative samples
            representative_indices = []
            for sample in representative_X:
                representative_indices.append(np.where((self.unlabelled_X_new == sample).all(axis=1))[0][0])

            # Update labelled and unlabelled sets
            self.oracle_annotations(representative_indices)

            # Train model
            self.train_model()


    def step2(self, max_iter=20, n=5, model_type='SVM', max_samples=1000, sampling_type='BvSB'):
        """
        Perform uncertainty sampling and model training
        """
        print("Starting uncertainty sampling and model training")
        iter = 0
        num_samples = len(self.labelled_X_new)
        self.training_type = 'BvSB'

        while iter <= max_iter and num_samples < max_samples:
            print(f"Iteration {iter}  /{max_iter}     |Labelled data size: {len(self.labelled_X_new)}  |Unlabelled data size: {len(self.unlabelled_X_new)}", end='\r')
            # Train SVM
            clf = self.train_model()        
            
            # Run classification on unlabelled data
            self.run_classification(clf)

            # Get probability estimates for unlabelled data
            probalities = clf.predict_proba(self.unlabelled_X_new)

            if sampling_type == 'BvSB':
                # Find most useful samples to annotate
                indices = self.BvSB_Sampling(probalities, n)
            elif sampling_type == 'Random':
                indices = self.Random_sampling
            elif sampling_type == 'Entropy':
                indices = self.Entropy_sampling
            
            # Update labelled and unlabelled sets
            self.oracle_annotations(indices)

            # Update iteration count and number of samples
            num_samples = len(self.labelled_X_new)
            iter += 1
        
        # Train final SVM
        clf = self.train_model()
        
        # Run classification on unlabelled data
        self.run_classification(clf)

        return clf
    

    def run_classification(self, clf):
        """
        Attempt to classify remaining data points.

        Parameters
        ----------
        clf : sklearn classifier
            Trained classifier.

        Returns
        -------
        f1_score : float
            F1 score of the final model on the remaining data points.
        accuracy : float
            Accuracy of the final model on the remaining data points.
        num_annotations : int
            Number of annotations present at the point of classification.
        """
        # Load final model and remaining data
        X_test = self.unlabelled_X_new
        X_train = self.labelled_X_new

        # Get predictions
        y_test_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)

        # Get true labels
        y_test_true = self.unlabelled_y_new
        y_train_true = self.labelled_y_new

        # Calculate f1 score
        f1 = f1_score(y_test_true, y_test_pred, average='weighted')

        # Calculate accuracy
        test_accuracy = accuracy_score(y_test_true, y_test_pred)
        train_accuracy = accuracy_score(y_train_true, y_train_pred)

        # self.data = pd.DataFrame(columns=['model_type', 'accuracy', 'f1_score', 'Train Accuracy', 'Number of Annotations', 'damping', 'preference'])# Concat the results to data df
        df = pd.DataFrame({
            'Accuracy': test_accuracy,
            'F1 Score': f1,
            'Train Accuracy': train_accuracy,
            'Number of Annotations': len(self.labelled_X_new),
            'damping': self.damping,
            'preference': self.preference,
            'Train_type': self.training_type
        }, index=[0])

        # Concat the results to data df
        self.data = pd.concat([self.data, df], ignore_index=True)

        self.data.to_csv(f'Results\\{self.output_path}', index=False)

        return f1, test_accuracy, len(self.labelled_X_new)

    
    def copy(self):
        """
        Copy the object.
        """
        return copy.deepcopy(self)


if __name__ == '__main__':
    from OCluDAL import OCluDAL

    from OCluDAL import OCluDAL
    import numpy as np

    # Path to the data
    indices = np.random.choice(3000, 10, replace=False)
    # indices = np.arange(138, 148)
    path = 'PreProcessing\\USC\\CompiledData_7.csv'
    annotations = 10

    damping_pref_tuples = {
        'combination1': (0.75, -190),
        'combination2': (0.75, -180),
        'combination3': (0.75, -300),
    }

    for key, (damping, pref) in damping_pref_tuples.items():
        OC = OCluDAL(path, annotations, damping=damping, preference=pref)
        OC.initialise_data(indices=indices, output_path=f'test.csv')
        OC.preprocessing()
        OC.step1(max_iter=1)
        clf = OC.step2(max_iter=500, n=10, max_samples=500)

