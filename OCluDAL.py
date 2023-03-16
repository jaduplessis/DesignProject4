import pandas as pd
import numpy as np

# Import sklearn modules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.svm import OneClassSVM, SVC
from sklearn import metrics


class OCluDAL():
    def __init__(self, file_path, annotations):
        self.df_main = pd.read_csv(file_path)
        self.annotations = annotations


    def initialise_data(self):
        # Get length of data
        df_length = len(self.df_main)
        print(f"Total data: {df_length}")


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


    def preprocessing(self):
        # Standardise data
        print('Preprocessing data: Applying StandardScaler')
        scaler = StandardScaler()

        # Apply scaler to data
        self.unlabelled_X_original = scaler.fit_transform(self.unlabelled.drop(['Label'], axis=1))
        self.labelled_X_original = scaler.transform(self.labelled.drop(['Label'], axis=1))

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


    def train_SVM(self):
        X = self.labelled_X_new.copy()

        # Convert labels to array
        y = self.labelled_y_new.copy()


        # define the SVM model
        clf = SVC(kernel='rbf', C=1, probability=True)

        # train the SVM model
        print('Training SVM...')
        clf.fit(X, y)

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

        # Get the indices of the n samples with the highest difference
        indices = np.argsort(diff)[-n:]

        return indices


    def step1(self, max_iter=5):
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
                ap = AffinityPropagation().fit(novel_X)
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


    def step2(self, max_iter=20):

        iter = 0

        while iter <= max_iter:
            # Train SVM
            clf = self.train_SVM()        

            # Get probability estimates for unlabelled data
            print('Predicting...')
            probalities = clf.predict_proba(self.unlabelled_X_new)

            # Find most useful samples to annotate
            indices = self.BvSB_Sampling(probalities, 20)
            
            # Update labelled and unlabelled sets
            self.oracle_annotations(indices)

            # Print diagnostics on data sizes
            print(f"Labelled data size: {len(self.labelled_X_new)}")
            print(f"Unlabelled data size: {len(self.unlabelled_X_new)}")

            iter += 1
        
        # Train final SVM
        clf = self.train_SVM()

        return clf
        

if __name__ == '__main__':
    from OCluDAL import OCluDAL

    # Path to the data
    path = 'PreProcessing\\USC\\CompiledData.csv'
    annotations = 200

    # Create OCluDAL object
    OC = OCluDAL(path, annotations)

    OC.initialise_data()
    OC.preprocessing()
    OC.step1()
    OC.step2()