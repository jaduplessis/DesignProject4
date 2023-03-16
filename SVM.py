from sklearn import svm

def SVMTraining(L):
    # L: labelled samples
    
    # separate features and labels
    X = L[:,:-1] # features
    y = L[:,-1]  # labels

    # define the SVM model
    clf = svm.SVC(kernel='linear', C=1, probability=True)

    # train the SVM model
    clf.fit(X, y)

    # return the trained SVM model
    return clf