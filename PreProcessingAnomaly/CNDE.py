import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope


def split_k_folds(dff, k):
    """
    Randomly splits the dataset into k-folds.
    """
    df = dff.copy()
    idx = np.random.permutation(df.index)
    # ValueError: cannot reindex on an axis with duplicate labels
    df = df.reindex(idx)
    df = df.reset_index(drop=True)

    k_folds = []
    # Overflow remainder is added to the last fold
    remainder = len(df) % k
    fold_size = int(len(df) / k)
    for i in range(k):
        if i == k - 1:
            k_folds.append(df[i * fold_size: (i + 1) * fold_size + remainder])
        else:
            k_folds.append(df[i * fold_size: (i + 1) * fold_size])

    return k_folds


def create_models(df, k_folds, method):
    """
    Creates a parent model using the algorithm in method and k-child models using the same algorithm.
    Parent model is trained on all data except.
    Child models are trained on each k-fold of data.

    Parameters:
    df: Pandas DataFrame
        The entire dataset
    k_folds: List
        List of k-folds of the dataset
    method: sklearn model
        The model to be used

    Returns:
    parent_model: sklearn model
        The parent model
    child_models: List
        List of k child models
    """
    parent_model = method
    parent_model.fit(df)

    child_models = []
    for i in range(len(k_folds)):
        child_models.append(method)
        child_models[i].fit(k_folds[i])

    return parent_model, child_models


class Models:
    def __init__(self, path, k, contamination):
        self.df_train = pd.read_csv(path)
        self.k_folds = split_k_folds(self.df_train, k)
        self.methods =[IsolationForest(contamination=contamination), 
                       LocalOutlierFactor(novelty=True, contamination=contamination),
                       OneClassSVM(nu=contamination, gamma=0.1), 
                       EllipticEnvelope(support_fraction=0.95, contamination=contamination)]

    def instantiate_models(self):
        """ 
        Instantiates all models and child models
        """
        models = {}
        for method in self.methods:
            # Create string name of model
            model_name = str(method).split('(')[0]
            parent_model, child_models = create_models(self.df_train, self.k_folds, method)
            models[model_name] = {
                'parent_model': parent_model,
                'child_models': child_models,
                'weights': 1
            }
        self.models = models

# Compute internal consensus score
def compute_icv_ics(child_models, data_point):
    """
    Computes the internal consensus vote (ICV) and internal consensus score (ICS) for each data point in the training data.

    Parameters:
    child_models: List
        List of child models
    data_point: Pandas DataFrame row
        A single row of the training data

    Returns:
    icv: int

    """
    votes = []
    for child_model in child_models:
        votes.append(child_model.predict([data_point]))

    # Compute internal consensus score
    # ics = sum(votes) / len(votes)
    ics = child_model.decision_function([data_point])

    # If any of the votes are 1, then the internal consensus vote is 1
    if 1 in votes:
        icv = 1
    else:
        icv = 0

    return icv#, ics


def compute_cics(models, data_point):
    """
    Computes the consensus internal consensus score (CICS) for each data point in the training data.

    Parameters:
    models: List
        List of models
    data_point: Pandas DataFrame row
        A single row of the training data

    Returns:
    ICV: List
        List of internal consensus votes
    CICS: float
        The consensus internal consensus score
    """
    # Compute internal consensus score for each child model
    combined_score = []
    ICV = []
    total_weights = 0
    for model in models:
        # icv, ics = compute_icv_ics(models[model]['child_models'], data_point)
        icv = compute_icv_ics(models[model]['child_models'], data_point)
        ICV.append(icv)
        # combined_score.append(ics * models[model]['weights'])
        total_weights += models[model]['weights']

    # CICS = sum(combined_score) / total_weights

    return ICV#, CICS


def compute_cecs(models, data_point):
    """
    Computes the consensus external consensus score (CECS) for each data point in the training data.

    Parameters:
    models: List
        List of models
    data_point: Pandas DataFrame row
        A single row of the training data

    Returns:
    CECV: int
        The consensus external consensus vote
    CECS: float
        The consensus external consensus score
    """
    # Compute external consensus score for each model
    combined_score = []
    for model in models:
        # Predict data point using parent model
        ecs = models[model]['parent_model'].predict([data_point])

        combined_score.append(ecs)

    CECS = sum(combined_score) / len(models)

    if CECS >= 0:
        CECV = 1
    else:
        CECV = 0

    return CECV#, CECS


def calculate_weights(CECV_all, ICV_all, models):
    """
    Calculates the weights for each model.

    Parameters:
    CECV_all: List
        List of consensus external consensus votes
    ICV_all: List of Lists
        List of internal consensus votes
    models: Models object
        Models object
    
    Returns:
    models: Models object
        Updated models object
    """
    model_names = list(models.models.keys())

    errors = np.zeros(len(model_names))

    for xi in range(len(CECV_all)):
        Vi = CECV_all[xi]
        ICV = ICV_all[xi]
        for vi in ICV:
            if vi != Vi:
                errors[ICV.index(vi)] += 1
    print('\n --------------------- \nUpdating weights\n ---------------------')

    for model in model_names:
        # w_f = w_i * (e / n) * w_i
        weight_i = models.models[model]['weights']
        error_i = errors[model_names.index(model)]
        n = len(CECV_all)
        weight_f = weight_i - (error_i / n) * weight_i
        
        print('Model {} performance: {}/{}. Weight: {} -> {}'.format(model, n - error_i, n, weight_i, weight_f))

        models.models[model]['weights'] = weight_f


    return models
    

def train_ensemble(models, df_train):
    """
    Iterates through the training data and compute the CICS, CECS and ICV for each data point. 
    The weights for each model are then updated.
    """
    print('Training ensemble...')
    ICV_all = []
    CECV_all = []

    for i in range(len(df_train)):
    # for i in range(10):
        print('Training data point: {}/{}'.format(i+1, len(df_train)), end='\r')

        data_point = df_train.iloc[i]
        # ICV, CICS = compute_cics(models.models, data_point)
        ICV = compute_cics(models.models, data_point)
        ICV_all.append(ICV)
        # CECV, CECS = compute_cecs(models.models, data_point)
        CECV = compute_cecs(models.models, data_point)
        CECV_all.append(CECV)

    # Update weights
    models = calculate_weights(CECV_all, ICV_all, models)
    print('Training complete. \n------------------')
    return models


def calculate_CECS_CICS_scores(models, data):
    """
    Calculates the normality score for each data point in the data.

    Parameters:
    models: List
        List of models
    data: Pandas DataFrame
        Data to calculate normality score for

    Returns:
    normality_scores: List
        List of normality scores
    """

    # Calculate normality score
    print('------------------ \n')
    for model in models:
        # Predict data point using parent model
        ecs = models[model]['parent_model'].decision_function(data)

        ics = []
        for child_model in models[model]['child_models']:
            ics.append(child_model.decision_function(data))
        
        cics = sum(ics) / models[model]['weights']

        # If no scores present, create list. Otherwise concatenate
        if 'ecs' not in models[model]:
            models[model]['ecs'] = ecs
            models[model]['cics'] = cics
        else:
            models[model]['ecs'] = np.concatenate((models[model]['ecs'], ecs))
            models[model]['cics'] = np.concatenate((models[model]['cics'], cics))

    return models



def perform_CNDE(models):
    """
    Performs the CNDE algorithm on the training data.

    Parameters:
    models: List
        List of models
    df_train: Pandas DataFrame
        Training data

    Returns:
    models: List
        List of models
    """
    df_train = models.df_train

    # Train ensemble
    models = train_ensemble(models, df_train)

    # Score ensemble
    # models.models = calculate_CECS_CICS_scores(models.models, df_train)
    
    # for model in models.models:
    #     # Normalise scores
    #     models.models[model]['standardised_ecs'] = (models.models[model]['ecs'] - np.mean(models.models[model]['ecs'])) / np.std(models.models[model]['ecs'])
    #     models.models[model]['standardised_cics'] = (models.models[model]['cics'] - np.mean(models.models[model]['cics'])) / np.std(models.models[model]['cics'])

    #     # # Any values below -4 std are set to -4 std
    #     # models.models[model]['ecs'] = np.where(models.models[model]['ecs'] < np.mean(models.models[model]['ecs']) - 4 * np.std(models.models[model]['ecs']), np.mean(models.models[model]['ecs']) - 4 * np.std(models.models[model]['ecs']), models.models[model]['ecs'])
    #     # models.models[model]['cics'] = np.where(models.models[model]['cics'] < np.mean(models.models[model]['cics']) - 4 * np.std(models.models[model]['cics']), np.mean(models.models[model]['cics']) - 4 * np.std(models.models[model]['cics']), models.models[model]['cics'])

    #     # Print scores
    #     print('Model {} ECS: {}'.format(model, models.models[model]['ecs']))
    #     print('Model {} CICS: {}'.format(model, models.models[model]['cics']))

    # # Calculate average CICs and ECSs
    # CICS = np.average([models.models[model]['standardised_cics'] for model in models.models], axis=0)
    # CECS = np.average([models.models[model]['standardised_ecs'] for model in models.models], axis=0)

    # # Calculate normality score
    # # normality_scores = (CICS + CECS) / 2

    # # models.normality_scores = normality_scores
    # models.CICS = CICS
    # models.CECS = CECS

    models.weights = [models.models[model]['weights'] for model in models.models]
    return models


def test_ensemble(models, path):
    df_test = pd.read_csv(path)
    df_train = models.df_train

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Score ensemble
    models.models = calculate_CECS_CICS_scores(models.models, df)
    
    for model in models.models:
        # Normalise scores
        models.models[model]['standardised_ecs'] = (models.models[model]['ecs'] - np.mean(models.models[model]['ecs'])) / np.std(models.models[model]['ecs'])
        models.models[model]['standardised_cics'] = (models.models[model]['cics'] - np.mean(models.models[model]['cics'])) / np.std(models.models[model]['cics'])

        # Print scores
        print('Model {} ECS: {}'.format(model, models.models[model]['standardised_ecs']))
        print('Model {} CICS: {}'.format(model, models.models[model]['standardised_cics']))

    # Calculate average CICs and ECSs
    CICS = np.average([models.models[model]['standardised_cics'] for model in models.models], axis=0)
    CECS = np.average([models.models[model]['standardised_ecs'] for model in models.models], axis=0)

    # Calculate normality score
    normality_scores = (CICS + CECS) / 2

    # Concatenate train and test scores
    models.normality_scores = normality_scores
    models.CICS = CICS
    models.CECS = CECS

    return models
    


if __name__ == '__main__':
    from CNDE import Models
    from CNDE import perform_CNDE, test_ensemble

    import pandas as pd
    import numpy as np

    # Turn off user warnings
    import warnings
    warnings.filterwarnings('ignore')

    path_train = 'df_train_sleep.csv'
    path_test = 'df_test_sleep.csv'

    c = 0.01

    sleep1 = Models(path_train, k=3, contamination=c)
    sleep1.instantiate_models()
    sleep1 = perform_CNDE(sleep1)
    print('\n Testing ensemble')
    sleep1 = test_ensemble(sleep1, path_test)
    print(sleep1.normality_scores)