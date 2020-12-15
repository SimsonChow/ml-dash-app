import pandas as pd
import numpy as np
import pickle

def train_model(train_model, X_train, y_train):
    """
    Trains a classification model
    Args:
        model: sklearn estimator for classification
        X_train: Model features training data values
        y_train: Model target variable training data values
    Returns:
        model: trained classification model
    """
    model = train_model
    model.fit(X_train, y_train)

    return model

def load_model(model, filepath):
    """
    Load model from output folder 

    Args:
        filepath: location of mdoel pkl file

    returns:
        model: trained model
    """
    
    with open(filepath + '.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    with open(filepath + '.pkl', 'rb') as file:
        trained_model = pickle.load(file)
    return trained_model


