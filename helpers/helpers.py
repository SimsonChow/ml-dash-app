import pandas as pd
import numpy as np

def train_model(model, X_train, y_train):
    """
    Trains a classification model
    Args:
        model: sklearn estimator for classification
        X_train: Model features training data values
        y_train: Model target variable training data values
    Returns:
        model: trained classification model
    """
    model = model
    model.fit(X_train, y_train)

    return model