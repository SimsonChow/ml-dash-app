import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load input data from flat file with target variable as first column followed by however many feature variables

    Args:
        filepath: location of flat file

    returns:
        labels: list of class labels for classification mdoel
        features: list of string values containing names of model features
        target: list of singular string value containing the target variable name
        X_train: model features training of data values
        X_test: model features test data values
        y_train: mdoel target variable training data values
        y_test: model target variable test data values

    """
    df = pd.read_csv(filepath)
    target = df.columns[0]
    features = [col for col in df.columns if col not in target]

    X = df[features].values
    y = df[target].values
    labels = df[target].unique().tolist()
    labels.sort()

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = .30, random_state=42)

    return X_train, y_train, X_test, y_test, labels, features, target