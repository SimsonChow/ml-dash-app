import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.model_selection import train_test_split
from config import *
from metrics_visualizer import Visualizer
import json, shap, base64

def processInputDataset(dataset, targetVariable):
    dataFrame = pd.read_csv(dataset)
    X = dataFrame.drop(targetVariable, axis = 1)
    y = dataFrame[targetVariable]
    valueCount = y.value_counts().to_json()
    labels = list(json.loads(valueCount).keys())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    return X_train, X_test, y_train, y_test, labels

def modelTraining(model, X_train, y_train, OUTPUT_PATH):
    trainedModel = model.fit(X_train, y_train)
    return trainedModel

def shapTreeExplainerObject(model, label, X_train):
    treeExplainer = shap.TreeExplainer(model)
    shapValues = treeExplainer.shap_values(X_train)
    finalCorrelationDataframe = shapCorrelationDataframe(shapValues, label, X_train)
    return finalCorrelationDataframe

def shapCorrelationDataframe(shapValues, label, X_train):
    shapValueDataframe = pd.DataFrame(shapValues[label])
    featureNameList = X_train.columns
    shapValueDataframe.columns = featureNameList
    trainValueDataFrame = X_train.copy().reset_index().drop('index',axis=1)

    correlationList = list()
    for feature in featureNameList:
        correlation = np.corrcoef(shapValueDataframe[feature],trainValueDataFrame[feature])[1][0]
        correlationList.append(correlation)
    finaleCorrelationDataFrame = pd.concat([pd.Series(featureNameList),pd.Series(correlationList)],axis=1).fillna(0)
    
    finaleCorrelationDataFrame.columns  = ['Features','Correlation']
    return finaleCorrelationDataFrame

def saveObject(objectOutput, objectName, OUTPUT_PATH):
    objectPath = "".join([OUTPUT_PATH, objectName, ".pkl"])
    with open(objectPath, 'wb') as pklFile:
        pkl.dump(objectOutput, pklFile)

def saveDataFrame(dataframe, OUTPUT_PATH):
    dataframe.to_csv(OUTPUT_PATH, index = False)

def loadObject(objectName, OUTPUT_PATH):
    objectPath = "".join([OUTPUT_PATH, objectName, ".pkl"])
    objectOutput = pd.read_pickle(objectPath)
    return objectOutput

def createMetricImgs(X_train, X_test, y_train, y_test, labels, model, visualizer, full_filepath, filename):
    metricsViz = Visualizer(X_train, X_test, y_train, y_test, labels, model, visualizer, full_filepath,filename)
    metricsViz.evaluate()
    metricsViz.save_img()

def encodedImage(imageFile):
    """
    Encode image for Dash Application
    Args:
        full_filepath: location of image
    returns:
        encode: decoded image
    """
    imageFile = "".join([METRICS_PATH, imageFile])
    encoded = base64.b64encode(open(imageFile, 'rb').read())
    return 'data:image/jpg;base64,{}'.format(encoded.decode())