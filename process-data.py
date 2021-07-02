import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.model_selection import train_test_split
from config import *
from metrics_visualizer import Visualizer
import json

def processInputDataset(dataset, targetVariable):
    dataFrame = pd.read_csv(dataset)
    X = dataFrame.drop(targetVariable, axis = 1)
    y = dataFrame[targetVariable]
    valueCount = y.value_counts().to_json()
    labels = list(json.loads(valueCount).keys())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    return X_train, X_test, y_train, y_test, labels

def saveTrainedModel(trainedModel, OUTPUT_PATH):
    trainedModelName = str(trainedModel).replace("()","")
    trainedModelPath = "".join([OUTPUT_PATH, trainedModelName, ".pkl"])
    with open(trainedModelPath, 'wb') as pklFile:
        pkl.dump(trainedModel, pklFile)

def loadTrainedModel(trainedModel, OUTPUT_PATH):
    trainedModelName = str(trainedModel).replace("()","")
    trainedModelPath = "".join([OUTPUT_PATH, trainedModelName, ".pkl"])
    trainedModelObj = pd.read_pickle(trainedModelPath)
    return trainedModelObj

def modelTraining(model, X_train, y_train, OUTPUT_PATH):
    trainedModel = model.fit(X_train, y_train)
    saveTrainedModel(trainedModel, OUTPUT_PATH)

def createMetricImgs(X_train, X_test, y_train, y_test, labels, model, visualizer, full_filepath, filename):
    metricsViz = Visualizer(X_train, X_test, y_train, y_test, labels, model, visualizer, full_filepath,filename)
    metricsViz.evaluate()
    metricsViz.save_img()


def main():
    filename = 'heart_failure_clinical_records_dataset.csv'
    X_train, X_test, y_train, y_test, labels = processInputDataset(INPUT_PATH + filename, 'DEATH_EVENT')
    randomForestModel = MODELS[0]
    modelTraining(randomForestModel, X_train, y_train, OUTPUT_PATH) 

    #Call metrics function
    outputTrainedModel = loadTrainedModel(randomForestModel, OUTPUT_PATH)
    outputFilePath = 'Data/Output/RandomForestClassifier.pkl'
    for visualizer in VISUALIZERS:
        createMetricImgs(X_train, X_test, y_train, y_test, labels, outputTrainedModel, visualizer,outputFilePath, filename )



if __name__ == '__main__':
    main()
