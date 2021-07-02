import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.model_selection import train_test_split
from config import *

def processInputDataset(dataset, targetVariable):
    dataFrame = pd.read_csv(dataset)
    X = dataFrame.drop(targetVariable, axis = 1)
    y = dataFrame[targetVariable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    return X_train, X_test, y_train, y_test

def saveTrainedModel(trainedModel, OUTPUT_PATH):
    trainedModelName = str(trainedModel).replace("()","")
    trainedModelPath = "".join([OUTPUT_PATH, trainedModelName, ".pkl"])
    with open(trainedModelPath, 'wb') as pklFile:
        pkl.dump(trainedModel, pklFile)
    
def modelTraining(model, X_train, y_train, OUTPUT_PATH):
    trainedModel = model.fit(X_train, y_train)
    saveTrainedModel(trainedModel, OUTPUT_PATH)
    
def main():
    X_train, X_test, y_train, y_test = processInputDataset(INPUT_PATH + 'heart_failure_clinical_records_dataset.csv', 'DEATH_EVENT')
    randomForestModel = MODELS[0]
    modelTraining(randomForestModel, X_train, y_train, OUTPUT_PATH)



if __name__ == '__main__':
    main()