from config import *
from ml_utils_xai import *
from metrics_visualizer import Visualizer


def main():
    filename = 'heart_failure_clinical_records_dataset.csv'
    X_train, X_test, y_train, y_test, labels = processInputDataset(INPUT_PATH + filename, 'DEATH_EVENT')

    saveDataFrame(X_test,OUTPUT_PATH + 'test.csv')
    randomForestModel = MODELS[0]
    modelName = str(randomForestModel).replace("()","")

    trainedModel = modelTraining(randomForestModel, X_train, y_train, OUTPUT_PATH) 
    saveObject(trainedModel, modelName,OUTPUT_PATH)
    outputTrainedModel = loadObject(modelName, OUTPUT_PATH)

    # Create a SHAP dataframe for each label (Death and Benign) with positive/negative correlation
    for label in labels:
        shapObject = shapTreeExplainerObject(randomForestModel, int(label), X_train)
        saveObject(shapObject, modelName + "_SHAP_LABEL_{}".format(label), OUTPUT_PATH)
    
    for visualizer in VISUALIZERS:
        createMetricImgs(X_train, X_test, y_train, y_test, labels, outputTrainedModel, visualizer, OUTPUT_PATH, filename)

if __name__ == '__main__':
    main()
