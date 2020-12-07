from utils.load_data import load_data
from config import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle
import shap



#X_train, y_train, X_test, y_test, labels, features, target = load_data('../data/input_data/Ecoregion_22_herb.csv')
# Load and split data
data = load_iris()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.3, random_state=4)

# Create a model
model = RandomForestClassifier(random_state=101)
iris_model = model.fit(Xtrain, Ytrain)
print(iris_model.score(Xtest, Ytest))
pkl_filename = 'iris.pkl'

with open(OUTPUT_DATA_FILEPATH + pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# Load from file
with open(OUTPUT_DATA_FILEPATH + pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
    
# Calculate the accuracy score and predict target values
score = pickle_model.score(Xtest, Ytest)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(Xtest)

#Shap function
shap_values = shap.TreeExplainer(pickle_model).shap_values(Xtrain)
