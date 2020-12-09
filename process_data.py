from utils.load_data import load_data
from utils.helpers import load_model, train_model
from config import *
from sklearn.model_selection import train_test_split
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-filename", "--filename", dest="filename", help="Please input filename", default="", required=True)
args = parser.parse_args()

filename = args.filename 

#Load data
X_train, X_test, Y_train, Y_test, labels, features, target = load_data(INPUT_DATA_FILEPATH + filename)

#Train Model
model = train_model(MODELS[0], X_train, Y_train)

#Load trained model to Output directory as a pkl file
trained_model = load_model(model, OUTPUT_DATA_FILEPATH + filename)

