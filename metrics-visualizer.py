from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ConfusionMatrix
from config import *
import utils_xai.helpers 

class Visualizer():
    def __init__(self, X_train, X_test, y_train, y_test, labels, model, viz_selection, full_filepath, filename):
        """
        Class for yellowbrick classifier visualizer
        Args:
            X_train: numpy ndarray of model features training data values
            X_test: numpy ndarray of model features test data values
            y_train: numpy ndarray of model target variable training data values
            y_test: numpy ndarray of model target variable test data values
            labels: list of class labels for binary classification
            model: sklearn estimator for classification
            viz_selection: string value used to reference yellowbrick classification visualizer
            filename: string value used to store metrics to correspondent filename
        """

        self.labels = labels
        self.model = model
        self.viz_selection = viz_selection
        self.filename = filename
        self.metric_filepath = "/".join([full_filepath,"Metrics/"])
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        
        # We need to check whether we are training a multiclass or a binary class model. If it is a binary classification model, 
        # We need to use encode the test set with values of 1's and 0's for yellowbrick Precision Recall, Classification Report and 
        # ROC AUC python package.   
        self.isBinaryClassification = True if len(self.labels) == 2 else False
        if self.isBinaryClassification:
            labelencoder = LabelEncoder()
            self.y_test_decode = labelencoder.fit_transform(self.y_test)
        
        if self.viz_selection == 'ClassificationReport':
            self.visualizer = ClassificationReport(self.model, support=True)
        elif self.viz_selection == 'ROCAUC':
            self.visualizer = ROCAUC(self.model, support=True)
        elif self.viz_selection == 'ConfusionMatrix':
            self.visualizer = ConfusionMatrix(model)
        elif self.viz_selection == 'PrecisionRecallCurve':
            self.visualizer = PrecisionRecallCurve(self.model)
        else:
            return print("Error: viz_selection does not match accepted values. View Visualizer Class for accepted values.")

    def evaluate(self):
        """
        Fit and score model associated with visualizer
        
        """
        self.visualizer.fit(self.X_train, self.y_train)

        if self.isBinaryClassification is True and self.viz_selection in ['ROCAUC',  'ClassificationReport', 'PrecisionRecallCurve']:
            self.visualizer.score(self.X_test, self.y_test_decode)
        else:
            self.visualizer.score(self.X_test, self.y_test)
    
    def save_img(self):
        """
        Save image output of visualizer to output directory
        Returns:
            matplotlib image saved as png
        """
        #Create the filename directory to store all metrics images in there
        utils_xai.helpers.create_dir(self.metric_filepath)

        #Migrate all metrics results images to the created directory
        self.outpath_ = self.metric_filepath + self.viz_selection + '.png'
        return self.visualizer.show(outpath=self.outpath_, clear_figure=True)

