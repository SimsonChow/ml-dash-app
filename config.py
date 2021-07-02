from sklearn.ensemble import RandomForestClassifier

MODELS = [RandomForestClassifier()]
VISUALIZERS = ['ClassificationReport','ConfusionMatrix', 'ROCAUC', 'PrecisionRecallCurve']

INPUT_PATH = "Data/Input/"
OUTPUT_PATH = "Data/Output/"
