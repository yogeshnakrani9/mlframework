from sklearn import ensemble 
from xgboost import XGBClassifier

MODELS = {
    "randomforest" : ensemble.RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 2),
    "extratrees" : ensemble.ExtraTreesClassifier(n_estimators = 200, n_jobs = -1, verbose = 2),
    "xgboost" : XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
}