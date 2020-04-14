import pandas as pd
from sklearn import svm
from utils import PreProcessing
from sklearn.model_selection import GridSearchCV

# Preprocessing
path_file = "../dataset/downloadedB.csv"
# max_features = [20, 50, 100, 250, 500, 750, 1000, 1500, 2000]
# parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
#      {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

max_features = [100, 1000]
parameters = [{'kernel': ['rbf'], 'gamma': [1e-4], 'C': [100, 1000]}, {'kernel': ['linear'], 'C': [1, 10]}]

for max_feature in max_features:
    preprocessing = PreProcessing(max_feature)
    raw_dataset = preprocessing._read_file__(path_file)
    dataset, features_names = preprocessing._fit__(raw_dataset, col_names='tweet')

    #parameters = {'kernel': ('linear', 'rbf'), 'C': [ 10, 20]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(dataset, raw_dataset["sentiment"])
    df = pd.DataFrame(clf.cv_results_, columns=clf.cv_results_.keys())
    df.to_csv("results/normal/svm_" + str(max_feature) + ".csv", index=None, header=True)


