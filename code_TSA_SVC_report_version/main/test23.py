from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from numpy import nan
from sklearn.metrics import accuracy_score
import pandas as pd
from utils import PreProcessing2ClassesDataset2, HandlingIO


# Preprocessing
handle_io = HandlingIO()
pathsave = "results/test23/"
train_file = "dataset/train_E6oV3lV.csv"
test_file = "dataset/test_tweets_anuFYb8.csv"

max_features = [10, 50, 100, 250, 500, 1000, 1500, 2000]
kernels = ['rbf', 'linear']

for max_feature in max_features:
    preprocessing = PreProcessing2ClassesDataset2(max_feature)
    train_dataframe = preprocessing._read_file__(train_file)
    test_dataframe = preprocessing._read_file__(test_file)
    train_dataframe = train_dataframe.replace(to_replace='None', value=nan).dropna()
    raw_dataset = train_dataframe.append(test_dataframe, ignore_index=True, sort=False)
    dataset, features_names = preprocessing._fit__(raw_dataset, col_names='tweet')
    X_training_set = dataset[:len(train_dataframe), :]
    X_testing_set = dataset[len(train_dataframe):, :]

    for kernel in kernels:
        if kernel == "rbf":
            tuned_parameters = [{'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 500, 1000]}]
        else:
            tuned_parameters = [{'C': [1, 10, 50, 100, 200, 500, 1000]}]

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5)
        t1 = len(X_training_set)
        t2 = raw_dataset["label"][:len(X_training_set)].values
        clf.fit(X_training_set, t2)
        y_pred = clf.best_estimator_.predict(X_testing_set)

        ## Results in training process
        df = pd.DataFrame(clf.cv_results_, columns=clf.cv_results_.keys())
        df.to_csv(pathsave + "svm_" + str(max_feature) + "_" + kernel + ".csv", index=None, header=True)

        ## Get accuracy, save results and save model
        model_name = "svm_" + str(max_feature) + "_" + kernel
        handle_io._save_model__(clf, model_name, pathsave)

        test_dataframe['label'] = y_pred
        submission = test_dataframe[['id', 'label']]
        submission.to_csv(pathsave + "submission_svm_" + str(max_feature) + "_" + kernel + ".csv", index=False)  # writing data to a CSV file


