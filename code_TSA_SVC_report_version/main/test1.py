from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from utils import PreProcessing, HandlingIO

# Preprocessing
handle_io = HandlingIO()
pathsave = "results/test1/"
path_file = "dataset/downloadedB.csv"
max_features = [20, 50, 100, 250, 500, 750, 1000, 1500, 2000]
test_sizes = [0.2, 0.3, 0.4, 0.5]
kernels = ['rbf', 'linear']

results_table = []
for max_feature in max_features:
    preprocessing = PreProcessing(max_feature)
    raw_dataset = preprocessing._read_file__(path_file)
    dataset, features_names = preprocessing._fit__(raw_dataset, col_names='tweet')

    for test_size in test_sizes:
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(dataset, raw_dataset["sentiment"], test_size=test_size, random_state=0)
        for kernel in kernels:
            if kernel == "rbf":
                tuned_parameters = [{'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 5, 10, 20, 50, 100, 200, 500, 1000]}]
            else:
                tuned_parameters = [{'C': [1, 5, 10, 20, 50, 100, 200, 500, 1000]}]

            clf = GridSearchCV(SVC(), tuned_parameters)
            clf.fit(X_train, y_train)
            y_true, y_pred = y_test, clf.best_estimator_.predict(X_test)

            ## Results in training process
            df = pd.DataFrame(clf.cv_results_, columns=clf.cv_results_.keys())
            df.to_csv(pathsave + "svm_" + str(max_feature) + "_" + str(test_size*10) + "_" + kernel + ".csv", index=None, header=True)

            ## Get accuracy, save results and save model
            acc_true = accuracy_score(y_true, y_pred)
            model_name = "svm_" + str(max_feature) + "_" + str(test_size*10) + "_" + kernel
            final_result = [model_name, acc_true]
            handle_io._save_all_results_to_csv__(final_result, "test1_final", pathsave)
            handle_io._save_model__(clf, model_name, pathsave)