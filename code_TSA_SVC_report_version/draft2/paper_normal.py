from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from utils import PreProcessing

# Preprocessing
path_file = "../dataset/downloadedB.csv"
#max_features = [20, 50, 100, 250, 500, 750, 1000, 1500, 2000]
#test_sizes = [0.2, 0.3, 0.4, 0.5]
#scores = ['precision', 'recall']

# Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

max_features = [100, 1000]
test_sizes = [0.4, 0.5]
scores = ['recall']
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]

for max_feature in max_features:
    preprocessing = PreProcessing(max_feature)
    raw_dataset = preprocessing._read_file__(path_file)
    dataset, features_names = preprocessing._fit__(raw_dataset, col_names='tweet')

    for test_size in test_sizes:
        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(dataset, raw_dataset["sentiment"], test_size=test_size, random_state=0)

        print("===============================================")
        clf = GridSearchCV(SVC(C=1), tuned_parameters)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()