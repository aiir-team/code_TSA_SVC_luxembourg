from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pickle
from sklearn.model_selection import train_test_split
from utils import PreProcessing

def MySvm(path_file, model_file, max_features, C, kernel, poly_degree, test_size=0.2):

    # Preprocessing
    preprocessing = PreProcessing(max_features)
    raw_dataset = preprocessing._read_file__(path_file)
    dataset, features_names = preprocessing._fit__(raw_dataset, col_names='tweet')

    # splitting data into training and validation set
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(dataset, raw_dataset["sentiment"], random_state=42, test_size=test_size)

    # Building model
    clf = SVC(C=C, kernel=kernel, degree=poly_degree)

    # Training model
    clf.fit(xtrain_bow, ytrain)

    # save the model to disk
    pickle.dump(clf, open(model_file, 'wb'))

    # Predicting
    pred = clf.predict(xvalid_bow)

    acc = accuracy_score(yvalid.values, pred)
    return acc, xvalid_bow, yvalid

acc, x_test, y_test = MySvm("dataset/downloadedB.csv", 'draft/finalized_model.sav', 100, 10, "rbf", 100, 0.2)
print(acc)


# load the model from disk
filename = 'draft/finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)

pred = loaded_model.predict(x_test)
print(pred)


