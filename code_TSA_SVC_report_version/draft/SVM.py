from sklearn.svm import SVC
from draft.prepare_data import prepareData
from sklearn.metrics import accuracy_score
from statistics import mean

def svm(word_frequency, C, kernel, poly_degree):
    train_X, train_Y, test_X, test_Y = prepareData(word_frequency)

    clf = SVC(C=C, kernel=kernel, degree=poly_degree)

    clf.fit(train_X, train_Y)

    pred = clf.predict(test_X)

    acc = accuracy_score(test_Y, pred)
    '''
    for i in range(100):
        print(test_Y[i], list(pred)[i])
    '''
    return acc

def grid_search_word_frequency():
    word_frequencys = [250, 500, 750, 1000, 1250, 1500]
    score = []
    for word_frequency in word_frequencys:
        accuracys = []
        for i in range(3):
            accuracy = svm(word_frequency, 3, 'rbf', 3)
            accuracys.append(accuracy)
        mean_accuracy = mean(accuracys)
        score.append([word_frequency, mean_accuracy])

    print(score)

def grid_search_hyperparameters():
    Cs = [3, 4, 5, 6, 7, 8, 9, 10]
    kernels = ['linear', 'rbf']
    poly_degrees = [2, 3, 4, 5, 6]
    score = []
    for C in Cs:
        for kernel in kernels:
            if kernel == 'poly':
                for poly_degree in poly_degrees:
                    print([C, kernel, poly_degree])
                    accuracy = svm(word_frequency=1000, C=C, kernel=kernel, poly_degree=poly_degree)
                    score.append([accuracy, C, kernel, poly_degree])
            else:
                print([C, kernel, None])
                accuracy = svm(word_frequency=1000, C=C, kernel=kernel, poly_degree=3)
                score.append([accuracy, C, kernel, None])

    print(score)


#grid_search_word_frequency()

svm(word_frequency=1250, C=3, kernel='rbf', poly_degree=3)
