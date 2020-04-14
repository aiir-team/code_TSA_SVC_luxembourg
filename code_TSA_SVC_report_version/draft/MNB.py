from sklearn.naive_bayes import MultinomialNB
#from prepare_data import prepareData
from draft.preprocessing import prepare_dataset
from sklearn.metrics import accuracy_score


def mnb():
    #train_X, train_Y, test_X, test_Y = prepareData(word_frequency=1000)
    train_X, train_Y, test_X, test_Y = prepare_dataset(word_frequency=1000)

    clf = MultinomialNB()
    clf.fit(train_X, train_Y)

    pred = clf.predict(test_X)

    acc = accuracy_score(test_Y, pred)
    print(acc)

    for i in range(100):
        print(test_Y[i], list(pred)[i])


# def grid_search():
#     mnb()
#
# grid_search()

mnb()