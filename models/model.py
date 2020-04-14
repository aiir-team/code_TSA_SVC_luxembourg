from numpy import array, round
from string import punctuation
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv
import pickle


def _save_model__(model=None, filename=None, pathsave=""):
    # save the model to disk
    pickle.dump(model, open(pathsave + filename + ".pkl", 'wb'))
    return 0

def _load_model__(filename=None, pathsave=""):
    # load the model from disk
    return pickle.load(open(pathsave + filename + ".pkl", 'rb'))


# load vocab into memory
def load_vocab(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load doc into memory
def load_doc(filename, classes=2):
    # read all text
    data = read_csv(filename, delimiter='\t', header=None, names=["id1", "id2", "sentiment", "tweet"])
    data = data.drop(data[(data['tweet'] == 'Not Available')].index)
    if classes == 2:
        data = data.drop(data[(data['sentiment'] == 'neutral')].index)
    raw_data = data.sample(frac=1).reset_index(drop=True)
    return raw_data


# turn a sentence into clean tokens
def clean_doc(doc):
    doc = doc.lower()
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 2]
    return tokens


# load all tweets
def process_tweets(tweets, vocab):
    lines = list()
    # walk through all files in the folder
    for tweet in tweets:
        # clean tweet
        tokens = clean_doc(tweet)
        # filter by vocab
        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        # add to list
        lines.append(line)
    return lines

#
# # load the vocabulary
# vocab_filename = '../models/vocab.txt'
# vocab = load_vocab(vocab_filename)
# vocab = vocab.split()
# vocab = set(vocab)
# test_ratio = 0.1
#
# # load all training and testing
# raw_tweets = load_doc('../database/downloadedB.csv', classes=2)
# length_train = int((1 - test_ratio) * len(raw_tweets))
# train_raw = raw_tweets["tweet"][:length_train]
# test_raw = raw_tweets["tweet"][length_train:]
# train_formatted = process_tweets(train_raw, vocab)
#
# # summarize what we have
# print(len(train_formatted))
# print(train_formatted)
#
# # create the tokenizer
# tokenizer = Tokenizer()
# # fit the tokenizer on the documents
# tokenizer.fit_on_texts(train_formatted)
#
# # encode training data set
# Xtrain = tokenizer.texts_to_matrix(train_formatted, mode='freq')
# print(Xtrain.shape)
#
# # load all test tweets
# test_formatted = process_tweets(test_raw, vocab)
# Xtest = tokenizer.texts_to_matrix(test_formatted, mode='freq')
# print(Xtest.shape)
#
# # The model will have an input layer that equals the number of words in the vocabulary
# n_words = Xtest.shape[1]
#
# ytrain = array([0 if raw_tweets["sentiment"][_] == "negative" else 1 for _ in range(0, length_train)])
# ytest = array([0 if raw_tweets["sentiment"][_] == "negative" else 1 for _ in range(length_train, len(raw_tweets))])
#
# # define network
# model = Sequential()
# model.add(Dense(25, input_shape=(n_words,), activation='elu'))
# model.add(Dense(1, activation='sigmoid'))
# # compile network
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # fit network
# model.fit(Xtrain, ytrain, epochs=100, verbose=2)
#
# # evaluate
# loss, acc = model.evaluate(Xtest, ytest, verbose=0)
# print('Test Accuracy: %f' % (acc * 100))
#
# # Save model for next time usage
# _save_model__(model, "model", "")
# _save_model__(tokenizer, "tokenizer", "")


# classify a review as negative (0) or positive (1)
def predict_sentiment(tweet, vocab, tokenizer, model):
    # clean
    tokens = clean_doc(tweet)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='freq')
    # prediction
    yhat = model.predict(encoded, verbose=0)
    return round(yhat[0, 0])


# classify a review as negative (0) or positive (1)
def predict_sentiments(tweets, vocab, tokenizer, model):
    lines = list()
    for tweet in tweets:

        # clean
        tokens = clean_doc(tweet)
        # filter by vocab
        tokens = [w for w in tokens if w in vocab]
        # convert to line
        line = ' '.join(tokens)
        lines.append(line)
    # encode
    encoded = tokenizer.texts_to_matrix(lines, mode='freq')
    # prediction
    yhat = model.predict(encoded, verbose=0)
    return round(yhat)


# # test positive text
# text = 'Best movie ever!'
# print(predict_sentiment(text, vocab, tokenizer, model))
# # test negative text
# text = 'This is a bad movie.'
# print(predict_sentiment(text, vocab, tokenizer, model))
#
#
