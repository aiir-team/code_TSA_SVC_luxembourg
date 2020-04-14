import re
import csv
import pickle
from numpy import vectorize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from pandas import read_csv

class PreProcessing:

    def __init__(self, max_features=100):
        self.max_features = max_features

    def _read_file__(self, pathfile=None, delimiter=None, header=None, names=None):
        data = read_csv(pathfile, delimiter='\t', header=None, names=["id1", "id2", "sentiment", "tweet"])
        data = data.drop(data[(data['tweet'] == 'Not Available')].index)
        # data.loc[(data.sentiment == 'positive'), 'sentiment'] = 2
        # data.loc[(data.sentiment == 'neutral'), 'sentiment'] = 1
        # data.loc[(data.sentiment == 'negative'), 'sentiment'] = 0
        raw_data = data.sample(frac=1).reset_index(drop=True)
        return raw_data

    def __remove_pattern__(self, input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    def _fit__(self, dataset, col_names):
        dataset = dataset[[col_names]].copy()
        # remove URLs
        dataset[col_names] = dataset[col_names].apply(lambda x: re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', x))

        # remove twitter handles (@user)
        dataset[col_names] = vectorize(self.__remove_pattern__)(dataset[col_names], "@[\w]*")
        #print(dataset.head())

        # remove the # in #hashtag
        dataset[col_names] = dataset[col_names].apply(lambda x: re.sub(r'#([^\s]+)', '', x))

        # remove special characters, numbers, punctuations
        dataset[col_names] = dataset[col_names].str.replace("[^a-zA-Z#]", " ")
        #print(dataset.head())

        #  Removing Short Words
        dataset[col_names] = dataset[col_names].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
        #print(dataset.head())

        # Tokenized tweets
        tokenized_tweet = dataset[col_names].apply(lambda x: x.split())
        #print(dataset.head())

        # Remove stem word
        stemmer = PorterStemmer()
        tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])  # stemming
        #print(dataset.head())

        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

        # bag-of-words feature matrix
        bow_vectorizer = CountVectorizer(max_features=self.max_features, stop_words='english')
        bow = bow_vectorizer.fit_transform(tokenized_tweet)
        features_names = bow_vectorizer.get_feature_names()  # 1D numpy array ['and', 'document', 'first', 'is', , 'this']
        dataset = bow.toarray()

        return dataset, features_names


class PreProcessing2Classes(PreProcessing):

    def __init__(self, max_features=100):
        PreProcessing.__init__(self, max_features)

    def _read_file__(self, pathfile=None, delimiter=None, header=None, names=None):
        data = read_csv(pathfile, delimiter='\t', header=None, names=["id1", "id2", "sentiment", "tweet"])
        data = data.drop(data[(data['tweet'] == 'Not Available')].index)
        data = data.drop(data[(data['sentiment'] == 'neutral')].index)
        # data.loc[(data.sentiment == 'positive'), 'sentiment'] = 2
        # data.loc[(data.sentiment == 'neutral'), 'sentiment'] = 1
        # data.loc[(data.sentiment == 'negative'), 'sentiment'] = 0
        raw_data = data.sample(frac=1).reset_index(drop=True)
        return raw_data


class PreProcessing2ClassesDataset2(PreProcessing):

    def __init__(self, max_features=100):
        PreProcessing.__init__(self, max_features)

    def _read_file__(self, pathfile=None, delimiter=None, header=None, names=None):
        return read_csv(pathfile)


class HandlingIO:

    def _save_all_results_to_csv__(self, item=None, log_filename=None, pathsave=None):
        with open(pathsave + log_filename + ".csv", "a+") as file:
            wr = csv.writer(file, dialect='excel')
            wr.writerow(item)

    def _save_model__(self, model=None, filename=None, pathsave=None):
        # save the model to disk
        pickle.dump(model, open(pathsave + filename + ".sav", 'wb'))
        return 0

    def _load_model__(self, model=None, filename=None, pathsave=None):
        # load the model from disk
        return pickle.load(open(pathsave + filename + ".sav", 'rb'))

