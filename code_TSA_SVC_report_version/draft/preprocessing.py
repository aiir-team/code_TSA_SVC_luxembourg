import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
from sklearn.model_selection import train_test_split
import time
from textblob import TextBlob

def _handle_raw_data__(data):
    # convert text to lower-case
    tweets = data.lower()

    # remove URLs
    tweets = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', tweets)
    #tweets = re.sub(r'http\S+', ' ', tweets)

    # remove usernames
    tweets = re.sub('@[^\s]+', ' ', tweets)
    # remove the # in #hashtag
    tweets = re.sub('#([^\s]+)', ' ', tweets)

    # print('---remove numbers---')
    tweets = re.sub(r'\d+', ' ', tweets)
    # print(len(tweets))
    # print(tweets[:100])

    # remove repeated characters
    # print('---Tokenize---')
    tweets = word_tokenize(tweets)
    # print(len(tweets))
    # print(tweets[:100])

    #Remove punctuation from list of tokenized words"""
    new_words = [ re.sub(r'[^\w\s]', '', word) for word in tweets ]
    tweets = [ word for word in new_words if word != '']

    # remove stopwords from final word list
    # print('---remove Stopwords and short words 1---')
    tweets = [word for word in tweets if word not in stopwords.words('english') and len(word) > 2]
    # print(len(tweets))
    # print(tweets[:100])

    # print('---stem words---')
    stemmer = PorterStemmer()
    tweets = [stemmer.stem(w) for w in tweets]
    # print(len(tweets))
    # print(tweets[:100])

    return ' '.join(tweets)

def create_dataset_and_features__(df, word_frequency):
    ## Make tweet content better
    for index in range(len(df)):
        if index % 100 == 0:
            print(index)
        df.iloc[index]['tweet_content'] = _handle_raw_data__(df.iloc[index]['tweet_content'])

    df.to_csv('formatted_dataset.csv', index=None, header=True)

    ## Create features
    tweets = df['tweet_content'].tolist()
    tweets = ', '.join(tweets)
    # spell correction
    #tweets = TextBlob(tweets).correct().string
    tweets = re.sub("[^\w]", " ", tweets).split()

    ## remove infrequent words
    counted = Counter(tweets)
    words = sorted(counted, key=counted.get)
    features = (words)[max(0, len(words) - word_frequency):]            # remove infrequent words
    print(features)

    ## Create dataset
    for feature in features:
        df[feature] = [0]*len(df)
    for index in range(len(df)):
        words = re.sub("[^\w]", " ", df.iloc[index]['tweet_content']).split()
        for word in words:
            if word in features:
                df.at[index, word] += 1
    df.pop('tweet_content')
    return df


def prepare_dataset(word_frequency):
    t1 = time.time()
    df = pd.read_csv('../dataset/downloadedB.csv', delimiter='\t', encoding="ISO-8859-1", names=['id1', 'id2', 'sentiment', 'tweet_content'])
    df = df.drop(columns=['id1', 'id2'])
    df = df.drop(df[(df['tweet_content'] == 'Not Available')].index)
    # df = df.drop(df[(df['sentiment'] == 'neutral')].index)
    df.loc[(df.sentiment == 'positive'), 'sentiment'] = 1
    df.loc[(df.sentiment == 'neutral'), 'sentiment'] = 0
    df.loc[(df.sentiment == 'negative'), 'sentiment'] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    df = create_dataset_and_features__(df, word_frequency)

    train, test = train_test_split(df, test_size=0.2)

    train_Y = train.pop('sentiment').values.tolist()
    train_X = train.values.tolist()
    test_Y = test.pop('sentiment').values.tolist()
    test_X = test.values.tolist()

    print("Done")
    t2 = time.time() - t1
    print("Time: ", t2)
    return train_X, train_Y, test_X, test_Y


# word_frequencys = [250, 500, 750, 1000, 1250, 1500]
#
# for word_frequency in word_frequencys:

#prepare_dataset(2000)