
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
from sklearn.model_selection import train_test_split
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
        df.iloc[index]['tweet_content'] = _handle_raw_data__(df.iloc[index]['tweet_content'])

    all_words = ' '.join([text for text in df['tweet_content']])

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('word_cloud.png', bbox_inches='tight')
    plt.show()


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

prepare_dataset(2000)



