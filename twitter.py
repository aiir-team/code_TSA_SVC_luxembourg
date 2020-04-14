import os
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from app_settings import *

class TwitterObject(object):
    def __init__(self, query=None):
        # keys and tokens from the Twitter Dev Console
        consumer_key = CONSUMER_KEY
        consumer_secret = CONSUMER_KEY_SECRET
        access_token = ACCESS_TOKEN
        access_token_secret = ACCESS_TOKEN_SECRET
        # Attempt authentication
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.retweets_only = RETWEETS_ONLY
            self.with_sentiment = WITH_SENTIMENT
            self.query = query
            self.api = tweepy.API(self.auth)
            self.tweet_count_max = MAX_NUMBER_TWEETS  # To prevent Rate Limiting
        except:
            print("Error: Authentication Failed")

    def set_query(self, query=''):
        self.query = query

    def clean_tweet(self, tweet):
        # tweet = tweet.lower()
        # print(tweet)
        #
        # # remove URLs
        # tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', tweet)
        # tweet = re.sub(r'http\S+', '', tweet)
        # print(tweet)
        #
        # # remove usernames
        # tweet = re.sub('@[^\s]+', '', tweet)
        # # remove the # in #hashtag
        # tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        # print(tweet)
        #
        # tweet = word_tokenize(tweet)
        # print(tweet)

        # new_words = [re.sub(r'[^\w\s]', '', word) for word in tweet]
        # tweet = [word for word in new_words if word != '']
        # print(tweet)

        # remove stopwords from final word list
        # tweet = [word for word in tweet if word not in stopwords.words('english') and len(word) > 1]

        # print('---remove numbers---')
        # tweet = [word for word in tweet if not word.isdigit()]

        # print('---stem words---')
        # stemmer = PorterStemmer()
        # tweet = [stemmer.stem(w) for w in tweet]
        # print(tweet)
        #
        # return ' '.join(tweet)

        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return "positive"
        elif analysis.sentiment.polarity == 0:
            return "neutral"
        else:
            return "negative"

    def get_tweets(self, since_date = '2012-06-06'):
        tweets = []

        try:
            query = "#" + self.query + " -filter:retweets"
            recd_tweets = self.api.search(q=query, lang='en', since=since_date, count=self.tweet_count_max)
            if not recd_tweets:
                pass
            for tweet in recd_tweets:
                parsed_tweet = {}

                parsed_tweet["text"] = tweet.text
                parsed_tweet["user"] = tweet.user.screen_name

                if self.with_sentiment == 1:
                    parsed_tweet["sentiment"] = self.get_tweet_sentiment(tweet.text)
                else:
                    parsed_tweet["sentiment"] = "unavailable"
                tweets.append(parsed_tweet)
            return tweets

        except tweepy.TweepError as e:
            print("Error : " + str(e))