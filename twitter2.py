import re
import tweepy
from pandas import DataFrame
from tweepy import OAuthHandler
from keras.preprocessing.text import Tokenizer
from textblob import TextBlob
from models.model import load_vocab, process_tweets, _load_model__, predict_sentiments
from app_settings import *

class TwitterObject2(object):
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
            self.query = query
            self.api = tweepy.API(self.auth)        # , wait_on_rate_limit=True
            self.tweet_count_max = 100  # To prevent Rate Limiting
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", tweet).split())

    def get_tweets(self, since_date='2012-06-06'):
        try:
            query = "#" + self.query + " -filter:retweets"
            tweets = tweepy.Cursor(self.api.search, q=query, lang='en', since=since_date).items(self.tweet_count_max)
            if not tweets:
                pass

            # Remove URLs and create textblob object for each tweet
            tweets_no_urls_for_textblob = [self.clean_tweet(tweet.text) for tweet in tweets]

            sentiment_objects  = [TextBlob(tweet) for tweet in tweets_no_urls_for_textblob]

            # Create list of polarity values and tweet text
            sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]

            # Create dataframe containing the polarity value and tweet text
            sentiment_df = DataFrame(sentiment_values, columns=["polarity", "tweet"])

            # To get a better visual of the polarit values, it can be helpful to remove the polarity values equal to
            # zero and create a break in the histogram at zero.

            # Remove polarity values equal to zero
            sentiment_df = sentiment_df[sentiment_df.polarity != 0]
            sentiment_df.loc[sentiment_df['polarity'] < 0, 'sentiment_textblob'] = "negative"
            sentiment_df.loc[sentiment_df['polarity'] > 0, 'sentiment_textblob'] = "positive"
            sentiment_df.reset_index(drop=True)

            sentiment_df = sentiment_df.drop(sentiment_df[(sentiment_df['polarity'] == 0)].index)
            # # df = df.drop(df[(df['sentiment'] == 'neutral')].index)
            # df.loc[(df.sentiment == 'positive'), 'sentiment'] = 1
            # df.loc[(df.sentiment == 'neutral'), 'sentiment'] = 0
            # df.loc[(df.sentiment == 'negative'), 'sentiment'] = -1
            #
            # print(df)
            sentiment_df = sentiment_df.sample(frac=1).reset_index(drop=True)

            if len(sentiment_df["tweet"]) != 0:
                ## Load vocab
                # load the vocabulary
                vocab = load_vocab('models/vocab.txt')
                vocab = set(vocab.split())
                print(sentiment_df["tweet"])
                # encode testing data set
                test_formatted = process_tweets([tweet for tweet in sentiment_df["tweet"]], vocab)

                # load the tokenizer
                tokenizer = _load_model__("tokenizer", "models/")
                model = _load_model__("model", "models/")

                senti_predict = predict_sentiments(test_formatted, vocab, tokenizer, model)
                sentiment_df["sentiment_model"] = DataFrame(senti_predict)
                sentiment_df.loc[sentiment_df['sentiment_model'] == 0, 'sentiment_model'] = "negative"
                sentiment_df.loc[sentiment_df['sentiment_model'] == 1, 'sentiment_model'] = "positive"
                print(sentiment_df["sentiment_model"])
                return sentiment_df
            else:
                return None

        except tweepy.TweepError as e:
            print("Error : " + str(e))


class TwitterObject3(TwitterObject2):
    def __init__(self, query=None):
        TwitterObject2.__init__(self, query)

    def get_tweets(self, since_date='2012-06-06'):
        try:
            query = "#" + self.query + " -filter:retweets"
            tweets = self.api.search(q=query, lang='en', since=since_date, count=self.tweet_count_max)
            if not tweets:
                pass

            # Remove URLs and create textblob object for each tweet
            tweets_sentiments = [TextBlob(self.clean_tweet(tweet.text)) for tweet in tweets]

            # Create list of polarity values and tweet text
            sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in tweets_sentiments]

            # Create dataframe containing the polarity value and tweet text
            sentiment_df = DataFrame(sentiment_values, columns=["polarity", "tweet"])
            sentiment_df.loc[sentiment_df['polarity'] == 0, 'sentiment'] = "neutral"
            sentiment_df.loc[sentiment_df['polarity'] < 0, 'sentiment'] = "negative"
            sentiment_df.loc[sentiment_df['polarity'] > 0, 'sentiment'] = "positive"

            return sentiment_df

        except tweepy.TweepError as e:
            print("Error : " + str(e))