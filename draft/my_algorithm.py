import tweepy
import re
from textblob import TextBlob

consumer_key = "gHbinS2s5HGxvbgbTJmkwaGCw"
consumer_key_secret = "iYiC4JnVgNVM7pYq6HdCfR8Ge3le3f2OYLDMShhQgt5gNeJTsP"

access_token = "1967161124-BUn8QBrnal07dBaRE48V2x3Y8Co3SaHscSmHS2s"
access_token_secret = "Vbwkd9lLIls1nClxpzPn4xNznedQcStvLaFjgl2orzJAo"

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Now we can create tweets, delete tweets, find twitter users

public_tweets = api.search("Obama")


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


for tweet in public_tweets:
    print("====================================")
    print(tweet.text)
    print("----")
    print(clean_tweet(tweet.text))
    print("----")
    print(tweet.user.screen_name)
    print("+++")
    analysis1 = TextBlob(tweet.text)
    print(analysis1.sentiment)
    print("+++")
    analysis2 = TextBlob(clean_tweet(tweet.text))
    print(analysis2.sentiment)

