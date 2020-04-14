import os

APP_CONFIG_KEY = "15e3c01b2f4393cc0cbebaefe3559c12"

CONSUMER_KEY = "gHbinS2s5HGxvbgbTJmkwaGCw"
CONSUMER_KEY_SECRET = "iYiC4JnVgNVM7pYq6HdCfR8Ge3le3f2OYLDMShhQgt5gNeJTsP"

ACCESS_TOKEN = "1967161124-BUn8QBrnal07dBaRE48V2x3Y8Co3SaHscSmHS2s"
ACCESS_TOKEN_SECRET = "Vbwkd9lLIls1nClxpzPn4xNznedQcStvLaFjgl2orzJAo"

PATH_TO_ORIGINAL_DATA = "database/downloadedB.csv"
PATH_TO_FORMATTED_DATA_3C = "database/formatted_dataset_3classes.csv"
PATH_TO_FORMATTED_DATA_2C = "database/formatted_dataset_2classes.csv"

MAX_NUMBER_TWEETS = 100     # default = 15, maximum = 100 by the api
RETWEETS_ONLY = False
WITH_SENTIMENT = True

IMAGES_FOLDER = os.path.join('static', 'images')
