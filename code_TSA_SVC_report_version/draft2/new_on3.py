import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import seaborn as sns
import nltk




data = pd.read_csv('../dataset/downloadedB.csv', delimiter='\t', header=None, names=["id1", "id2", "sentiment", "tweet"])
data = data.drop(data[(data['tweet'] == 'Not Available')].index)
data = data.sample(frac=1).reset_index(drop=True)
print(data.head())

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

# remove URLs
data['tidy_tweet'] = data['tweet'].apply(lambda x: re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', x))

# remove twitter handles (@user)
data['tidy_tweet'] = np.vectorize(remove_pattern)(data['tidy_tweet'], "@[\w]*")
print(data['tidy_tweet'].head())

# remove the # in #hashtag
data['tidy_tweet'] = data['tidy_tweet'].apply(lambda x: re.sub(r'#([^\s]+)', '', x))

# remove special characters, numbers, punctuations
data['tidy_tweet'] = data['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
print(data['tidy_tweet'].head())


#  Removing Short Words
data['tidy_tweet'] = data['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
print(data['tidy_tweet'].head())


# Tokenized tweets
tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())
print(data['tidy_tweet'].head())

# Remove stem word
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
print(data['tidy_tweet'].head())

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

data['tidy_tweet'] = tokenized_tweet
print(data['tidy_tweet'].head())


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(data['tidy_tweet'])

features_name = bow_vectorizer.get_feature_names()      # 1D numpy array ['and', 'document', 'first', 'is', , 'this']

dataset = bow.toarray()

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data.loc[(data.sentiment == 'positive'), 'sentiment'] = 1
data.loc[(data.sentiment == 'neutral'), 'sentiment'] = 0
data.loc[(data.sentiment == 'negative'), 'sentiment'] = -1

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(dataset, data["sentiment"], random_state=42, test_size=0.25)

clf = SVC(C=3, kernel='linear', degree=3)

clf.fit(xtrain_bow, ytrain)

pred = clf.predict(xvalid_bow)

acc = accuracy_score(yvalid, pred)
print(acc)
for i in range(100):
    print(yvalid.values[i], pred[i])



# lreg = LogisticRegression()
# lreg.fit(xtrain_bow, ytrain) # training the model
#
# prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
# prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
# prediction_int = prediction_int.astype(np.int)
#
# print(f1_score(yvalid, prediction_int)) # calculating f1 score





