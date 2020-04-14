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

# remove twitter handles (@user)
data['tidy_tweet'] = np.vectorize(remove_pattern)(data['tweet'], "@[\w]*")
print(data['tidy_tweet'].head())

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

# Word Cloud
all_words = ' '.join([text for text in data['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()



# Words in non racist/sexist tweets
normal_words = ' '.join([text for text in data['tidy_tweet'][data['sentiment'] == 'neutral']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Words in non racist/sexist tweets
negative_words = ' '.join([text for text in data['tidy_tweet'][data['sentiment'] == 'negative']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Racist/Sexist Tweets
postive_words = ' '.join([text for text in data['tidy_tweet'][data['sentiment'] == 'positive']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(postive_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


# extracting hashtags from non racist/sexist tweets
HT_normal = hashtag_extract(data['tidy_tweet'][data['sentiment'] == 'neutral'])

# extracting hashtags from non racist/sexist tweets
HT_postive = hashtag_extract(data['tidy_tweet'][data['sentiment'] == 'positive'])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(data['tidy_tweet'][data['sentiment'] == 'negative'])


# unnesting list
HT_normal = sum(HT_normal,[])
HT_postive = sum(HT_postive,[])
HT_negative = sum(HT_negative,[])


# Normal Tweets
ht_normal = nltk.FreqDist(HT_normal)
ht_no = pd.DataFrame({'Hashtag': list(ht_normal.keys()), 'Count': list(ht_normal.values())})
# selecting top 10 most frequent hashtags
ht_no = ht_no.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=ht_no, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()



# Positive Tweets
ht_positive = nltk.FreqDist(HT_postive)
ht_po = pd.DataFrame({'Hashtag': list(ht_positive.keys()), 'Count': list(ht_positive.values())})
# selecting top 10 most frequent hashtags
ht_po = ht_po.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=ht_po, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# Negative Tweets
ht_negative = nltk.FreqDist(HT_negative)
ht_ne = pd.DataFrame({'Hashtag': list(ht_negative.keys()), 'Count': list(ht_negative.values())})
# selecting top 10 most frequent hashtags
ht_ne = ht_ne.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=ht_ne, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()





from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(data['tidy_tweet'])



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# train_test_len = int(0.7 * len(bow))
# train_bow = bow[:train_test_len,:]
# test_bow = bow[train_test_len:,:]


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(bow, data["sentiment"], random_state=42, test_size=0.25)

clf = SVC(C=3, kernel='linear', degree=3)

clf.fit(xtrain_bow, ytrain)

pred = clf.predict(xvalid_bow)

acc = accuracy_score(yvalid, pred)
print(acc)
for i in range(100):
    print(yvalid[i], list(pred)[i])



# lreg = LogisticRegression()
# lreg.fit(xtrain_bow, ytrain) # training the model
#
# prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
# prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
# prediction_int = prediction_int.astype(np.int)
#
# print(f1_score(yvalid, prediction_int)) # calculating f1 score





