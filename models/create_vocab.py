from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from pandas import read_csv


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


# load doc and add to vocab
def add_doc_to_vocab(doc, vocab):
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)


# load all docs in a directory
def create_vocab(filename, test_ratio, vocab):
    raw_tweets = load_doc(filename, classes=2)
    lengh = int(len(raw_tweets) * (1 - test_ratio))
    # walk through all files in the folder
    for tweet in raw_tweets["tweet"][:lengh]:
        add_doc_to_vocab(tweet, vocab)


# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()


# define vocab
vocab = Counter()
# add all docs to vocab
test_ratio = 0.1
create_vocab('../database/downloadedB.csv', test_ratio, vocab)

# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))

# keep tokens with a min occurrence
min_occurane = 10
tokens = [k for k, c in vocab.items() if c >= min_occurane]
print(len(tokens))

# save tokens to a vocabulary file
save_list(tokens, '../models/vocab.txt')

