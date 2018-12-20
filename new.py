import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True,)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stop = stopwords.words('english')

reviews = pd.read_csv("Tweets.csv")
reviews["text"] = reviews['text'].str.replace('[^\w\s]','')
reviews = reviews.drop(['negativereason',"negativereason_gold", "retweet_count", "tweet_coord",
                        "tweet_created", "tweet_location", "user_timezone", "negativereason_confidence",
                        "tweet_id", "airline_sentiment_gold", "name", "airline", "airline_sentiment_confidence"], axis=1)
reviews = reviews[reviews["airline_sentiment"] != 'neutral']
reviews['text'] = reviews['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#reviews['text'] = reviews['text'].apply(tokenizer.tokenize)


#reviews['text'] = reviews['text'].map(CleanList)
print(reviews)

v = TfidfVectorizer(min_df = 20)
# x = v.fit_transform(reviews['text'])
# tweets = list(x)
# np.set_printoptions(threshold=np.inf)
# print(x.toarray()[0])
reviews['tweetsVect'] = list(v.fit_transform(reviews['text']).toarray())
print(reviews)

X_train, X_test, y_train, y_test = train_test_split(reviews['tweetsVect'].tolist(),reviews['airline_sentiment'], test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
clf = LogisticRegression(random_state=42, solver='lbfgs').fit(X_train, y_train)
#clf = LinearSVC(random_state=0, tol=1e-5).fit(X_train,y_train)
#clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
from sklearn.metrics import accuracy_score
print(confusion_matrix(clf.predict(X_train), y_train))
print(confusion_matrix(clf.predict(X_test), y_test))
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(clf, X_train, y_train, cv=5)
#print(scores)
#print(accuracy_score(clf.predict(X_test), y_test))
#hello
