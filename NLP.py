import matplotlib
matplotlib.use("TkAgg")
import pandas as pd

from nltk.tokenize import TweetTokenizer
import string
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('stopwords')
from sklearn.metrics import accuracy_score

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


reviews = pd.read_csv("Tweets.csv")
print(reviews.columns)
### All excess columns removed
reviews = reviews.drop(['negativereason'], axis=1)
reviews = reviews.drop(["negativereason_gold"], axis=1)
reviews = reviews.drop(["retweet_count"], axis=1)
reviews = reviews.drop(["tweet_coord"], axis=1)
reviews = reviews.drop(["tweet_created"], axis=1)
reviews = reviews.drop(["tweet_location"], axis=1)
reviews = reviews.drop(["user_timezone"], axis=1)
reviews = reviews.drop(["negativereason_confidence"], axis=1)
reviews = reviews.drop(["tweet_id"], axis=1)
reviews = reviews.drop(["airline_sentiment_gold"], axis=1)
reviews = reviews.drop(["name"], axis=1)
reviews = reviews.drop(["airline"], axis=1)
reviews = reviews.drop(["airline_sentiment_confidence"], axis=1)
###
tt = TweetTokenizer(strip_handles=True, reduce_len=True)

def CleanList(list):
    x = [''.join(c for c in s if c not in string.punctuation) for s in list]

    x = [s for s in x if s]
    x = ' '.join(x)
    return x

reviews['text'] = reviews['text'].apply(tt.tokenize)
reviews['text'] = reviews['text'].map(CleanList)

reviews = reviews[reviews["airline_sentiment"] != 'neutral']
reviews = reviews.reset_index()

train, test = train_test_split(reviews ,test_size = 0.1)
train_pos = train[ train['airline_sentiment'] == 'positive']
train_pos = train_pos['text']
train_neg = train[ train['airline_sentiment'] == 'negative']
train_neg = train_neg['text']


def wordcloud_draw(data, color='black'):
    words = ' '.join(data)
    #print(words)
    cleaned_word = " ".join([word for word in words.split()
                             if 'http' not in word
                             and not word.startswith('@')
                             and not word.startswith('#')
                             and word != 'RT'
                             ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          width=2500,
                          height=2000
                          ).generate(cleaned_word)
    plt.figure(1, figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# print("Positive words")
# wordcloud_draw(train_pos, 'white')
# print("Negative words")
# wordcloud_draw(train_neg)
# tt = TweetTokenizer(strip_handles=True, reduce_len=True)
# reviews['text'] = reviews['text'].apply(tt.tokenize)
# reviews['text'] = reviews['text'].map(CleanList)

lemmatizer = WordNetLemmatizer()


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = "".join(text)
    #print(text)
    text = text.replace("<br />", " ")
    #text = text.decode("utf-8")

    return text


def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """

    sentiment = 0.0
    tokens_count = 0

    text = clean_text(text)

    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))

        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1

    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 'negative'

    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 'positive'

    # negative sentiment
    return 'negative'
print(reviews['text'])
# print(swn_polarity(reviews['text'][1]), reviews['airline_sentiment'][1]) # 1 1
# print(swn_polarity(reviews['text'][3]), reviews['airline_sentiment'][3]) # 1 1
# print(swn_polarity(reviews['text'][4]), reviews['airline_sentiment'][4]) # 1 1

print(swn_polarity(reviews['text']))
predict_sentiment = pd.DataFrame()


predict_sentiment['predict'] = reviews['text'].map(swn_polarity)
print(predict_sentiment)
print(accuracy_score(predict_sentiment, reviews['airline_sentiment']))
#print(reviews.loc[[12]])
