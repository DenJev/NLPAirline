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
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('sentiwordnet')

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
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


def clean_text(text):
    print("LOL")
    text = "".join(text)
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
    print(text)

    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        print(tagged_sentence)

        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            print(wn_tag)
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
            print(word)
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            print(sentiment)
            tokens_count += 1

    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0

    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1

    # negative sentiment
    return 0


text = " it's really aggressive to blast obnoxious entertainment in your guests faces &amp; they have little recourse"
swn_polarity(text)
