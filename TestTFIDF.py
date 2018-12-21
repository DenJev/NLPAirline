from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True,)

v = TfidfVectorizer()

text = ["The quick brown fox jumped over the lazy dog."]
texttwo= ["How are the you quick ,"]

text_train = v.fit_transform(text)
text_test = v.transform(texttwo)


print(text_train.shape)
print(text_test.toarray())

