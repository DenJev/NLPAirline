
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd


# In[2]:


pd.options.mode.chained_assignment = None


# In[3]:


from nltk.tokenize import TweetTokenizer

# In[4]:


tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True,)


# In[ ]:


stop = stopwords.words('english')
reviews = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin1', header = None)
test = pd.read_csv('testdata.manual.2009.06.14.csv', encoding='latin1', header = None)
reviews = reviews.append(test)


# In[ ]:





# In[ ]:


#reviews[5] = reviews[5].str.replace('[^\w\s]','')
reviews = reviews.iloc[:, [0,5]]
#print(reviews.columns)
reviews = reviews[reviews[0] != 1]
reviews = reviews[reviews[0] != 2]
reviews = reviews[reviews[0] != 3]


# In[ ]:


#print(reviews[5])


# In[ ]:


reviews[5] = reviews[5].apply(tokenizer.tokenize)


# In[ ]:


#print(reviews[reviews[0] == 4])


# In[ ]:


reviews[5] = reviews[5].apply(lambda x: ' '.join(map(str, x)))
#print(reviews[5])


# In[ ]:


reviews[5] = reviews[5].str.replace('[^\w\s]','')
#print(reviews[5])


# In[ ]:


reviews[5] = reviews[5].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#print(reviews[5])


# In[ ]:


v = TfidfVectorizer(min_df = 5)
reviews['tweetsVect'] = list(v.fit_transform(reviews[5]).toarray())
y_train = reviews[[0]].iloc[:16000]


# In[ ]:


X_train = reviews['tweetsVect'].iloc[:160000].tolist()
y_train = reviews[[0]].iloc[:160000]
X_test = reviews['tweetsVect'].iloc[160000:].tolist()
y_test = reviews[[0]].iloc[160000:]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
clf = LogisticRegression(random_state=42, solver='lbfgs').fit(X_train, y_train)
print(confusion_matrix(clf.predict(X_test), y_test))


