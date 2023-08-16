#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# ## Importing data

# In[7]:


df=pd.read_csv("IMDBDataset.csv")
df.head()


# # Clean the data

# In[8]:


#This function will create a parse tree and extract the plain text from a given html text
#Eg "<h1>Hello how are you</h1>  <div class=\"Main\">Ramesh</div>" will give
# Hello how are you Ramesh
def strip_html(text):
    parse_obj=BeautifulSoup(text,'html.parser')
    return parse_obj.get_text()

#re.sub() function to replace occurrences of the specified regular expression pattern with an empty string 
def remove_square_bracket(text):
    text = re.sub(r'\[[^]]*\]', '', text)
    return text

def remove_special_char(text,remove_digits=True):
    text=re.sub(r'[^a-zA-Z0-9\s]','',text)
    return text

def clean_data(text):
    text=strip_html(text)
    text=remove_square_bracket(text)
    text=remove_special_char(text)
    return text

df['review']=df['review'].apply(clean_data)



# ## Text Stemming

# In[13]:


#Stemming is done not lemmetisation because stemming work good in information retrival
def porter_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text=' '.join([ps.stem(word) for word in text.split()])
    return text
df['review']=df['review'].apply(porter_stemmer)


# ## Removing Stop Words
# 

# In[14]:


stopwords_list=set(stopwords.words('english'))
tokenizer=ToktokTokenizer()

def remove_stopwords(text):
    tokens=tokenizer.tokenize(text)
    tokens=[token.strip() for token in tokens] #strip will remove leading and trailing whitespaces
    new_token=[token for token in tokens if token.lower() not in stopwords_list]
    new_text=' '.join(new_token)
    return new_text

df['review']=df['review'].apply(remove_stopwords)


# ## Labelling Sentimental data

# In[18]:


lb=LabelBinarizer()
df['sentiment']=lb.fit_transform(df['sentiment'])
df.head()


# ## Spliting test train data

# In[19]:


reviews=df.review
sentiments=df.sentiment

train_review,test_review,train_sentiment,test_sentiment=train_test_split(reviews,sentiments,test_size=0.2,random_state=42)


# ## Tf-idf Vectorizer

# In[20]:


tfidf_obj=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
tf_train_reviews=tfidf_obj.fit_transform(train_review)
tf_test_reviews=tfidf_obj.transform(test_review)


# ## MultiNomial Naive Bayes

# In[21]:


MNB=MultinomialNB()
#MNB_bow=MNB.fit(cv_train_)
MNB_model=MNB.fit(tf_train_reviews,train_sentiment)
print(MNB_model)


# ## Accuracy

# In[22]:


prediction=MNB.predict(tf_test_reviews)
accuracy=accuracy_score(test_sentiment,prediction)


# In[23]:


print(accuracy)

