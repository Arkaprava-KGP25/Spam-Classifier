import string

import nltk as nltk
import streamlit as st
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()  ## converting to lower cases
    text = nltk.word_tokenize(text)  ## tokenizing the words

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)  ## removing special characters only keeping alpha numeric characters

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)  ## removing stopwords and punctuations

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))  ## complete the stemming

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model_mnb.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    #1. Preprocess
    transformed_sms = transform_text(input_sms)
    #2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    #3. Predict
    result = model.predict(vector_input)[0]
    #4. Display
    if result == 1:
        st.header("Spam!")
    else:
        st.header("Not Spam!")
