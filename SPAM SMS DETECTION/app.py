import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
import streamlit as st

# Function to load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    df.rename(columns={"v1": "category", "v2": "message"}, inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to train the model
@st.cache_data
def train_model(df):
    df['message'] = df['message'].apply(preprocess_text)
    df["spam_int"] = df["category"].apply(lambda x: 1 if x == "spam" else 0)
    
    df_majority = df[df.spam_int == 0]
    df_minority = df[df.spam_int == 1]
    
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    tfidf_vectorizer = TfidfVectorizer()
    x = tfidf_vectorizer.fit_transform(df_upsampled['message'])
    y = df_upsampled['spam_int']
    
    model = MultinomialNB()
    model.fit(x, y)
    
    return model, tfidf_vectorizer

# Load data and train model
df = load_data()
model, tfidf_vectorizer = train_model(df)

# Streamlit interface
st.title("SMS Spam Classifier")
st.write("Enter a message to classify it as Spam or Not Spam")

user_input = st.text_area("Message")

if st.button("Classify"):
    if user_input:
        user_input_preprocessed = preprocess_text(user_input)
        user_input_tfidf = tfidf_vectorizer.transform([user_input_preprocessed])
        prediction = model.predict(user_input_tfidf)
        confidence = model.predict_proba(user_input_tfidf)
        
        if prediction == 0:
            st.markdown(f"<span style='color: green;'>Prediction: Not Spam - Confidence: {confidence[0][0] * 100:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color: red;'>Prediction: Spam - Confidence: {confidence[0][1] * 100:.2f}%</span>", unsafe_allow_html=True)
    else:
        st.write("Please enter a message.")
