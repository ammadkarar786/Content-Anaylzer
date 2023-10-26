#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
# Title for your web app
st.title("Social Media Content Analyzer")

# Main content
st.write("Welcome to the Social Media Content Analyzer. This is a basic setup.")


# In[3]:


st.markdown(
    """
    <style>
    body {
        background-color: #e6f7ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# In[4]:


# Add an image with a description
st.image("logo.jpg", caption="Your Logo")


# In[5]:


st.markdown("## **Social Media Content Analyzer**", unsafe_allow_html=True)


# In[6]:


st.markdown(
    """<style>
    div.stButton > button {
        background-color: #ff9900;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Create a button with the specified style
st.button("Click me")


# In[7]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')


# In[8]:


def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)

    if sentiment_scores["compound"] >= 0.05:
        return "Positive"
    elif sentiment_scores["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

    


# In[11]:


# Text input box
user_text = st.text_area("Enter the text you want to analyze:")

if st.button("Analyze Sentiment"):
    if user_text:
        sentiment = analyze_sentiment(user_text)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter some text to analyze.")
    # Display sentiment with different colors
    if sentiment == "Positive":
        st.markdown("<p style='color: green;'>Sentiment: Positive</p>", unsafe_allow_html=True)
    elif sentiment == "Negative":
        st.markdown("<p style='color: red;'>Sentiment: Negative</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: blue;'>Sentiment: Neutral</p>", unsafe_allow_html=True)
                                   


# In[12]:


from nltk.tokenize import word_tokenize
from collections import Counter


# In[13]:

def analyze_keywords_and_hashtags(text):
    # Tokenize the text
    words = word_tokenize(text)
    
    # Filter for keywords and hashtags
    keywords = [word for word in words if word.isalnum()]
    hashtags = [word for word in words if word.startswith('#')]

    keyword_counts = Counter(keywords)
    hashtag_counts = Counter(hashtags)

    return keyword_counts, hashtag_counts




# In[16]:


# Text input box
user_text = st.text_area("Enter the text you want to analyze")

if st.button("Analyze Keywords and Hashtags"):
    if user_text:
        keyword_counts, hashtag_counts = analyze_keywords_and_hashtags(user_text)
        
    else:
        st.write("Please enter some text to analyze.")
    
# Display most common keywords
st.markdown("<h3>Most Common Keywords:</h3>", unsafe_allow_html=True)
keyword_counts, hashtag_counts = analyze_keywords_and_hashtags(user_text)
most_common_keywords = keyword_counts.most_common(5)
for i, (keyword, count) in enumerate(most_common_keywords):
    st.write(f"{i + 1}. {keyword} - Count: {count}")


# In[15]:
# Engagement prediction function
def predict_engagement(text):
    text_length = len(text)
    if text_length < 20:
        return "Low Engagement"
    elif 20 <= text_length <= 100:
        return "Medium Engagement"
    else:
        return "High Engagement"

# Display engagement prediction
st.markdown("<h3>Engagement Prediction:</h3>", unsafe_allow_html=True)
engagement = predict_engagement(user_text)
st.write(f"Predicted Engagement: {engagement}")





# In[ ]:
import matplotlib.pyplot as plt
# Display most common keywords
st.markdown("<h3>Most Common Keywords:</h3>", unsafe_allow_html=True)
keyword_counts, hashtag_counts = analyze_keywords_and_hashtags(user_text)  # Ensure you have both keyword and hashtag counts

most_common_keywords = keyword_counts.most_common(10)
if most_common_keywords:  # Check if there are most common keywords
    keywords, counts = zip(*most_common_keywords)
    for i, (keyword, count) in enumerate(zip(keywords, counts)):
        st.write(f"{i + 1}. {keyword} - Count: {count}")
else:
    st.write("No common keywords found.")




from textblob import TextBlob

# Text input box for a collection of text
text_collection = st.text_area("Enter a collection of text (one text per line):")

if st.button("Analyze Sentiment for Collection"):
    if text_collection:
        # Split the collection into individual text items
        texts = text_collection.split('\n')
        
        # Analyze sentiment for each text and store the results
        sentiment_results = []
        for i, text in enumerate(texts):
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            sentiment_results.append((f"Text {i+1}", text, sentiment))

        # Display sentiment results
        st.markdown("<h3>Sentiment Analysis for Collection:</h3>", unsafe_allow_html=True)
        for i, text, sentiment in sentiment_results:
            st.write(f"{i}:")
            st.write(text)
            if sentiment > 0:
                st.write("Sentiment: Positive")
            elif sentiment < 0:
                st.write("Sentiment: Negative")
            else:
                st.write("Sentiment: Neutral")
    else:
        st.write("Please enter a collection of text for analysis.")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wordcloud

# Word cloud input
st.markdown("<h3>Generate Word Cloud:</h3>", unsafe_allow_html=True)
word_cloud_text = st.text_area("Enter text to generate a word cloud:")

if st.button("Generate Word Cloud"):
    if word_cloud_text:
        word_cloud = generate_word_cloud(word_cloud_text)
        plt.figure(figsize=(8, 4))
        plt.imshow(word_cloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.write("Please enter text to generate a word cloud.")


import spacy

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# Extract named entities
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Named entity recognition input
st.markdown("<h3>Named Entity Recognition:</h3>", unsafe_allow_html=True)
ner_text = st.text_area("Enter text for Named Entity Recognition:")

if st.button("Extract Named Entities"):
    if ner_text:
        named_entities = extract_named_entities(ner_text)
        st.markdown("<h3>Named Entities:</h3>", unsafe_allow_html=True)
        for i, (entity, label) in enumerate(named_entities):
            st.write(f"{i + 1}. Entity: {entity} - Label: {label}")
    else:
        st.write("Please enter text for Named Entity Recognition.")
