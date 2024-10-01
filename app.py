import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import datetime

# Initialize the multilingual sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Function to analyze sentiment of each sentence
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Streamlit UI
st.title("Sentiment Analysis of Customer Conversations")

# Input section for a customer service conversation
st.write("Enter a customer service conversation (each line is a new interaction between customer and service agent):")
conversation = st.text_area("Conversation", height=300, placeholder="Enter customer-service interaction here...")

# Initialize an empty DataFrame for the conversation and sentiment analysis
if conversation:
    # Split the conversation into separate messages (each line is a new message)
    messages = conversation.split("\n")
    
    # Analyze each message for sentiment
    sentiments = []
    for msg in messages:
        label, score = analyze_sentiment(msg)
        sentiments.append({"timestamp": datetime.datetime.now(), "text": msg, "sentiment": label, "score": score})

    # Convert the results into a DataFrame
    df = pd.DataFrame(sentiments)

    # Display DataFrame of results
    st.write("Sentiment Analysis Results:")
    st.dataframe(df)

    # Plotting sentiment over time using Plotly
    fig = px.line(df, x='timestamp', y='score', color='sentiment', title="Sentiment Score Over Time", markers=True)
    st.plotly_chart(fig)

    # Show conversation with sentiment labels
    st.write("Conversation with Sentiment Labels:")
    for i, row in df.iterrows():
        st.write(f"{row['timestamp']} - {row['text']} -> {row['sentiment']} (Score: {row['score']:.2f})")

else:
    st.write("Please enter a conversation above to see the sentiment analysis.")

