import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import datetime
import os

# Disable GPU computation (Force CPU usage)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Use a smaller and lighter model (distilbert instead of XLM-Roberta)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to analyze sentiment of each sentence in a memory-efficient way
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Streamlit UI
st.title("Sentiment Analysis of Customer Conversations")

# Input section for customer service conversation
st.write("Enter a customer service conversation (each line is a new interaction between customer and service agent):")
conversation = st.text_area("Conversation", height=300, placeholder="Enter customer-service interaction here...")

# Add a button to run the analysis
if st.button('Run Sentiment Analysis'):
    if conversation:
        # Split conversation into separate messages (lines) for chunked processing
        messages = conversation.split("\n")
        
        # Limit processing of large conversations (for memory optimization)
        MAX_MESSAGES = 20  # Only process up to 20 messages at once (modify if needed)
        if len(messages) > MAX_MESSAGES:
            st.warning(f"Only analyzing the first {MAX_MESSAGES} messages for memory efficiency.")
            messages = messages[:MAX_MESSAGES]

        # Analyze each message for sentiment in small chunks
        sentiments = []
        for msg in messages:
            if msg.strip():  # Ignore empty lines
                label, score = analyze_sentiment(msg)
                sentiments.append({
                    "timestamp": datetime.datetime.now(), 
                    "text": msg, 
                    "sentiment": label, 
                    "score": score
                })

        # Convert the results into a DataFrame
        df = pd.DataFrame(sentiments)

        # Display the DataFrame
        st.write("Sentiment Analysis Results:")
        st.dataframe(df)

        # Plot sentiment over time using Plotly (optimize for small datasets)
        fig = px.line(df, x='timestamp', y='score', color='sentiment', title="Sentiment Score Over Time", markers=True)
        st.plotly_chart(fig)

        # Show conversation with sentiment labels
        st.write("Conversation with Sentiment Labels:")
        for i, row in df.iterrows():
            st.write(f"{row['timestamp']} - {row['text']} -> {row['sentiment']} (Score: {row['score']:.2f})")
    else:
        st.warning("Please enter a conversation before running the analysis.")
