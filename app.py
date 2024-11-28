import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import datetime
import speech_recognition as sr

# Use a smaller and lighter model (distilbert instead of XLM-Roberta)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to analyze sentiment of each sentence in a memory-efficient way
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']
    
    # Map score to -5 to 5 scale
    if label == "POSITIVE":
        # Scale the score from 0.5 to 1 to 0 to 5
        new_score = 5 * (score - 0.5) / 0.5  # Scaling
    else:  # Assuming it's NEGATIVE
        # Scale the score from 0 to 0.5 to 0 to -5
        new_score = -5 * (1 - score) / 0.5  # Scaling

    return label, round(new_score, 2)  # Return rounded score for readability

# Function to transcribe audio to text
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"

# Streamlit UI
st.title("Sentiment Analysis of Customer Conversations (Text and Audio)")

# Option to upload an audio file
st.subheader("Upload an Audio File")
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# Process uploaded audio file
if audio_file is not None:
    st.write("Transcribing audio...")
    transcript = transcribe_audio(audio_file)
    st.write("Transcription:")
    st.text_area("Transcript", transcript, height=150, disabled=True)

    # Sentiment analysis on transcribed audio
    if st.button('Run Sentiment Analysis on Audio'):
        if transcript.strip():
            # Split transcript into sentences for sentiment analysis
            sentences = transcript.split(". ")
            sentiments = []
            for sentence in sentences:
                if sentence.strip():  # Ignore empty lines
                    label, score = analyze_sentiment(sentence)
                    sentiments.append({
                        "Timestamp": datetime.datetime.now(), 
                        "Message": sentence, 
                        "Sentiment": label, 
                        "Score": score
                    })

            # Convert results into a DataFrame
            df = pd.DataFrame(sentiments)

            # Display the results
            st.write("Sentiment Analysis Results:")
            st.table(df)

            # Plot sentiment over time using Plotly
            fig = px.line(df, x='Timestamp', y='Score', color='Sentiment', title="Sentiment Score Over Time", markers=True)
            st.plotly_chart(fig)

# Input section for text-based conversation
st.subheader("Or Enter a Text-based Conversation")
conversation = st.text_area("Conversation (each line is a new interaction)", height=200)

# Add a button to run sentiment analysis on text
if st.button('Run Sentiment Analysis on Text'):
    if conversation.strip():
        # Split conversation into separate messages
        messages = conversation.split("\n")
        sentiments = []
        for msg in messages:
            if msg.strip():
                label, score = analyze_sentiment(msg)
                sentiments.append({
                    "Timestamp": datetime.datetime.now(),
                    "Message": msg,
                    "Sentiment": label,
                    "Score": score
                })

        # Convert the results into a DataFrame
        df = pd.DataFrame(sentiments)

        # Display the results
        st.write("Sentiment Analysis Results:")
        st.table(df)

        # Plot sentiment over time using Plotly
        fig = px.line(df, x='Timestamp', y='Score', color='Sentiment', title="Sentiment Score Over Time", markers=True)
        st.plotly_chart(fig)
    else:
        st.warning("Please enter a conversation before running the analysis.")
