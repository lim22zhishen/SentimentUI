import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import datetime

# Load Sentiment and ASR pipelines
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h")

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']

    # Map score to -5 to 5 scale
    if label == "POSITIVE":
        new_score = 5 * (score - 0.5) / 0.5
    else:
        new_score = -5 * (1 - score) / 0.5

    return label, round(new_score, 2)

# Streamlit UI
st.title("Sentiment Analysis of Customer Conversations (Text and Audio)")

# Audio Upload Section
st.subheader("Upload an Audio File")
audio_file = st.file_uploader("Choose an audio file (WAV format recommended)", type=["wav"])

if audio_file is not None:
    st.write("Processing audio...")

    # Transcribe audio using Hugging Face ASR
    try:
        transcript = asr_pipeline(audio_file.read())["text"]
        st.write("Transcription:")
        st.text_area("Transcript", transcript, height=150, disabled=True)

        # Sentiment analysis on transcribed audio
        if st.button("Run Sentiment Analysis on Audio"):
            sentences = transcript.split(". ")
            sentiments = []
            for sentence in sentences:
                if sentence.strip():
                    label, score = analyze_sentiment(sentence)
                    sentiments.append({
                        "Timestamp": datetime.datetime.now(),
                        "Message": sentence,
                        "Sentiment": label,
                        "Score": score
                    })

            # Convert results into a DataFrame
            df = pd.DataFrame(sentiments)

            # Display results
            st.write("Sentiment Analysis Results:")
            st.table(df)

            # Plot sentiment over time
            fig = px.line(df, x="Timestamp", y="Score", color="Sentiment", title="Sentiment Score Over Time", markers=True)
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error processing audio file: {e}")

# Text-based Conversation Section
st.subheader("Or Enter a Text-based Conversation")
conversation = st.text_area("Conversation (each line is a new interaction)", height=200)

if st.button("Run Sentiment Analysis on Text"):
    if conversation.strip():
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

        # Convert results into a DataFrame
        df = pd.DataFrame(sentiments)

        # Display results
        st.write("Sentiment Analysis Results:")
        st.table(df)

        # Plot sentiment over time
        fig = px.line(df, x="Timestamp", y="Score", color="Sentiment", title="Sentiment Score Over Time", markers=True)
        st.plotly_chart(fig)
    else:
        st.warning("Please enter a conversation before running the analysis.")
