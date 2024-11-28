import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import whisper
from transformers import pipeline

# Load Whisper model and sentiment analysis pipeline
whisper_model = whisper.load_model("base")  # You can use "tiny", "base", "small", "medium", "large"
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to transcribe audio
def transcribe_audio_whisper(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

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
st.title("Sentiment Analysis with OpenAI Whisper (Audio and Text)")

# Upload audio file
st.subheader("Upload an Audio File")
audio_file = st.file_uploader("Choose an audio file (e.g., WAV, MP3, etc.)", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Save audio file locally
    audio_path = f"temp_audio.{audio_file.name.split('.')[-1]}"
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())
    
    # Transcribe audio
    st.write("Transcribing audio using OpenAI Whisper...")
    try:
        transcript = transcribe_audio_whisper(audio_path)
        st.write("Transcription:")
        st.text_area("Transcript", transcript, height=150, disabled=True)

        # Sentiment analysis on transcribed text
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

# Text-based conversation analysis
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
