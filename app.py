import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import datetime
import os
import speech_recognition as sr
from pydub import AudioSegment

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to analyze sentiment of each sentence
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

# Function to transcribe audio using SpeechRecognition library
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    
    # Convert audio file to WAV if it's in another format
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Mono channel, 16kHz for compatibility
    
    with open("temp.wav", "wb") as f:
        audio.export(f, format="wav")
    
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not transcribe the audio"
        except sr.RequestError:
            return "STT service is unavailable"
    finally:
        os.remove("temp.wav")  # Clean up temp file

# Streamlit UI
st.title("Sentiment Analysis of Customer Conversations (Text and Audio)")

# Text Input Section
st.write("### Text Input")
conversation = st.text_area("Enter a text-based conversation (each line is a new interaction):", height=200)

# Audio Input Section
st.write("### Audio Input")
audio_file = st.file_uploader("Upload an audio file (WAV, MP3)", type=["wav", "mp3"])

# If audio is uploaded, transcribe it
if audio_file:
    st.write("Transcribing audio...")
    transcribed_text = transcribe_audio(audio_file)
    st.write("Transcribed Text:", transcribed_text)
    conversation = transcribed_text

# Run sentiment analysis on text input
if st.button('Run Sentiment Analysis') and conversation:
    # Split conversation into separate messages (lines)
    messages = conversation.split("\n")
    
    # Limit processing of large conversations
    MAX_MESSAGES = 20
    if len(messages) > MAX_MESSAGES:
        st.warning(f"Only analyzing the first {MAX_MESSAGES} messages for memory efficiency.")
        messages = messages[:MAX_MESSAGES]

    # Analyze each message for sentiment
    sentiments = []
    for msg in messages:
        if msg.strip():  # Ignore empty lines
            label, score = analyze_sentiment(msg)
            if ": " in msg:
                speaker, content = msg.split(": ", 1)
            else:
                speaker, content = "Unknown", msg

            sentiments.append({
                "Timestamp": datetime.datetime.now(),
                "Speaker": speaker,
                "Message": content,
                "Sentiment": label,
                "Score": round(score, 2)
            })

    # Display results in DataFrame
    df = pd.DataFrame(sentiments)
    st.write("Conversation with Sentiment Labels:")
    st.table(df)

    # Plot sentiment over time
    fig = px.line(df, x='Timestamp', y='Score', color='Sentiment', title="Sentiment Score Over Time", markers=True)
    st.plotly_chart(fig)
else:
    st.warning("Please enter text or upload audio before running the analysis.")
