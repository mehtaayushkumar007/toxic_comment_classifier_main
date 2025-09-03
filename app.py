import os
import io
import torch
import numpy as np
import speech_recognition as sr
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Initialize the FastAPI App
app = FastAPI(
    title="Toxicity Multi-Input Classifier API",
    description="Identify toxic comments from text, audio files, or live microphone.",
    version="3.0",
)

# Load Pretrained Model and Tokenizer
try:
    MODEL_NAME = "unitary/unbiased-toxic-roberta"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    toxicity_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {e}")

# Define Input Data Model for API Request
class CommentRequest(BaseModel):
    comment: str

# Thresholds for Toxic Categories
thresholds = {
    'toxicity': 0.5,
    'severe_toxic': 0.5,
    'obscene': 0.5,
    'threat': 0.5,
    'insult': 0.3,
    'identity_hate': 0.5,
}

# Classification Logic
def classify_text(comment: str):
    if not comment.strip():
        return {"error": "Comment is empty."}

    results = toxicity_classifier(comment)

    toxic_categories = {}
    is_toxic = False

    for category_scores in results:
        for score_info in category_scores:
            label = score_info["label"].lower()
            score = score_info["score"]
            if label in thresholds and score >= thresholds[label]:
                toxic_categories[label] = round(score, 2)
                is_toxic = True

    response = {
        "comment": comment,
        "is_toxic": is_toxic,
        "toxic_categories": toxic_categories if is_toxic else {},
    }
    return response

# --------- API Endpoints ---------

# Health Check
@app.get("/")
def health_check():
    return {"message": "🎯 Toxicity Multi-Input Classifier API is running!"}

# Text Classification
@app.post("/classify/")
def classify_comment(request: CommentRequest):
    try:
        return classify_text(request.comment)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Audio File Classification
from pydub import AudioSegment




# Live Microphone Listening
@app.get("/live-mic/")
def live_mic_listen():
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        print("\n🎤 Speak into the microphone! (Listening for one sentence...)")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f"\n🔎 Recognized Text: {text}")

            # Classify the transcribed text
            result = classify_text(text)

            return {
                "recognized_text": text,
                "toxicity_result": result
            }

        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Could not understand the audio.")
        except sr.RequestError:
            raise HTTPException(status_code=503, detail="Speech recognition service unavailable.")

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --------- Main Runner ---------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
