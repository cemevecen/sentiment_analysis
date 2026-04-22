#backend/app.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import os
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI(title="AI Sentiment Analysis API")

# Lazy-loaded tokenizer and model to avoid hard failures at import time
tokenizer = None
model = None

API_KEY = os.getenv("API_KEY")

class TextRequest(BaseModel):
    text: str


def load_model():
    """Load tokenizer and model into module-level variables.
    Uses `from_pt=True` to allow loading PyTorch weights into TF if needed.
    """
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = TFAutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
            from_pt=True
        )
        logging.info("Model and tokenizer loaded successfully")
    except Exception:
        logging.exception("Failed to load model/tokenizer")
        tokenizer = None
        model = None


@app.on_event("startup")
def on_startup():
    load_model()


@app.post("/predict")
def predict(
    request: TextRequest,
    x_api_key: str = Header(...)
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    if tokenizer is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details."
        )

    inputs = tokenizer(
        request.text,
        padding=True,
        truncation=True,
        return_tensors="tf"
    )

    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1)

    return {
        "negative": float(probs[0][0]),
        "positive": float(probs[0][1])
    }