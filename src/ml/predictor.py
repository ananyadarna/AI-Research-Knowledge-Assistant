import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.ml.dataset_prep import CATEGORIES

class DocumentClassifier:
    def __init__(self, model_path: str = "./models/tf_classifier.h5", tokenizer_path: str = "./models/tokenizer.pickle"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None

    def _load_resources(self):
        """
        Lazy load model and tokenizer to prevent overhead on startup
        and ensure they are loaded only when classification is requested.
        """
        if self.model is None:
            if not os.path.exists(self.model_path) or not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(
                    f"Model or tokenizer assets not found in './models/'. "
                    f"Please run the model training step first: 'python src/ml/train_classifier.py'."
                )
            
            # Load the pre-trained Keras model and the pickled tokenizer
            self.model = tf.keras.models.load_model(self.model_path)
            with open(self.tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)

    def predict_text(self, text: str) -> str:
        """
        Predict the category of a raw string chunk.
        """
        self._load_resources()
        if not text.strip():
            return "Other/General"

        # Preprocess text sequence
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=200, padding="post", truncating="post")

        # Run prediction
        preds = self.model.predict(padded, verbose=0)
        class_idx = np.argmax(preds[0])
        return CATEGORIES[class_idx]

    def classify_document(self, pages: list[dict]) -> str:
        """
        Classifies an entire document by analyzing the first 3 pages.
        Document titles, introductions, and abstracts are strong indicators
        of the overall domain category. We take a consensus average vote.
        """
        self._load_resources()
        if not pages:
            return "Other/General"

        # Look at the first 3 pages
        sample_pages = pages[:3]
        page_texts = [p["text"] for p in sample_pages if p["text"].strip()]

        if not page_texts:
            return "Other/General"

        # Convert to token sequences and pad
        sequences = self.tokenizer.texts_to_sequences(page_texts)
        padded = pad_sequences(sequences, maxlen=200, padding="post", truncating="post")

        # Get probabilities for each page
        preds = self.model.predict(padded, verbose=0)
        
        # Take the mean probability across the pages and choose the highest
        avg_preds = np.mean(preds, axis=0)
        class_idx = np.argmax(avg_preds)
        
        return CATEGORIES[class_idx]
