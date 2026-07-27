import os
import pickle
import numpy as np

# Try loading tensorflow if it is available, otherwise fall back to a scikit-learn model pipeline
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from src.ml.dataset_prep import CATEGORIES

class DocumentClassifier:
    def __init__(self, 
                 model_path: str = "./models/tf_classifier.h5", 
                 tokenizer_path: str = "./models/tokenizer.pickle",
                 fallback_path: str = "./models/sklearn_classifier.pickle"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.fallback_path = fallback_path
        self.model = None
        self.tokenizer = None
        self.fallback_pipeline = None
        self.use_fallback = not HAS_TENSORFLOW

    def _load_resources(self):
        """
        Lazy load model and tokenizer (or fallback scikit-learn pipeline) to prevent overhead on startup
        and ensure they are loaded only when classification is requested.
        """
        if self.use_fallback:
            if self.fallback_pipeline is None:
                if not os.path.exists(self.fallback_path):
                    raise FileNotFoundError(
                        f"Classifier model assets not found in './models/'. "
                        f"Please run the model training step first: 'python src/ml/train_classifier.py'."
                    )
                # Load the pre-trained scikit-learn pipeline (vectorizer + classifier)
                with open(self.fallback_path, "rb") as f:
                    self.fallback_pipeline = pickle.load(f)
        else:
            if self.model is None:
                if not os.path.exists(self.model_path) or not os.path.exists(self.tokenizer_path):
                    # Check if sklearn fallback is available as an alternative
                    if os.path.exists(self.fallback_path):
                        self.use_fallback = True
                        self._load_resources()
                        return
                    raise FileNotFoundError(
                        f"Model or tokenizer assets not found in './models/'. "
                        f"Please run the model training step first: 'python src/ml/train_classifier.py'."
                    )
                try:
                    self.model = tf.keras.models.load_model(self.model_path)
                    with open(self.tokenizer_path, "rb") as f:
                        self.tokenizer = pickle.load(f)
                except Exception:
                    # If tensorflow loading fails (e.g. Python version incompatibility), try sklearn fallback
                    if os.path.exists(self.fallback_path):
                        self.use_fallback = True
                        self._load_resources()
                        return
                    raise

    def predict_text(self, text: str) -> str:
        """
        Predict the category of a raw string chunk.
        """
        self._load_resources()
        if not text.strip():
            return "Other/General"

        if self.use_fallback:
            # Predict using scikit-learn pipeline (returns list of predicted label indices)
            preds = self.fallback_pipeline.predict([text])
            class_idx = int(preds[0])
            return CATEGORIES[class_idx]
        else:
            # Predict using TensorFlow model
            sequences = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=200, padding="post", truncating="post")
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

        if self.use_fallback:
            # Run predict_proba for each page and average the category likelihood arrays
            preds = self.fallback_pipeline.predict_proba(page_texts)
            avg_preds = np.mean(preds, axis=0)
            class_idx = np.argmax(avg_preds)
            return CATEGORIES[class_idx]
        else:
            # Predict probabilities using TensorFlow sequences
            sequences = self.tokenizer.texts_to_sequences(page_texts)
            padded = pad_sequences(sequences, maxlen=200, padding="post", truncating="post")
            preds = self.model.predict(padded, verbose=0)
            avg_preds = np.mean(preds, axis=0)
            class_idx = np.argmax(avg_preds)
            return CATEGORIES[class_idx]

