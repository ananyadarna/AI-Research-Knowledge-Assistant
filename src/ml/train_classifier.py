import os
import csv
import pickle
import numpy as np

# Try importing tensorflow, but do not fail if it's missing
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from src.ml.dataset_prep import prepare_dataset

def train_document_classifier():
    dataset_path = "./data/dataset/training_data.csv"
    
    # 1. Prepare synthetic dataset if it doesn't exist
    if not os.path.exists(dataset_path):
        print("Dataset not found. Generating synthetic dataset...")
        prepare_dataset(dataset_path)
        
    # 2. Load dataset using built-in csv module
    texts = []
    labels = []
    with open(dataset_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))

    
    os.makedirs("./models", exist_ok=True)
    
    # 3. Always train and save the fallback scikit-learn model (fast & robust)
    print("Training scikit-learn fallback document classifier model...")
    sklearn_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    sklearn_pipeline.fit(texts, labels)
    
    with open("./models/sklearn_classifier.pickle", "wb") as f:
        pickle.dump(sklearn_pipeline, f)
    print("scikit-learn fallback classifier saved to ./models/sklearn_classifier.pickle")
    
    # 4. Train TensorFlow model if tensorflow is installed
    if HAS_TENSORFLOW:
        # Text Preprocessing & Tokenization
        max_vocab_size = 5000
        max_length = 200
        
        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
        
        X = np.array(padded_sequences)
        y = np.array(labels)
        
        # Define Neural Network Architecture
        model = Sequential([
            Embedding(input_dim=max_vocab_size, output_dim=64, input_length=max_length),
            GlobalAveragePooling1D(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(7, activation="softmax")  # 7 target categories
        ])
        
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        # Train Model
        print("Training TensorFlow document classifier model...")
        model.fit(X, y, epochs=12, batch_size=32, validation_split=0.1, verbose=1)
        
        # Persist Model and Tokenizer
        model.save("./models/tf_classifier.h5")
        
        with open("./models/tokenizer.pickle", "wb") as f:
            pickle.dump(tokenizer, f)
            
        print("TensorFlow model and tokenizer successfully saved to ./models/")
    else:
        print("TensorFlow is not installed. Skipping TensorFlow neural network training.")

if __name__ == "__main__":
    train_document_classifier()

