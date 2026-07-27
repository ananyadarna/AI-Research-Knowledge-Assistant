import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from src.ml.dataset_prep import prepare_dataset

def train_document_classifier():
    dataset_path = "./data/dataset/training_data.csv"
    
    # 1. Prepare synthetic dataset if it doesn't exist
    if not os.path.exists(dataset_path):
        print("Dataset not found. Generating synthetic dataset...")
        prepare_dataset(dataset_path)
        
    # 2. Load dataset
    df = pd.read_csv(dataset_path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    
    # 3. Text Preprocessing & Tokenization
    max_vocab_size = 5000
    max_length = 200
    
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    
    X = np.array(padded_sequences)
    y = np.array(labels)
    
    # 4. Define Neural Network Architecture
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
    
    # 5. Train Model
    print("Training TensorFlow document classifier model...")
    model.fit(X, y, epochs=12, batch_size=32, validation_split=0.1, verbose=1)
    
    # 6. Persist Model and Tokenizer
    os.makedirs("./models", exist_ok=True)
    model.save("./models/tf_classifier.h5")
    
    with open("./models/tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)
        
    print("Model and tokenizer successfully saved to ./models/")

if __name__ == "__main__":
    train_document_classifier()
