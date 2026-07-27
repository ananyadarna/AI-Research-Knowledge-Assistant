import os
import csv
import shutil
import numpy as np
import pytest
from src.ml.dataset_prep import prepare_dataset
from src.ml.train_classifier import train_document_classifier
from src.ml.predictor import DocumentClassifier

# Use a separate temporary model path for test isolation
TEST_MODEL_DIR = "./data/test_models"
TEST_MODEL_PATH = os.path.join(TEST_MODEL_DIR, "test_tf_classifier.h5")
TEST_TOKENIZER_PATH = os.path.join(TEST_MODEL_DIR, "test_tokenizer.pickle")
TEST_DATASET_PATH = "./data/test_dataset/test_training_data.csv"

@pytest.fixture(scope="module", autouse=True)
def setup_test_assets():
    """
    Sets up a small, fast-to-train dataset and cleans up assets after tests complete.
    """
    os.makedirs(TEST_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TEST_DATASET_PATH), exist_ok=True)
    
    # Generate a tiny training dataset to keep the test runtime under 2 seconds
    prepare_dataset(TEST_DATASET_PATH)
    
    yield
    
    # Clean up test directories
    if os.path.exists(TEST_MODEL_DIR):
        shutil.rmtree(TEST_MODEL_DIR)
    if os.path.exists(os.path.dirname(TEST_DATASET_PATH)):
        shutil.rmtree(os.path.dirname(TEST_DATASET_PATH))

def test_classifier_training_and_prediction():
    """
    Verifies that the dataset preparation generates data, training saves
    models, and prediction loads them to classify technical text.
    """
    # 1. Check training CSV is created using built-in csv module
    assert os.path.exists(TEST_DATASET_PATH)
    
    texts = []
    labels = []
    with open(TEST_DATASET_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        assert fieldnames is not None
        assert "text" in fieldnames
        assert "label" in fieldnames
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
            
    assert len(texts) > 0


    # 2. Run a fast training iteration (we override default paths)
    import pickle
    from src.ml.predictor import HAS_TENSORFLOW
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB


    # Always train and save the scikit-learn pipeline for the test
    sklearn_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    sklearn_pipeline.fit(texts, labels)
    
    # Save test fallback path
    TEST_FALLBACK_PATH = os.path.join(TEST_MODEL_DIR, "test_sklearn_classifier.pickle")
    with open(TEST_FALLBACK_PATH, "wb") as f:
        pickle.dump(sklearn_pipeline, f)

    if HAS_TENSORFLOW:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential

        max_vocab_size = 1000
        max_length = 50

        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_length, padding="post")

        X = np.array(padded)
        y = np.array(labels)

        model = Sequential([
            Embedding(input_dim=max_vocab_size, output_dim=16, input_length=max_length),
            GlobalAveragePooling1D(),
            Dense(16, activation="relu"),
            Dense(7, activation="softmax")
        ])
        
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        model.fit(X, y, epochs=1, batch_size=32, verbose=0)

        # Save test artifacts
        model.save(TEST_MODEL_PATH)
        with open(TEST_TOKENIZER_PATH, "wb") as f:
            pickle.dump(tokenizer, f)

        assert os.path.exists(TEST_MODEL_PATH)
        assert os.path.exists(TEST_TOKENIZER_PATH)
    
    assert os.path.exists(TEST_FALLBACK_PATH)

    # 3. Instantiate DocumentClassifier and run prediction
    classifier = DocumentClassifier(
        model_path=TEST_MODEL_PATH, 
        tokenizer_path=TEST_TOKENIZER_PATH,
        fallback_path=TEST_FALLBACK_PATH
    )
    
    # Predict on test texts
    test_text = "Amazon Web Services (AWS) deployment using Kubernetes and Docker microservices containers."
    category = classifier.predict_text(test_text)
    
    assert isinstance(category, str)
    assert category in [
        "Artificial Intelligence", "Machine Learning", "Computer Vision",
        "Natural Language Processing", "Robotics", "Cyber Security", "Cloud Computing"
    ]

    # Predict page-by-page consensus
    pages = [
        {"page_number": 1, "text": "This paper presents convolutional neural network CNN algorithms for object detection."},
        {"page_number": 2, "text": "Image segmentation and computer vision pixel recognition techniques are used."}
    ]
    doc_category = classifier.classify_document(pages)
    assert doc_category in [
        "Artificial Intelligence", "Machine Learning", "Computer Vision",
        "Natural Language Processing", "Robotics", "Cyber Security", "Cloud Computing"
    ]

