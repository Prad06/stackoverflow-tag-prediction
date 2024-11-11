import argparse

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_model(model_path):
    return joblib.load(model_path)


def load_vectorizer(vectorizer_path):
    return joblib.load(vectorizer_path)


def predict_tags(model, vectorizer, text):
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)
    return prediction


def main():
    parser = argparse.ArgumentParser(description="Predict tags for a given text.")
    parser.add_argument("text", type=str, help="Text to predict tags for")
    parser.add_argument("model_path", type=str, help="Path to the trained model")
    parser.add_argument("vectorizer_path", type=str, help="Path to the vectorizer")

    args = parser.parse_args()

    model = load_model(args.model_path)
    vectorizer = load_vectorizer(args.vectorizer_path)

    prediction = predict_tags(model, vectorizer, args.text)
    print(f"Predicted tags: {prediction}")


if __name__ == "__main__":
    main()
