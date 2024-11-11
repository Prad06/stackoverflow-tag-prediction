import pandas

from base_models import logisticRegressionOut, randomForestOut, svmOut
from cnn_model import neuralNetOut
from data_cleaner import preprocess_data


def main():
    # Load and preprocess data
    X, y, df = preprocess_data()
    # Run models
    logisticRegressionOut(X, y)
    randomForestOut(X, y)
    svmOut(X, y)
    neuralNetOut(X, y)


if __name__ == "__main__":
    main()
