import pandas

from data_cleaner import preprocess_data
from base_models import logisticRegressionOut, svmOut, randomForestOut
from cnn_model import neuralNetOut

def main():
    # Load and preprocess data
    X, y, df = preprocess_data()
    # # Run models
    # logisticRegressionOut(X, y)
    # randomForestOut(X, y)
    # svmOut(X, y)
    neuralNetOut(X, y)

if __name__ == "__main__":
    main()