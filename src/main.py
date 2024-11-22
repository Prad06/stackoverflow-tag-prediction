from base_models import logisticRegressionOut, randomForestOut, svmOut
from cnn_model import neuralNetOut
from data_cleaner import preprocess_data
from distilbert import distilbertOut


def main():
    X, y, df = preprocess_data()

    logisticRegressionOut(X, y)
    randomForestOut(X, y)
    svmOut(X, y)
    neuralNetOut(X, y)
    distilbertOut(df)


if __name__ == "__main__":
    main()
