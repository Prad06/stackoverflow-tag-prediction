import string

# Download nltk corpus
import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("omw-1.4")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("word_tokenize")
print("\nCorpus Downloaded")


def preprocess_data():
    # Load and preprocess the data
    df_javascript = pd.read_csv("~/project--eece7205/data/QueryResults_JS.csv")
    df_cplusplus = pd.read_csv("~/project--eece7205/data/QueryResults_cplusplus.csv")
    df_python = pd.read_csv("~/project--eece7205/data/QueryResults_Python.csv")
    df_sql = pd.read_csv("~/project--eece7205/data/QueryResults_SQL.csv")
    df_java = pd.read_csv("~/project--eece7205/data/QueryResults_Java.csv")
    print("\nData Read")

    def merge_add_labels(*args):
        labeled_dfs = [(df.assign(Label=label)) for df, label in args]
        return pd.concat(labeled_dfs, ignore_index=True)

    df_labeled = merge_add_labels(
        (df_javascript, "JavaScript"),
        (df_cplusplus, "CPP"),
        (df_python, "Python"),
        (df_sql, "SQL"),
        (df_java, "Java"),
    )
    print("\nDF Labeled")

    def preprocess_text(text):
        text = "".join([char for char in text if char not in string.punctuation])
        text = word_tokenize(text.lower())
        text = [
            WordNetLemmatizer().lemmatize(word)
            for word in text
            if word not in stopwords.words("english")
        ]
        return " ".join(text)

    df_labeled["Processed_Text"] = df_labeled["Title"].apply(preprocess_text)
    print("\nText Processed")

    # TF-IDF transformation
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df_labeled["Processed_Text"])
    y = df_labeled["Label"]
    print("\nTFIDF Transformed")

    df_cleaned_sep = df_labeled[["Title", "Label"]]

    path = "~/project--eece7205/data/df_cleaned_sep_labeled.csv"
    df_cleaned_sep.to_csv(path)
    print(f"\nData saved to {path}")
    # path = "~/project--eece7205/data/tfidf.csv"
    # X.to_csv(path)
    print(f"\nData saved to {path}")

    return X, y, df_cleaned_sep
