import fasttext
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import recall_score, plot_confusion_matrix, top_k_accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def train_models(path_train: str, path_test: str, fastText_path: str, is_stopwords: bool, n_grames: int):
    if is_stopwords:
        prefix = f'(AMB STOPWORDS) ({n_grames}-grames)'
    else:
        prefix = '(NO STOPWORDS) ({n_grames}-grames)'
    df_train = pd.read_csv(path_train, encoding='utf-8')
    df_test = pd.read_csv(path_test, encoding='utf-8')
    model = fasttext.train_supervised(fastText_path, dim=300, wordNgrams=n_grames, verbose=0)
    le = LabelEncoder()
    le.fit(df_train['Classificació'].unique())

    x_train = []
    y_train = []
    for _, row in df_train.iterrows():
        x_train.append(model.get_sentence_vector(row['Description']))
        y_train.append(row['Classificació'])

    x_test = []
    y_test = []
    for _, row in df_test.iterrows():
        x_test.append(model.get_sentence_vector(row['Description']))
        y_test.append(row['Classificació'])

    # SVM
    sgd = SGDClassifier()
    sgd.fit(x_train, y_train)
    y_pred = sgd.predict(x_test)

    # print(f"{prefix} SVM RECALL (macro): {recall_score(y_test, y_pred, average='macro')}")
    # plot_confusion_matrix(sgd, x_test, y_test, include_values=False, normalize='all')
    # plt.show()

    print(classification_report(y_test, y_pred, zero_division=0, digits=3))
    y_test = le.transform(y_test)
    y_pred = sgd.decision_function(x_test)
    print(f"{prefix} SVM ACC@{2}: {top_k_accuracy_score(y_test, y_pred, k=2)}")
    print(f"{prefix} SVM ACC@{3}: {top_k_accuracy_score(y_test, y_pred, k=3)}")
    print(f"{prefix} SVM ACC@{4}: {top_k_accuracy_score(y_test, y_pred, k=4)}")
    print(f"{prefix} SVM ACC@{5}: {top_k_accuracy_score(y_test, y_pred, k=5)}")


if __name__ == "__main__":
    train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', r'data/FastText/corpus_ambStopwords_ft_train.txt', True, 1)
    train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', r'data/FastText/corpus_noStopwords_ft_train.txt', False, 1)
    train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', r'data/FastText/corpus_ambStopwords_ft_train.txt', True, 2)
    train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', r'data/FastText/corpus_noStopwords_ft_train.txt', False, 2)
    train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', r'data/FastText/corpus_ambStopwords_ft_train.txt', True, 3)
    train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', r'data/FastText/corpus_noStopwords_ft_train.txt', False, 3)
