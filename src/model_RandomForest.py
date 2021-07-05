import fasttext
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, plot_confusion_matrix,
                             recall_score, top_k_accuracy_score)
from sklearn.preprocessing import LabelEncoder


def train_models(path_train: str, path_test: str, fastText_path: str, is_stopwords: bool, n_grames: int):
    if is_stopwords:
        prefix = f'(AMB STOPWORDS) ({n_grames}-grames)'
    else:
        prefix = f'(NO STOPWORDS) ({n_grames}-grames)'
    df_train = pd.read_csv(path_train, encoding='utf-8')
    df_test = pd.read_csv(path_test, encoding='utf-8')
    model = fasttext.train_supervised(fastText_path, dim=300, wordNgrams=n_grames, verbose=0, epoch=10)
    le = LabelEncoder()
    le.fit(df_train['Classificació'].unique())

    x_train = []
    y_train = []
    for _, row in df_train.iterrows():
        x_train.append(model.get_sentence_vector(row['Description']))
        y_train.append(row['Classificació'])
    del df_train

    x_test = []
    y_test = []
    for _, row in df_test.iterrows():
        x_test.append(model.get_sentence_vector(row['Description']))
        y_test.append(row['Classificació'])
    del df_test

    del model

    # Random Forests
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    print(f"{prefix} SVM RECALL (macro): {recall_score(y_test, y_pred, average='macro')}")
    plot_confusion_matrix(rf, x_test, y_test, include_values=False, normalize='true', xticks_rotation='vertical')
    plt.show()

    print(classification_report(y_test, y_pred, zero_division=0, digits=3))
    y_test = le.transform(y_test)
    y_pred = rf.predict_proba(x_test)
    print(f"{prefix} SVM ACC@{2}: {top_k_accuracy_score(y_test, y_pred, k=2)}")
    print(f"{prefix} SVM ACC@{3}: {top_k_accuracy_score(y_test, y_pred, k=3)}")
    print(f"{prefix} SVM ACC@{4}: {top_k_accuracy_score(y_test, y_pred, k=4)}")
    print(f"{prefix} SVM ACC@{5}: {top_k_accuracy_score(y_test, y_pred, k=5)}")

    del le
    del x_train
    del x_test
    del y_train
    del y_test
    del y_pred
    del rf


if __name__ == "__main__":
    train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', r'data/FastText/corpus_ambStopwords_ft_train.txt', True, 1)
    train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', r'data/FastText/corpus_noStopwords_ft_train.txt', False, 1)
    train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', r'data/FastText/corpus_ambStopwords_ft_train.txt', True, 2)
    train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', r'data/FastText/corpus_noStopwords_ft_train.txt', False, 2)
    train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', r'data/FastText/corpus_ambStopwords_ft_train.txt', True, 3)
    train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', r'data/FastText/corpus_noStopwords_ft_train.txt', False, 3)
