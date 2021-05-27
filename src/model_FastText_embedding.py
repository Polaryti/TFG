import fasttext
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import recall_score, plot_confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def train_models(path_train: str, path_test: str, fastText_path: str, is_stopwords: bool):
    df_train = pd.read_csv(path_train, encoding='utf-8')
    df_test = pd.read_csv(path_test, encoding='utf-8')
    model = fasttext.train_supervised(fastText_path, dim=300, wordNgrams=3)
    k = 2
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

    # MULTINOMIAL
    # clf = MultinomialNB()
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)

    # print(f"NB RECALL (macro): {recall_score(y_test, y_pred, average='macro')}")

    # plot_confusion_matrix(clf, x_test, y_test, include_values=False)
    # plt.show()

    # Random Forest
    # rf = RandomForestClassifier(n_jobs=-1)
    # rf.fit(x_train, y_train)
    # y_pred = rf.predict(x_test)

    # print(f"RF RECALL (macro): {recall_score(y_test, y_pred, average='macro')}")

    # plot_confusion_matrix(rf, x_test, y_test, include_values=False)
    # plt.show()

    # SVM
    sgd = SGDClassifier()
    sgd.fit(x_train, y_train)
    y_pred = sgd.predict(x_test)

    print(f"SVM RECALL (macro): {recall_score(y_test, y_pred, average='macro')}")
    plot_confusion_matrix(sgd, x_test, y_test, include_values=False)
    plt.show()

    y_test = le.transform(y_test)
    y_pred = sgd.decision_function(x_test)
    print(f"SVM K-ACCURACY ({k}): {top_k_accuracy_score(y_test, y_pred, k=k)}")


if __name__ == "__main__":
    train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', r'data/FastText/corpus_noStopwords_ft_train.txt', False)
    train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', r'data/FastText/corpus_ambStopwords_ft_train.txt', True)
