import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (classification_report, plot_confusion_matrix,
                             recall_score)
from sklearn.naive_bayes import MultinomialNB


def train_models(path_train: str, path_test: str, is_stopwords: bool, n_grames: int, n_clases: str):
    if is_stopwords:
        prefix = f'(AMB STOPWORDS) ({n_grames}-grames) ({n_clases} classes)'
    else:
        prefix = f'(NO STOPWORDS) ({n_grames}-grames) ({n_clases} classes)'

    df_train = pd.read_csv(path_train, encoding='utf-8')
    df_test = pd.read_csv(path_test, encoding='utf-8')

    if n_clases == '4':
        df_train = pd.concat([df_train[df_train['Classificació'] == 'ESPORTS'], df_train[df_train['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_train[df_train['Classificació'] == 'POLÍTICA'], df_train[df_train['Classificació'] == 'SOCIETAT']])
        df_test = pd.concat([df_test[df_test['Classificació'] == 'ESPORTS'], df_test[df_test['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_test[df_test['Classificació'] == 'POLÍTICA'], df_test[df_test['Classificació'] == 'SOCIETAT']])
    elif n_clases == '6':
        df_train = pd.concat([df_train[df_train['Classificació'] == 'ESPORTS'], df_train[df_train['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_train[df_train['Classificació'] == 'POLÍTICA'], df_train[df_train['Classificació'] == 'SOCIETAT'], df_test[df_test['Classificació'] == 'ACCIDENTS I CATÀSTROFES'], df_test[df_test['Classificació'] == 'MEDICINA I SANITAT']])
        df_test = pd.concat([df_test[df_test['Classificació'] == 'ESPORTS'], df_test[df_test['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_test[df_test['Classificació'] == 'POLÍTICA'], df_test[df_test['Classificació'] == 'SOCIETAT'], df_test[df_test['Classificació'] == 'ACCIDENTS I CATÀSTROFES'], df_test[df_test['Classificació'] == 'MEDICINA I SANITAT']])
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

    # BAG OF WORDS
    vectorizer = TfidfVectorizer(ngram_range=(n_grames, n_grames))
    vectorizer.fit(df_train['Description'])

    x_train_tfidf = vectorizer.transform(df_train['Description'])
    x_test_tfidf = vectorizer.transform(df_test['Description'])

    # MULTINOMIAL
    clf = MultinomialNB()
    clf.fit(x_train_tfidf, df_train['Classificació'])
    y_pred = clf.predict(x_test_tfidf)

    print(f'NB {prefix}')
    print(classification_report(df_test['Classificació'], y_pred, zero_division=0, digits=3))

    print(f"NB RECALL (macro): {recall_score(df_test['Classificació'], y_pred, average='macro')}")

    plot_confusion_matrix(clf, x_test_tfidf, df_test['Classificació'], include_values=False, normalize='true', xticks_rotation='vertical')
    plt.show()

    # SVM
    sgd = SGDClassifier()
    sgd.fit(x_train_tfidf, df_train['Classificació'])
    y_pred = sgd.predict(x_test_tfidf)

    print(f'SVM {prefix}')
    print(classification_report(df_test['Classificació'], y_pred, zero_division=0, digits=3))

    print(f"SVM RECALL (macro): {recall_score(df_test['Classificació'], y_pred, average='macro')}")

    plot_confusion_matrix(sgd, x_test_tfidf, df_test['Classificació'], include_values=False, normalize='true', xticks_rotation='vertical')
    plt.show()


if __name__ == "__main__":
    for c in ('6'):
        for i in [1]:
            train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', True, i, c)
            train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', False, i, c)
