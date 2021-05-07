import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def train_models(path_train: str, path_test: str, is_stopwords: bool):
    df_train = pd.read_csv(path_train, encoding='utf-8')
    df_test = pd.read_csv(path_test, encoding='utf-8')

    # df_train = pd.concat([df_train[df_train['Classificació'] == 'ESPORTS'], df_train[df_train['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_train[df_train['Classificació'] == 'POLÍTICA'], df_train[df_train['Classificació'] == 'SOCIETAT'], df_train[df_train['Classificació'] == 'ACCIDENTS I CATÀSTROFE'], df_train[df_train['Classificació'] == 'FESTES I TRADICIONS']])
    # df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    # df_test = pd.concat([df_test[df_test['Classificació'] == 'ESPORTS'], df_test[df_test['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_test[df_test['Classificació'] == 'POLÍTICA'], df_test[df_test['Classificació'] == 'SOCIETAT'], df_test[df_test['Classificació'] == 'ACCIDENTS I CATÀSTROFE'], df_test[df_test['Classificació'] == 'FESTES I TRADICIONS']])
    # df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

    # BAG OF WORDS
    count_vect = CountVectorizer()
    count_vect.fit(pd.concat([df_train['Description'], df_test['Description']]))
    X_counts = count_vect.transform(df_train['Description'])
    X_train_counts = count_vect.transform(df_train['Description'])
    X_test_counts = count_vect.transform(df_test['Description'])
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(X_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    if is_stopwords:
        print(f'Paraules uniques (AMB STOPWORDS): {X_train_tfidf.shape[1]}')
    else:
        print(f'Paraules uniques (SENSE STOPWORDS): {X_train_tfidf.shape[1]}')

    # MULTINOMIAL
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, df_train['Classificació'])
    y_pred = clf.predict(X_test_tfidf)

    print(f"NB RECALL (macro): {recall_score(df_test['Classificació'], y_pred, average='macro')}")

    # cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(clf, X_test_tfidf, df_test['Classificació'], include_values=True)
    plt.show()

    # SVM
    sgd = SGDClassifier()
    sgd.fit(X_train_tfidf, df_train['Classificació'])
    y_pred = sgd.predict(X_test_tfidf)

    # tuned_parameters = [{'kernel': ['linear', 'rbf'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # clf = GridSearchCV(
    #     SVC(), tuned_parameters, scoring='recall_macro'
    # )
    # clf.fit(X_train_tfidf, df_train['Classificació'])
    # y_pred = clf.best_estimator_.predict(X_test_tfidf)

    print(f"SVM RECALL (macro): {recall_score(df_test['Classificació'], y_pred, average='macro')}")
    plot_confusion_matrix(sgd, X_test_tfidf, df_test['Classificació'], include_values=True)
    plt.show()


if __name__ == "__main__":
    train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', False)
    train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', True)
