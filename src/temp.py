import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def train_models(path: str, is_stopwords: bool):
    df = pd.read_csv(path, encoding="utf-8")

    # df = pd.concat([df[df['Classificació'] == 'ESPORTS'], df[df['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df[df['Classificació'] == 'POLÍTICA'], df[df['Classificació'] == 'SOCIETAT']])
    df = pd.concat([df[df['Classificació'] == 'ESPORTS'], df[df['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df[df['Classificació'] == 'POLÍTICA'], df[df['Classificació'] == 'SOCIETAT'], df[df['Classificació'] == 'ACCIDENTS I CATÀSTROFES'], df[df['Classificació'] == 'FESTES I TRADICIONS']])

    X_train, X_test, y_train, y_test = train_test_split(df['Description'], df['Classificació'], test_size=0.3)

    # BAG OF WORDS
    count_vect = CountVectorizer()
    count_vect.fit(df['Description'])
    X_counts = count_vect.transform(X_train)
    X_train_counts = count_vect.transform(X_train)
    X_test_counts = count_vect.transform(X_test)
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
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    print(f"NB RECALL (macro): {recall_score(y_test, y_pred, average='macro')}")

    # cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(clf, X_test_tfidf, y_test, include_values=True)
    plt.show()

    # SVM
    sgd = SGDClassifier()
    sgd.fit(X_train_tfidf, y_train)
    y_pred = sgd.predict(X_test_tfidf)

    print(f"SVM RECALL (macro): {recall_score(y_test, y_pred, average='macro')}")
    plot_confusion_matrix(sgd, X_test_tfidf, y_test, include_values=True)
    plt.show()


if __name__ == "__main__":
    train_models(r'res/corpus_ambStopwords.csv', True)
    train_models(r'res/corpus_noStopwords.csv', False)
