import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def train_models(path: str):
    df = pd.read_csv(path, encoding="utf-8")

    X_train, X_test, y_train, y_test = train_test_split(df['Description'], df['Classificació_01'], test_size=0.25, random_state=42)

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
    print(f'Paraules uniques (AMB STOPWORDS): {X_train_tfidf.shape[1]}')

    # MULTINOMIAL
    clf = MultinomialNB()
    clf.fit(X_train_counts, y_train)
    y_pred = clf.predict(X_test_tfidf)

    print(f"RECALL (macro): {recall_score(y_test, y_pred, average='macro')}")

    # cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(clf, X_test_tfidf, y_test, include_values=False)
    plt.show()

    # SVM
    sgd = SGDClassifier()
    sgd.fit(X_train_counts, y_train)
    y_pred = sgd.predict(X_test_tfidf)

    print(f"RECALL (macro): {recall_score(y_test, y_pred, average='macro')}")


if __name__ == "__main__":
    df = pd.read_csv(r'res\data_corpus.csv', encoding="utf-8")

    # CLASS COUNT
    simple_class_count = {}
    combined_class_count = {}
    for index, row in df.iterrows():
        raw = row['Classificació'].strip()
        if raw not in combined_class_count:
            combined_class_count[raw] = 0
        combined_class_count[raw] += 1

        for single_class in raw.split('|'):
            single_class = single_class.strip()
            if single_class not in simple_class_count:
                simple_class_count[single_class] = 0
            simple_class_count[single_class] += 1

    with open(r'res/class_corpus_indiviudal.csv', 'w', encoding="utf-8", newline='') as w_file:
        simple_class_count = {k: v for k, v in sorted(
            simple_class_count.items(), key=lambda item: item[1], reverse=True)}
        for key, value in simple_class_count.items():
            w_file.write(f'{key}% {value}\n')

    with open(r'res/class_corpus_dual.csv', 'w', encoding="utf-8", newline='') as w_file:
        combined_class_count = {k: v for k, v in sorted(
            combined_class_count.items(), key=lambda item: item[1], reverse=True)}
        for key, value in combined_class_count.items():
            w_file.write(f'{key}% {value}\n')

    train_models(r'res\data_corpus_full.csv')
    train_models(r'res\data_corpus_full_stopwords.csv')

    # # (AMB STOPWORDS)
    # df = pd.read_csv(r'res\data_corpus_full.csv', encoding="utf-8")
    # # BAG OF WORDS
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(df['Description'])
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # print(f'Paraules uniques (AMB STOPWORDS): {X_train_tfidf.shape[1]}')

    # # MULTINOMIAL
    # text_clf = Pipeline([('vect', CountVectorizer()),
    #                      ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    # text_clf = text_clf.fit(df['Description'], df['Classificació'])

    # parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
    #               'tfidf__use_idf': (True, False),
    #               'clf__alpha': (1e-2, 1e-3), }
    # gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    # gs_clf.fit(df['Description'], df['Classificació'])

    # plot_confusion_matrix(gs_clf, df['Description'], df['Classificació'])
    # plt.show()
    # print(f"Naive bayes (AMB STOPWORDS): {gs_clf.best_score_}")

    # # SVM
    # text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer(
    # )), ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])

    # parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
    #                   'tfidf__use_idf': (True, False),
    #                   'clf-svm__alpha': (1e-2, 1e-3)}

    # gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
    # gs_clf_svm.fit(df['Description'], df['Classificació'])

    # plot_confusion_matrix(gs_clf_svm, df['Description'], df['Classificació'])
    # plt.show()
    # print(f"SVM (AMB STOPWORDS): {gs_clf_svm.best_score_}")

    # # (SENSE STOPWORDS)
    # df = pd.read_csv(r'res\data_corpus_full_stopwords.csv', encoding="utf-8")
    # # BAG OF WORDS
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(df['Description'])
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # print(f'Paraules uniques (SENSE STOPWORDS): {X_train_tfidf.shape[1]}')

    # # MULTINOMIAL
    # text_clf = Pipeline([('vect', CountVectorizer()),
    #                      ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    # text_clf = text_clf.fit(df['Description'], df['Classificació'])

    # parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
    #               'tfidf__use_idf': (True, False),
    #               'clf__alpha': (1e-2, 1e-3), }
    # gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, scoring="recall_macro")
    # gs_clf.fit(df['Description'], df['Classificació'])

    # plot_confusion_matrix(gs_clf, df['Description'], df['Classificació'])
    # plt.show()
    # print(f"Naive bayes (SENSE STOPWORDS): {gs_clf.best_score_}")

    # # SVM
    # text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer(
    # )), ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])

    # parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
    #                   'tfidf__use_idf': (True, False),
    #                   'clf-svm__alpha': (1e-2, 1e-3)}

    # gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1, scoring="recall_macro")
    # gs_clf_svm.fit(df['Description'], df['Classificació'])

    # plot_confusion_matrix(gs_clf_svm, df['Description'], df['Classificació'])
    # plt.show()
    # print(f"SVM (SENSE STOPWORDS): {gs_clf_svm.best_score_}")
