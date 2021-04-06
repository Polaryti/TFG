import numpy as np
import sklearn
import pandas as pd
from envar import PROCESED_DATA, CATALAN_STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

if __name__ == "__main__":
    df = pd.read_csv(PROCESED_DATA)

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

    with open(r'res/simple_class_count.csv', 'w') as w_file:
        simple_class_count = {k: v for k, v in sorted(
            simple_class_count.items(), key=lambda item: item[1], reverse=True)}
        for key, value in simple_class_count.items():
            w_file.write(f'{key}, {value}\n')

    with open(r'res/combined_class_count.csv', 'w') as w_file:
        combined_class_count = {k: v for k, v in sorted(
            combined_class_count.items(), key=lambda item: item[1], reverse=True)}
        for key, value in combined_class_count.items():
            w_file.write(f'{key}, {value}\n')

    # BAG OF WORDS
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(df['Description'])
    print(f'Paraules uniques: {X_train_counts.shape[1]}')
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(f'Paraules uniques: {X_train_tfidf.shape}')

    # MULTINOMIAL
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf = text_clf.fit(df['Description'], df['Classificació'])

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3), }
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(df['Description'], df['Classificació'])
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

    # SVM
    text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer(
    )), ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])

    parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf-svm__alpha': (1e-2, 1e-3)}

    gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
    gs_clf_svm = gs_clf_svm.fit(df['Description'], df['Classificació'])
    print(gs_clf_svm.best_score_)
    print(gs_clf_svm.best_params_)
