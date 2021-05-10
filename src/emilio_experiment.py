import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# import fasttext
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def train_models(path_train: str, path_test: str, is_stopwords: bool):
    df_train = pd.read_csv(path_train, encoding='utf-8')
    df_test = pd.read_csv(path_test, encoding='utf-8')

    # df_train = pd.concat([df_train[df_train['Classificació'] == 'ESPORTS'], df_train[df_train['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_train[df_train['Classificació'] == 'POLÍTICA'], df_train[df_train['Classificació'] == 'SOCIETAT'], df_train[df_train['Classificació'] == 'ACCIDENTS I CATÀSTROFES'], df_train[df_train['Classificació'] == 'MEDICINA I SANITAT']])
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    # df_test = pd.concat([df_test[df_test['Classificació'] == 'ESPORTS'], df_test[df_test['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_test[df_test['Classificació'] == 'POLÍTICA'], df_test[df_test['Classificació'] == 'SOCIETAT'], df_test[df_test['Classificació'] == 'ACCIDENTS I CATÀSTROFES'], df_test[df_test['Classificació'] == 'MEDICINA I SANITAT']])
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

    # BAG OF WORDS
    count_vect = CountVectorizer(ngram_range=(3, 3))
    count_vect.fit(pd.concat([df_train['Description'], df_test['Description']]))
    x_counts = count_vect.transform(df_train['Description'])
    x_train_counts = count_vect.transform(df_train['Description'])
    x_test_counts = count_vect.transform(df_test['Description'])
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(x_counts)
    x_train_tfidf = tfidf_transformer.transform(x_train_counts)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)
    if is_stopwords:
        print(f'Paraules uniques (AMB STOPWORDS): {x_train_tfidf.shape[1]}')
    else:
        print(f'Paraules uniques (SENSE STOPWORDS): {x_train_tfidf.shape[1]}')

    # XGBoost
    le = LabelEncoder()
    le.fit(df_train['Classificació'].unique())

    dtrain = xgb.DMatrix(x_train_tfidf, label=le.transform(df_train['Classificació']))
    dtest = xgb.DMatrix(x_test_tfidf)
    param = {}
    param['eval_metric'] = 'auc'
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 10
    bst = xgb.train(param, dtrain, num_round, evallist)
    y_pred = bst.predict(dtest)

    print(f"XGBoost RECALL (macro): {recall_score(le.transform(df_test['Classificació']), y_pred, average='macro')}")
    plot_confusion_matrix(bst, dtest, le.transform(df_test['Classificació']), include_values=False)
    plt.show()

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(x_train_tfidf, df_train['Classificació'])
    y_pred = rf.predict(x_test_tfidf)

    print(f"SVM RECALL (macro): {recall_score(df_test['Classificació'], y_pred, average='macro')}")
    plot_confusion_matrix(rf, x_test_tfidf, df_test['Classificació'], include_values=False)
    plt.show()

    # FastText
    # model = fasttext.train_supervised(r'data/FastText/corpus_ambStopwords_ft.txt', wordNgrams=3)
    # train = []
    # test = []
    # for _, row in df_train.iterrows():
    #     train.append(model.get_word_vector(row['Description']))
    # for _, row in df_test.iterrows():
    #     test.append(model.get_word_vector(row['Description']))

    # MULTINOMIAL
    # clf = MultinomialNB()
    # clf.fit(train, df_train['Classificació'])
    # y_pred = clf.predict(test)

    # print(f"NB RECALL (macro): {recall_score(df_test['Classificació'], y_pred, average='macro')}")

    # # cm = confusion_matrix(y_test, y_pred)
    # plot_confusion_matrix(clf, x_test_tfidf, df_test['Classificació'], include_values=True)
    # plt.show()

    # SVM
    sgd = SGDClassifier()
    sgd.fit(x_train_tfidf, df_train['Classificació'])
    y_pred = sgd.predict(x_test_tfidf)

    # tuned_parameters = [{'kernel': ['linear', 'rbf'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # clf = GridSearchCV(
    #     SVC(), tuned_parameters, scoring='recall_macro'
    # )
    # clf.fit(x_train_tfidf, df_train['Classificació'])
    # y_pred = clf.best_estimator_.predict(x_test_tfidf)

    print(f"SVM RECALL (macro): {recall_score(df_test['Classificació'], y_pred, average='macro')}")
    plot_confusion_matrix(sgd, X_test_tfidf, df_test['Classificació'], include_values=True)
    # plt.show()

    print(sgd._predict_proba(X_train_tfidf[0]))


if __name__ == "__main__":
    train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', False)
    train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', True)
