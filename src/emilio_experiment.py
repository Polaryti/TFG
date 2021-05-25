import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import recall_score, plot_confusion_matrix
import matplotlib.pyplot as plt


# TODO: Revisar
def scorer(x_test, y_test, n):
    scores = {}
    i = 0
    for y in y_test:
        if y in sorted(range(len(x_test[i])), key=lambda k: x_test[i][k], reverse=True)[:n]:
            if y not in scores:
                scores[y] = [1, 1]
            else:
                scores[y][0] += 1
                scores[y][1] += 1
        else:
            if y not in scores:
                scores[y] = [0, 1]
            else:
                scores[y][1] += 1
        i += 1

    rec = 0.0
    for key, value in scores.items():
        rec = float(value[0]) / float(value[1])

    return rec / float(len(scores))


def train_models(path_train: str, path_test: str, is_stopwords: bool):
    df_train = pd.read_csv(path_train, encoding='utf-8')
    df_test = pd.read_csv(path_test, encoding='utf-8')

    case = '4'
    if case == '4':
        df_train = pd.concat([df_train[df_train['Classificació'] == 'ESPORTS'], df_train[df_train['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_train[df_train['Classificació'] == 'POLÍTICA'], df_train[df_train['Classificació'] == 'SOCIETAT']])
        df_test = pd.concat([df_test[df_test['Classificació'] == 'ESPORTS'], df_test[df_test['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_test[df_test['Classificació'] == 'POLÍTICA'], df_test[df_test['Classificació'] == 'SOCIETAT']])
    elif case == '6':
        df_train = pd.concat([df_train[df_train['Classificació'] == 'ESPORTS'], df_train[df_train['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_train[df_train['Classificació'] == 'POLÍTICA'], df_train[df_train['Classificació'] == 'SOCIETAT']])
        df_test = pd.concat([df_test[df_test['Classificació'] == 'ESPORTS'], df_test[df_test['Classificació'] == 'JUSTÍCIA I ORDRE  PÚBLIC'], df_test[df_test['Classificació'] == 'POLÍTICA'], df_test[df_test['Classificació'] == 'SOCIETAT'], df_test[df_test['Classificació'] == 'ACCIDENTS I CATÀSTROFES'], df_test[df_test['Classificació'] == 'MEDICINA I SANITAT']])
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

    # BAG OF WORDS
    # count_vect = CountVectorizer(analyzer='char_wb', ngram_range=(1, 5))
    count_vect = CountVectorizer(ngram_range=(3, 3))
    # count_vect.fit(pd.concat([df_train['Description'], df_test['Description']]))
    count_vect.fit(df_train['Description'])

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

    # MULTINOMIAL
    clf = MultinomialNB()
    clf.fit(x_train_tfidf, df_train['Classificació'])
    y_pred = clf.predict(x_test_tfidf)

    print(f"NB RECALL (macro): {recall_score(df_test['Classificació'], y_pred, average='macro')}")

    plot_confusion_matrix(clf, x_test_tfidf, df_test['Classificació'], include_values=False)
    plt.show()

    # SVM
    sgd = SGDClassifier()
    sgd.fit(x_train_tfidf, df_train['Classificació'])
    y_pred = sgd.predict(x_test_tfidf)

    print(f"SVM RECALL (macro): {recall_score(df_test['Classificació'], y_pred, average='macro')}")

    plot_confusion_matrix(sgd, x_test_tfidf, df_test['Classificació'], include_values=False)
    plt.show()


if __name__ == "__main__":
    train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', False)
    train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', True)
