import fasttext
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import top_k_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def train_models(path_train: str, path_test: str, fastText_path: str, is_stopwords: bool, n_grames: int, n_clases: str):
    if is_stopwords:
        prefix = f'(AMB STOPWORDS) ({n_grames}-grames)'
    else:
        prefix = f'(NO STOPWORDS) ({n_grames}-grames)'
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

    # SVM
    sgd = SGDClassifier(n_jobs=4)
    sgd.fit(x_train, y_train)
    y_pred = sgd.predict(x_test)

    # print(f"{prefix} SVM RECALL (macro): {recall_score(y_test, y_pred, average='macro')}")
    # plot_confusion_matrix(sgd, x_test, y_test, include_values=False, normalize='all')
    # plt.show()

    print(classification_report(y_test, y_pred, zero_division=0, digits=4))
    y_test = le.transform(y_test)
    y_pred = sgd.decision_function(x_test)
    print(f"{prefix} SVM ACC@{2}: {top_k_accuracy_score(y_test, y_pred, k=2)}")
    print(f"{prefix} SVM ACC@{3}: {top_k_accuracy_score(y_test, y_pred, k=3)}")
    if n_clases != '4':
        print(f"{prefix} SVM ACC@{4}: {top_k_accuracy_score(y_test, y_pred, k=4)}")
        print(f"{prefix} SVM ACC@{5}: {top_k_accuracy_score(y_test, y_pred, k=5)}")

    del le
    del x_train
    del x_test
    del y_train
    del y_test
    del y_pred
    del sgd


if __name__ == "__main__":
    for c in ('4', '6'):
        for i in (1, 2, 3):
            train_models(r'res/corpus_ambStopwords_train.csv', r'res/corpus_ambStopwords_test.csv', r'data/FastText/corpus_ambStopwords_ft_train.txt', True, i, c)
            train_models(r'res/corpus_noStopwords_train.csv', r'res/corpus_noStopwords_test.csv', r'data/FastText/corpus_noStopwords_ft_train.txt', False, i, c)
