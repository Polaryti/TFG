import fasttext
from sklearn.metrics import top_k_accuracy_score, classification_report


def parse_scorer(pred):
    aux = []
    res = []
    for i in range(len(pred[0])):
        aux.append(int(pred[0][i].replace('__label__', '')))
    for i in range(len(aux)):
        res.append(pred[1][aux.index(i)])
    return res


def train_models(path_train: str, path_test: str, is_stopwords: bool, n_grames: int):
    if is_stopwords:
        prefix = f'(AMB STOPWORDS) ({n_grames}-grames)'
    else:
        prefix = f'(NO STOPWORDS) ({n_grames}-grames)'

    model = fasttext.train_supervised(path_train, dim=300, wordNgrams=n_grames, thread=4, verbose=0)

    y_pred = []
    y_true = []
    y_prob = []
    with open(path_test, 'r', encoding='utf-8') as test_file:
        for line in test_file.readlines():
            line = line.replace('__label__', '')
            y_true.append(int(line[:2].strip()))
            y_pred.append(int(model.predict(line[2:].strip())[0][0].replace('__label__', '')))
            # y_prob.append(parse_scorer(model.predict(line[2:].strip(), k=4)))

    print(classification_report(y_true, y_pred, digits=4))

    # print(f"{prefix} FastText ACC@{2}: {top_k_accuracy_score(y_true, y_prob, k=2)}")
    # print(f"{prefix} FastText ACC@{3}: {top_k_accuracy_score(y_true, y_prob, k=3)}")
    # print(f"{prefix} FastText ACC@{4}: {top_k_accuracy_score(y_true, y_prob, k=4)}")
    # print(f"{prefix} FastText ACC@{5}: {top_k_accuracy_score(y_true, y_prob, k=5)}")


if __name__ == "__main__":
    train_models(r'data/FastText/corpus_ambStopwords_ft_train_6.txt', r'data/FastText/corpus_ambStopwords_ft_test_6.txt', True, 1)
    train_models(r'data/FastText/corpus_noStopwords_ft_train_6.txt', r'data/FastText/corpus_noStopwords_ft_test_6.txt', False, 1)
    train_models(r'data/FastText/corpus_ambStopwords_ft_train_6.txt', r'data/FastText/corpus_ambStopwords_ft_test_6.txt', True, 2)
    train_models(r'data/FastText/corpus_noStopwords_ft_train_6.txt', r'data/FastText/corpus_noStopwords_ft_test_6.txt', False, 2)
    train_models(r'data/FastText/corpus_ambStopwords_ft_train_6.txt', r'data/FastText/corpus_ambStopwords_ft_test_6.txt', True, 3)
    train_models(r'data/FastText/corpus_noStopwords_ft_train_6.txt', r'data/FastText/corpus_noStopwords_ft_test_6.txt', False, 3)
