import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq


def compute_info(classification, df):
    a = get_top_ngram(df['Description'], 3)
    a = [item[0] for item in a]
    stats[classification] = set(a)


def get_unique(classification):
    aux = set()
    for key, values in stats.items():
        if key != classification:
            for v in values:
                aux.add(v)

    print(classification, len(stats[classification]), len(stats[classification].difference(aux)))


if __name__ == "__main__":
    stats = {}
    df = pd.read_csv(r'res/corpus_noStopwords.csv', encoding="utf-8")

    for cla in df['Classificació'].unique():
        compute_info(cla, df[df['Classificació'] == cla])

    for cla in df['Classificació'].unique():
        get_unique(cla)

    stats = {}
    df = pd.read_csv(r'res/corpus_ambStopwords.csv', encoding="utf-8")

    for cla in df['Classificació'].unique():
        compute_info(cla, df[df['Classificació'] == cla], False, 12)
