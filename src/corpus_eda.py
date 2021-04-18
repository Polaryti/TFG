import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:10]


if __name__ == "__main__":
    stats_with_stopwords = {}
    stats_without_stopwords = {}

    df = pd.read_csv(r'res/data_corpus_full.csv', encoding="utf-8")
    for classification in df['Classificació'].unique():
        stats_with_stopwords[classification] = { 'mean_len_txts': max(df['Description'].str.split().map(lambda x: len(x)))}
        stats_with_stopwords[classification]['mean_len_words'] = max(df['Description'].str.split().apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x)))
        stats_with_stopwords[classification]['1_grama'] = get_top_ngram(df['Description'], 1)[:10]
        stats_with_stopwords[classification]['2_grama'] = get_top_ngram(df['Description'], 2)[:10]
        stats_with_stopwords[classification]['3_grama'] = get_top_ngram(df['Description'], 3)[:10]

    df = pd.read_csv(r'res/data_corpus_full_stopwords.csv', encoding="utf-8")
    for classification in df['Classificació'].unique():
        stats_without_stopwords[classification] = { 'mean_len_txts': max(df['Description'].str.split().map(lambda x: len(x)))}
        stats_without_stopwords[classification]['mean_len_words'] = max(df['Description'].str.split().apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x)))
        stats_without_stopwords[classification]['1_grama'] = get_top_ngram(df['Description'], 1)[:10]
        stats_without_stopwords[classification]['2_grama'] = get_top_ngram(df['Description'], 2)[:10]
        stats_without_stopwords[classification]['3_grama'] = get_top_ngram(df['Description'], 3)[:10]

    with open(r'res/stats_with_stopwords.csv', 'w', encoding="utf-8", newline='') as w_file:
        w_file.write('class,mean_len_txts,mean_len_words,1_grama,2_grama,3_grama')
        for key, value in stats_with_stopwords.items():
            w_file.write(f'{value.values()}\n')

    with open(r'res/stats_without_stopwords.csv', 'w', encoding="utf-8", newline='') as w_file:
        w_file.write('class,mean_len_txts,mean_len_words,1_grama,2_grama,3_grama')
        for key, value in stats_without_stopwords.items():
            w_file.write(f'{value.values()}\n')

