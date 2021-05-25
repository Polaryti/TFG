import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
# from joblib import Parallel, delayed
# import multiprocessing

stats_with_stopwords = {}
stats_without_stopwords = {}


def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:10]


def compute_info(classification, df, noStopwords, number_n_grama):
    if noStopwords:
        stats = stats_without_stopwords
    else:
        stats = stats_with_stopwords
    now = datetime.now()
    print(f'{now.strftime("%H:%M:%S")} - {classification}')
    stats[classification] = {'mean_len_txts': max(
        df['Description'].str.split().map(lambda x: len(x)))}
    stats[classification]['mean_len_words'] = max(
        df['Description'].str.split().apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x)))
    stats[classification]['1_grama'] = get_top_ngram(
        df['Description'], 1)[:number_n_grama]
    stats[classification]['2_grama'] = get_top_ngram(
        df['Description'], 2)[:number_n_grama]
    stats[classification]['3_grama'] = get_top_ngram(
        df['Description'], 3)[:number_n_grama]


if __name__ == "__main__":
    # num_cores = multiprocessing.cpu_count() - 1
    df = pd.read_csv(r'res/corpus_noStopwords.csv', encoding="utf-8")

    for cla in df['Classificació'].unique():
        compute_info(cla, df[df['Classificació'] == cla], True, 12)

    with open(r'res/stats_noStopwords.csv', 'w', encoding="utf-8", newline='') as w_file:
        w_file.write(
            'class,mean_len_txts,mean_len_words,1_grama,2_grama,3_grama')  # ,1_grama,2_grama,3_grama\n')
        for key, value in stats_without_stopwords.items():
            w_file.write(f'\n"{key}"')
            for v in stats_without_stopwords[key].values():
                w_file.write(f',"{str(v)}"')

    df = pd.read_csv(r'res/corpus_ambStopwords.csv', encoding="utf-8")

    for cla in df['Classificació'].unique():
        compute_info(cla, df[df['Classificació'] == cla], False, 12)

    with open(r'res/stats_ambStopwords.csv', 'w', encoding="utf-8", newline='') as w_file:
        w_file.write(
            'class,mean_len_txts,mean_len_words,1_grama,2_grama,3_grama')  # ,1_grama,2_grama,3_grama\n')
        for key, value in stats_with_stopwords.items():
            w_file.write(f'\n"{key}"')
            for v in stats_with_stopwords[key].values():
                w_file.write(f',"{str(v)}"')
