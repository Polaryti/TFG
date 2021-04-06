import pandas as pd
from envar import PROCESED_DATA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:10]


if __name__ == "__main__":
    df = pd.read_csv(PROCESED_DATA)

    # Longitud mitjana de les mostres
    data = df['Description'].str.split().map(lambda x: len(x)).hist()
    plt.plot(data=data)
    plt.title("Longitud mitjana de les mostres")
    plt.xlabel("Número de paraules")
    plt.ylabel("Número de mostres")
    plt.show()

    # Longitud mitjana de les paraules
    data = df['Description'].str.split().apply(lambda x: [len(i)
                                                          for i in x]).map(lambda x: np.mean(x)).hist()
    plt.plot(data=data)
    plt.title("Longitud mitjana de les paraules")
    plt.xlabel("Longitud de la paraula")
    plt.ylabel("Número de paraules")
    plt.show()

    # Bigrames més freqüents
    n_grama = 3
    top_n_bigrams = get_top_ngram(df['Description'], n_grama)[:10]
    x, y = map(list, zip(*top_n_bigrams))
    data = sns.barplot(x=y, y=x)
    plt.plot(data=data)
    plt.title(f"{n_grama}-grames")
    plt.xlabel(f"Número de {n_grama}-grames")
    plt.ylabel("Número de paraules")
    plt.show()

    # Wordcloud
    wordcloud = WordCloud(
        background_color='white',
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
   
    wordcloud = wordcloud.generate(str(df['Description']))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()
