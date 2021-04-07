import pandas as pd
from envar import PROCESED_DATA_FULL_STOPWORDS, PROCESED_DATA_FULL
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
    # Class count indivudal
    data = pd.read_csv(r'res/class_indiviudal.csv',
                       sep='%', header=None, index_col=0)

    data.plot(kind='bar')
    plt.ylabel('Frecuencia')
    plt.xlabel('Class')
    plt.title('Historiograma de les clases individuals')
    plt.show()

    # Class count dual
    data = pd.read_csv('res/class_dual.csv',
                       sep='%', header=None, index_col=0)

    data.plot(kind='bar')
    plt.ylabel('Frecuencia')
    plt.xlabel('Class')
    plt.title('Historiograma de les clases duals')
    plt.show()

    # (AMB STOPWORDS)
    df = pd.read_csv(PROCESED_DATA_FULL)

    # Longitud mitjana de les mostres
    data = df['Description'].str.split().map(lambda x: len(x)).hist()
    plt.plot(data=data)
    plt.title("Longitud mitjana delst textos (AMB STOPWORDS)")
    plt.xlabel("Número de paraules")
    plt.ylabel("Número de textos")
    plt.show()

    # Longitud mitjana de les paraules
    data = df['Description'].str.split().apply(lambda x: [len(i)
                                                          for i in x]).map(lambda x: np.mean(x)).hist()
    plt.plot(data=data)
    plt.title("Longitud mitjana de les paraules (AMB STOPWORDS)")
    plt.xlabel("Longitud de la paraula")
    plt.ylabel("Número de paraules")
    plt.show()

    # n-grames més freqüents
    n_grama = 3
    top_n_bigrams = get_top_ngram(df['Description'], n_grama)[:10]
    x, y = map(list, zip(*top_n_bigrams))
    data = sns.barplot(x=y, y=x)
    plt.plot(data=data)
    plt.title(f"{n_grama}-grames (AMB STOPWORDS)")
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

    # (SENSE STOPWORDS)
    df = pd.read_csv(PROCESED_DATA_FULL_STOPWORDS)

    # Longitud mitjana de les mostres
    data = df['Description'].str.split().map(lambda x: len(x)).hist()
    plt.plot(data=data)
    plt.title("Longitud mitjana delst textos (SENSE STOPWORDS)")
    plt.xlabel("Número de paraules")
    plt.ylabel("Número de textos")
    plt.show()

    # Longitud mitjana de les paraules
    data = df['Description'].str.split().apply(lambda x: [len(i)
                                                          for i in x]).map(lambda x: np.mean(x)).hist()
    plt.plot(data=data)
    plt.title("Longitud mitjana de les paraules (SENSE STOPWORDS)")
    plt.xlabel("Longitud de la paraula")
    plt.ylabel("Número de paraules")
    plt.show()

    # n-grames més freqüents
    n_grama = 3
    top_n_bigrams = get_top_ngram(df['Description'], n_grama)[:10]
    x, y = map(list, zip(*top_n_bigrams))
    data = sns.barplot(x=y, y=x)
    plt.plot(data=data)
    plt.title(f"{n_grama}-grames (SENSE STOPWORDS)")
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
