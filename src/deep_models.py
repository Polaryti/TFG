import pandas as pd
import gensim
from gensim import corpora, models, similarities
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def deep_learning(path_train: str, path_test: str):
    df_train = pd.read_csv(path_train, encoding='utf-8')
    # df_test = pd.read_csv(path_test, encoding='utf-8')

    # Feed-Forward Neural Networks (doc2vec)
    texts_train = df_train.to_dict('records')
    documents = [TaggedDocument(text['Description'].split(), [text['Classificació']]) for text in texts_train]

    model = gensim.models.Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, alpha=0.025, min_alpha=0.025, epochs=20)
    model.build_vocab(documents)
    model.train(documents, epochs=model.epochs, total_examples=model.corpus_count)


if __name__ == "__main__":
    deep_learning(r'res\corpus_noStopwords_train.csv', r'res\corpus_noStopwords_test.csv')
