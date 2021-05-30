import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from recall_metric import r_at_1, r_at_2, r_at_5

df_train = pd.read_csv(r'res/corpus_noStopwords_train.csv', encoding='utf-8')
df_test = pd.read_csv(r'res/corpus_noStopwords_test.csv', encoding='utf-8')

# BAG OF WORDS
vectorizer = TfidfVectorizer(ngram_range=(3, 3))
vectorizer.fit(df_train['Description'])
x_train_tfidf = vectorizer.transform(df_train['Description'])
x_test_tfidf = vectorizer.transform(df_test['Description'])

sgd = SGDClassifier()
sgd.fit(x_train_tfidf, df_train['Classificació'])
y_pred = sgd.predict(x_test_tfidf)

print(f1_score(df_test['Classificació'], y_pred, average='macro'))
