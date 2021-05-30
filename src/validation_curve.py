import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import validation_curve
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(r'res\corpus_noStopwords.csv', encoding='utf-8')

# BAG OF WORDS
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
# vectorizer = TfidfVectorizer((analyzer='char_wb', ngram_range=(1, 5))
vectorizer.fit(df['Description'])

X = vectorizer.transform(df['Description'])
y = df['Classificació']

param_range = np.arange(0.00001, 0.1, 0.00005)
train_scores, test_scores = validation_curve(
    SGDClassifier(), X, y, param_name="alpha", param_range=param_range,
    scoring="recall_macro", n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve con SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
