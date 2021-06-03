import fasttext
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, recall_score, top_k_accuracy_score
import random


def parse_scorer(pred):
    aux = []
    res = []
    for i in range(len(pred[0])):
        aux.append(int(pred[0][i].replace('__label__', '')))
    for i in range(len(aux)):
        res.append(pred[1][aux.index(i)])
    return res


generate_txt_file = False

df = pd.read_csv(r'res/corpus_noStopwords.csv', encoding="utf-8")
le = LabelEncoder()
le.fit(df['Classificació'].unique())
# print(f'Classess úniques: {len(le.classes_)}')

if generate_txt_file:
    test_split = 0.2
    df['Classificació'] = le.transform(df['Classificació'])

    samples_per_class = {}

    for _, row in df.iterrows():
        if row["Classificació"] not in samples_per_class:
            samples_per_class[row["Classificació"]] = []
        samples_per_class[row["Classificació"]].append(row["Description"])

    with open(r'data/FastText/corpus_noStopwords_ft_train.txt', 'w', encoding="utf-8", newline='') as w_file:
        for key, value in samples_per_class.items():
            random.shuffle(value)
            for sample in value[:int(len(value) * (1 - test_split))]:
                w_file.write(f'__label__{key} {sample}\n')

    y_true = []
    with open(r'data/FastText/corpus_noStopwords_ft_test.txt', 'w', encoding="utf-8", newline='') as w_file:
        for key, value in samples_per_class.items():
            random.shuffle(value)
            for sample in value[int(len(value) * (1 - test_split)):]:
                y_true.append(key)
                w_file.write(f'{sample}\n')


model = fasttext.train_supervised(r'data/FastText/corpus_noStopwords_ft_train.txt', dim=300, wordNgrams=1, thread=3)#, wordNgrams=2, epoch=8, thread=2)
# model = fasttext.train_supervised(r'data/FastText/corpus_noStopwords_ft_train.txt',
#                                   autotuneValidationFile=r'data/FastText/corpus_noStopwords_ft_test.txt',
#                                   autotuneDuration=300)

# model.get_word_vector()
# print(len(model.words))
# print(len(model.labels))

y_pred = []
y_true = []
y_prob = []
with open(r'data/FastText/corpus_noStopwords_ft_test.txt', 'r', encoding='utf-8') as test_file:
    for line in test_file.readlines():
        line = line.replace('__label__', '')
        y_true.append(int(line[:2].strip()))
        y_pred.append(int(model.predict(line[2:].strip())[0][0].replace('__label__', '')))
        y_prob.append(parse_scorer(model.predict(line[2:].strip(), k=38)))

print(recall_score(y_true, y_pred, average="macro"))

print(top_k_accuracy_score(y_true, y_prob, k=2))

print(classification_report(y_true, y_pred, digits=4))

# print(model.predict("banc espanya deficit"))
# print(model.predict("entrenador vcf"))
# print(model.predict("professorat universitat"))
# print(model.predict("falles declarades patrimoni inmaterial unesco"))

# print(model.predict(r'data/FastText/corpus_noStopwords_ft_test.txt'))

# def print_results(N, p, r):
#     print("N\t" + str(N))
#     print("P@{}\t{:.3f}".format(1, p))
#     print("R@{}\t{:.3f}".format(1, r))

# print_results(*model.test(r'data/FastText/corpus_noStopwords_ft_test.txt', k=1))

# print(model.predict("justicia", k=3))
# print(model.predict("paella", k=3))
# print(model.predict("causa penal", k=3))
# print(model.predict("ximo puig", k=3))
# print(model.predict("el dia del llibre a la ciutat de valència", k=3))

# i = 0
# for lasdas in le.classes_:
#     print(f'{lasdas}')
#     i += 1

# print(model.get_nearest_neighbors('paco'))
# print(len(model.get_word_vector("valència")))
# print(len(model.get_word_vector("la ciutat de valència")))