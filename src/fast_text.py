import fasttext
import pandas as pd
from sklearn.preprocessing import LabelEncoder

generate_txt_file = False

df = pd.read_csv(r'res/corpus_noStopwords.csv', encoding="utf-8")
le = LabelEncoder()
le.fit(df['Classificació'].unique())
print(f'Classess úniques: {len(le.classes_)}')

if generate_txt_file:
    df['Classificació'] = le.transform(df['Classificació'])

    with open(r'data/FastText/corpus_noStopwords_ft.txt', 'w', encoding="utf-8", newline='') as w_file:
        for _, row in df.iterrows():
            w_file.write(f'__label__{row["Classificació"]} {row["Description"]}\n')

# model = fasttext.train_unsupervised(r'data/FastText/corpus_noStopwords_ft.txt', model='skipgram')

model = fasttext.train_supervised(r'data/FastText/corpus_noStopwords_ft.txt')

print(len(model.words))
print(len(model.labels))

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

print_results(*model.test(r'data/FastText/corpus_noStopwords_ft_test.txt'))

print(model.predict("justicia", k=3))
print(model.predict("paella", k=3))
print(model.predict("causa penal", k=3))
print(model.predict("ximo puig", k=3))
print(model.predict("el dia del llibre a la ciutat de valència", k=3))

i = 0
for l in le.classes_:
    print(f'{i}     {l}')
    i += 1
# print(le.inverse_transform([model.predict("paco pons")[0][0].split('__')[1]]))

# ft = fasttext.load_model(r'data/FastText/cc.ca.300.bin')

# print(ft.get_nearest_neighbors('paco'))
