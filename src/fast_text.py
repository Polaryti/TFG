import fasttext
import pandas as pd
from sklearn.preprocessing import LabelEncoder

generate_txt_file = True

df = pd.read_csv(r'res/corpus_noStopwords.csv', encoding="utf-8")
le = LabelEncoder()
le.fit(df['Classificació'].unique())
print(le.classes_)

if generate_txt_file:
    df['Classificació'] = le.transform(df['Classificació'])

    with open(r'data/FastText/corpus_noStopwords_ft.txt', 'w', encoding="utf-8", newline='') as w_file:
        for _, row in df.iterrows():
            w_file.write(f'__label__{row["Classificació"]} {row["Description"]}\n')

# model = fasttext.train_unsupervised(r'data/FastText/corpus_noStopwords_ft.txt', model='skipgram')

model = fasttext.train_supervised(r'data/FastText/corpus_noStopwords_ft.txt')

print(len(model.words))
print(model.labels)

# def print_results(N, p, r):
#     print("N\t" + str(N))
#     print("P@{}\t{:.3f}".format(1, p))
#     print("R@{}\t{:.3f}".format(1, r))

# print_results(*model.test(r'res/stats_noStopwords_ft.txt'))

print(model.predict("esport"))
print(le.classes_[13])
# print(le.inverse_transform([model.predict("paco pons")[0][0].split('__')[1]]))

# ft = fasttext.load_model(r'data/FastText/cc.ca.300.bin')

# print(ft.get_nearest_neighbors('paco'))
