import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random

df_ambStopwords = pd.read_csv(r'res/corpus_ambStopwords.csv', encoding="utf-8")
df_noStopwords = pd.read_csv(r'res/corpus_noStopwords.csv', encoding="utf-8")

test_split = 0.2

samples_per_class_ambStopwords = {}
samples_per_class_noStopwords = {}

# Class split
for _, row in df_ambStopwords.iterrows():
    if row["Classificació"] not in samples_per_class_ambStopwords:
        samples_per_class_ambStopwords[row["Classificació"]] = []
    samples_per_class_ambStopwords[row["Classificació"]].append(row["Description"])

for _, row in df_noStopwords.iterrows():
    if row["Classificació"] not in samples_per_class_noStopwords:
        samples_per_class_noStopwords[row["Classificació"]] = []
    samples_per_class_noStopwords[row["Classificació"]].append(row["Description"])

# Equal train-test split (with random shuffle)
train_ambStopwords = pd.DataFrame(columns=['Description', 'Classificació'])
for key, value in samples_per_class_ambStopwords.items():
    random.shuffle(value)
    for sample in value[:int(len(value) * (1 - test_split))]:
        train_ambStopwords = train_ambStopwords.append({"Description": value, "Classificació": key}, ignore_index=True)
# train_ambStopwords.sample(frac=1).reset_index(inplace=True, drop=True)
# train_ambStopwords = train_ambStopwords.sample(frac=1)

train_noStopwords = pd.DataFrame(columns=['Description', 'Classificació'])
for key, value in samples_per_class_noStopwords.items():
    random.shuffle(value)
    for sample in value[:int(len(value) * (1 - test_split))]:
        train_noStopwords = train_noStopwords.append({"Description": value, "Classificació": key}, ignore_index=True)
# train_noStopwords.sample(frac=1).reset_index(inplace=True, drop=True)

test_ambStopwords = pd.DataFrame(columns=['Description', 'Classificació'])
for key, value in samples_per_class_ambStopwords.items():
    random.shuffle(value)
    for sample in value[int(len(value) * (1 - test_split)):]:
        test_ambStopwords = test_ambStopwords.append({"Description": value, "Classificació": key}, ignore_index=True)
# test_ambStopwords.sample(frac=1).reset_index(inplace=True, drop=True)

test_noStopwords = pd.DataFrame(columns=['Description', 'Classificació'])
for key, value in samples_per_class_noStopwords.items():
    random.shuffle(value)
    for sample in value[int(len(value) * (1 - test_split)):]:
        test_noStopwords = test_noStopwords.append({"Description": value, "Classificació": key}, ignore_index=True)
# test_noStopwords.sample(frac=1).reset_index(inplace=True, drop=True)

# Save to csv (general)
test_ambStopwords.to_csv(r'res/corpus_ambStopwords_test.csv', encoding="utf-8")
train_ambStopwords.to_csv(r'res/corpus_ambStopwords_train.csv', encoding="utf-8")
test_noStopwords.to_csv(r'res/corpus_noStopwords_test.csv', encoding="utf-8")
train_noStopwords.to_csv(r'res/corpus_ambStopwords_train.csv', encoding="utf-8")

# Save to txt (FastText)
le = LabelEncoder()
le.fit(test_ambStopwords['Classificació'].unique())

with open(r'data/FastText/class_dict.csv', newline='', encoding="utf-8") as class_dict_file:
    i = 0
    class_dict_file.write('index,class\n')
    for cla in le.classes_:
        class_dict_file.write(f'{i},{cla}\n')
        i += 1

test_ambStopwords['Classificació'] = le.transform(test_ambStopwords['Classificació'])
train_ambStopwords['Classificació'] = le.transform(train_ambStopwords['Classificació'])
test_noStopwords['Classificació'] = le.transform(test_noStopwords['Classificació'])
train_noStopwords['Classificació'] = le.transform(train_noStopwords['Classificació'])

with open(r'data/FastText/corpus_ambStopwords_ft_train.txt', 'w', encoding="utf-8", newline='') as w_file:
    for key, value in train_ambStopwords.items():
        for sample in value[:int(len(value) * (1 - test_split))]:
            w_file.write(f'__label__{key} {sample}\n')

with open(r'data/FastText/corpus_ambStopwords_ft_test.txt', 'w', encoding="utf-8", newline='') as w_file:
    for key, value in test_ambStopwords.items():
        for sample in value[int(len(value) * (1 - test_split)):]:
            w_file.write(f'__label__{key} {sample}\n')

with open(r'data/FastText/corpus_noStopwords_ft_train.txt', 'w', encoding="utf-8", newline='') as w_file:
    for key, value in train_noStopwords.items():
        for sample in value[:int(len(value) * (1 - test_split))]:
            w_file.write(f'__label__{key} {sample}\n')

with open(r'data/FastText/corpus_noStopwords_ft_test.txt', 'w', encoding="utf-8", newline='') as w_file:
    for key, value in test_noStopwords.items():
        for sample in value[int(len(value) * (1 - test_split)):]:
            w_file.write(f'__label__{key} {sample}\n')
