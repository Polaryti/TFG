import os
import numpy as np
import pandas as pd
import re
from envar import CATALAN_STOPWORDS
import csv
from string import digits

pattern_comments = re.compile(r'\(+(.*?)\)+')
pattern_ray = re.compile(r'\s*-+\s*')
pattern_multiple_dot = re.compile(r'\.\s(\.\s)+|\.\.+')
pattern_parentesis = re.compile(r'\(+|\)+')
pattenr_claudators = re.compile(r'\[+|\]+')
pattern_dot_space = re.compile(r'\.')
pattern_dot_m_space = re.compile(r'\.\s+')
pattern_m_spaces = re.compile(r'\s+')
pattern_question_marks = re.compile(r'\?+')
pattern_signs = re.compile(r"[¿?¡!@#$\.,'%&:\"]")
pattern_news = re.compile(r'$\s*NEW .*$')

remove_digits = str.maketrans('', '', digits)


def noise_removal(txt):
    if txt is np.NaN:
        return txt

    txt = str(txt).strip()
    txt = txt.replace('\n', ' ').replace('\r', '')
    txt = re.sub(pattern_comments, ' ', txt)
    txt = re.sub(pattern_ray, ' ', txt)
    txt = re.sub(pattern_multiple_dot, '', txt)
    txt = re.sub(pattern_parentesis, ' ', txt)
    txt = re.sub(pattenr_claudators, ' ', txt)
    txt = re.sub(pattern_dot_space, '. ', txt)
    txt = re.sub(pattern_dot_m_space, '. ', txt)
    txt = re.sub(pattern_m_spaces, ' ', txt)
    txt = re.sub(pattern_question_marks, '?', txt)
    txt = re.sub(pattern_signs, '', txt)
    txt = txt.replace('/ RÈTOL/', ' ')
    txt = re.sub(pattern_news, '', txt)

    if txt.isupper() or type(txt) != str or txt.isnumeric():
        txt = np.NaN
    else:
        txt = txt.strip()
        txt_aux = txt.lower()
        if 'suport directe' in txt_aux or 'suport directe amb' in txt_aux or 'fals directe' in txt_aux:
            txt = np.NaN
        if txt_aux == "dnc":
            txt = np.NaN

    return txt


def stopwords_removal(txt):
    if txt is np.NaN:
        return txt
    else:
        txt = txt.split()
        for stopword in CATALAN_STOPWORDS:
            txt = list(filter((stopword).__ne__, txt))
    return ' '.join(txt)


def digits_removal(txt):
    if txt is np.NaN:
        return txt
    else:
        return txt.translate(remove_digits)


def apply_functions(txt):
    return digits_removal(stopwords_removal(noise_removal(txt)))


if __name__ == "__main__":
    # Generació del dataset sense stopwords i estadistiques de les clases
    sample_count = 0
    df_res = pd.DataFrame()
    for path, subdirs, files in os.walk('data/TextClassification/corpus'):
        for name in files:
            df = pd.read_excel(os.path.join(path, name))
            df = df[['Description', 'Classificació']]
            sample_count += len(df.index)

            df.dropna(subset=['Classificació'], inplace=True)

            df['Description'] = df['Description'].apply(apply_functions)
            df['Description'].replace('', np.NaN, inplace=True)
            df.dropna(subset=['Description'], inplace=True)
            df.drop_duplicates(['Description'], inplace=True)

            df['Description'] = df['Description'].apply(noise_removal)
            df['Description'].replace('', np.NaN, inplace=True)
            df.dropna(subset=['Description'], inplace=True)
            df.drop_duplicates(['Description'], inplace=True)

            df_res = pd.concat([df_res, df], copy=False)

    unique_cat = []
    aux = set()

    for s in df['Classificació']:
        unique_cat += (s.split('|'))

    for s in unique_cat:
        aux.add(s.strip())

    print("Mostres abans del preprocessament: {}".format(sample_count))
    print("Mostres després del preprocessament: {}".format(len(df_res)))
    print("Nombre total de clases úniques: {}".format(len(aux)))

    with open('res/clases_corpus.csv', 'w', encoding="utf-8") as write_file:
        for clase in aux:
            write_file.write("{}\n".format(clase))

    max = -1
    with open('res/data_corpus_full_no_stopwords.csv', 'w', encoding="utf-8", newline='') as w_file:
        writer = csv.writer(w_file)
        writer.writerow(['Description', 'Classificació',
                         'Classificació_01', 'Classificació_02', 'Classificació_03'])
        for index, row in df_res.iterrows():
            aux = row['Classificació'].split('|')
            if len(aux) > 1:
                if len(aux) > max:
                    max = len(aux)
                writer.writerow([row['Description'], row['Classificació'].strip(
                ), aux[0].strip(), aux[1].strip()])
            else:
                writer.writerow([row['Description'],
                                 row['Classificació'].strip(), row['Classificació'].strip(), ''])
    print(f'MAXIM: {max}')

    # Generació del dataset amb stopwords
    df = pd.DataFrame()
    for path, subdirs, files in os.walk('data/TextClassification/corpus'):
        for name in files:
            df = pd.concat([df, pd.read_excel(os.path.join(path, name))])

    df.dropna(subset=['Classificació'], inplace=True)

    df['Description'] = df['Description'].apply(noise_removal)
    df['Description'] = df['Description'].apply(digits_removal)
    df['Description'].replace('', np.NaN, inplace=True)
    df.dropna(subset=['Description'], inplace=True)
    df.drop_duplicates(['Description'], inplace=True)

    df['Description'] = df['Description'].apply(noise_removal)
    df['Description'] = df['Description'].apply(digits_removal)
    df['Description'].replace('', np.NaN, inplace=True)
    df.dropna(subset=['Description'], inplace=True)
    df.drop_duplicates(['Description'], inplace=True)

    df.drop(df.columns.difference(
        ['Description', 'Classificació']), 1, inplace=True)

    with open('res/data_corpus_full.csv', 'w', encoding="utf-8", newline='') as w_file:
        writer = csv.writer(w_file)
        writer.writerow(['Description', 'Classificació',
                         'Classificació_01', 'Classificació_02', 'Classificació_03'])
        for index, row in df.iterrows():
            aux = row['Classificació'].split('|')
            if len(aux) > 1:
                writer.writerow([row['Description'], row['Classificació'].strip(
                ), aux[0].strip(), aux[1].strip()])
            else:
                writer.writerow([row['Description'],
                                 row['Classificació'].strip(), row['Classificació'].strip(), ''])
