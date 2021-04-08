# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import re
from envar import RAW_DATA, CATALAN_STOPWORDS
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

lemmatization_index = {}
apostrof_index = {}


def compute_lemmatization_index():
    with open(r'data/lemmatization-ca.txt', 'r') as r_file:
        for line in r_file.readlines():
            line = line.split()
            lemmatization_index[line[1]] = line[0]


def lemmatization(txt):
    if txt is np.NaN:
        return txt
    else:
        txt = txt.split()
        for token in txt:
            flag_is_upper = token.isupper()
            flag_is_lower = token.islower()
            flag_is_capit = not flag_is_upper and not flag_is_lower and token[0].isupper(
            )
            token = token.lower()
            if token in lemmatization_index:
                if flag_is_upper:
                    token = lemmatization_index[token].upper()
                elif flag_is_lower:
                    token = lemmatization_index[token].lower()
                elif flag_is_capit:
                    token = lemmatization_index[token].capitalize()
                else:
                    token = lemmatization_index[token]
    return ' '.join(txt)


def compute_apostrof_index():
    with open(r'data/apostrof.csv', 'r') as r_file:
        for line in r_file.readlines():
            line = line.split(',')
            apostrof_index[line[1]] = line[0]


def apostrof_removal(txt):
    if txt is np.NaN:
        return txt
    else:
        txt = txt.split()
        for token in txt:
            if '-' in token or '\'' in token:
                if '-' in token:
                    aux_token = token.split('-')
                else:
                    aux_token = token.split('-\'')

                if len(aux_token) > 3:
                    # print(f"WARNING: apostrog_removal() de {aux_token}")
                    pass

                token_res = ''
                for i_token in aux_token:
                    if len(i_token) > 0:
                        flag_is_upper = i_token.isupper()
                        flag_is_lower = i_token.islower()
                        flag_is_capit = not flag_is_upper and not flag_is_lower and i_token[0].isupper(
                        )
                        i_token = i_token.lower()
                        if i_token in lemmatization_index:
                            if flag_is_upper:
                                i_token += lemmatization_index[i_token].upper()
                            elif flag_is_lower:
                                i_token += lemmatization_index[i_token].lower()
                            elif flag_is_capit:
                                i_token += lemmatization_index[i_token].capitalize()
                            else:
                                i_token += lemmatization_index[i_token]

                token = token_res
    return ' '.join(txt)


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
    # txt = txt.replace(pattern_news, '', txt)

    if txt.isupper() or type(txt) != str or txt.isnumeric():
        txt = np.NaN
    else:
        txt = txt.strip()
        txt_aux = txt.lower()
        if 'suport directe' in txt_aux or 'suport directe amb' in txt_aux or 'fals directe' in txt_aux:
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


if __name__ == "__main__":
    # Generació del index de lemmatization
    compute_lemmatization_index()

    # Generació del index de apostros
    compute_apostrof_index()

    # Generació del dataset sense stopwords i estadistiques de les clases
    df = pd.read_excel(RAW_DATA)
    print("Mostres abans del preprocessament: {}".format(len(df)))

    df.dropna(subset=['Classificació'], inplace=True)

    #df['Description'] = df['Description'].apply(apostrof_removal)
    df['Description'] = df['Description'].apply(noise_removal)
    df['Description'] = df['Description'].apply(lemmatization)
    df['Description'] = df['Description'].apply(stopwords_removal)
    df['Description'] = df['Description'].apply(lemmatization)
    df['Description'] = df['Description'].apply(digits_removal)
    df['Description'].replace('', np.NaN, inplace=True)
    df.dropna(subset=['Description'], inplace=True)
    df.drop_duplicates(['Description'], inplace=True)

    df['Description'] = df['Description'].apply(noise_removal)
    df['Description'] = df['Description'].apply(lemmatization)
    df['Description'] = df['Description'].apply(stopwords_removal)
    df['Description'] = df['Description'].apply(digits_removal)
    df['Description'].replace('', np.NaN, inplace=True)
    df.dropna(subset=['Description'], inplace=True)
    df.drop_duplicates(['Description'], inplace=True)

    df.drop(df.columns.difference(
        ['Description', 'Classificació']), 1, inplace=True)

    unique_cat = []
    aux = set()

    for s in df['Classificació']:
        unique_cat += (s.split('|'))

    for s in unique_cat:
        aux.add(s.strip())

    print("Mostres després del preprocessament: {}".format(len(df)))
    print("Nombre total de clases úniques: {}".format(len(aux)))

    with open('res/clases.csv', 'w') as write_file:
        for clase in aux:
            write_file.write("{}\n".format(clase))

    df.to_csv('res/data.csv', index=False)

    with open('res/data_full_stopwords.csv', 'w') as w_file:
        writer = csv.writer(w_file)
        writer.writerow(['Description', 'Classificació',
                         'Classificació_01', 'Classificació_02'])
        for index, row in df.iterrows():
            aux = row['Classificació'].split('|')
            if len(aux) > 1:
                writer.writerow([row['Description'], row['Classificació'].strip(
                ), aux[0].strip(), aux[1].strip()])
            else:
                writer.writerow([row['Description'],
                                 row['Classificació'].strip(), row['Classificació'].strip(), ''])

    # Generació del dataset amb stopwords
    df = pd.read_excel(RAW_DATA)

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

    with open('res/data_full.csv', 'w') as w_file:
        writer = csv.writer(w_file)
        writer.writerow(['Description', 'Classificació',
                         'Classificació_01', 'Classificació_02'])
        for index, row in df.iterrows():
            aux = row['Classificació'].split('|')
            if len(aux) > 1:
                writer.writerow([row['Description'], row['Classificació'].strip(
                ), aux[0].strip(), aux[1].strip()])
            else:
                writer.writerow([row['Description'],
                                 row['Classificació'].strip(), row['Classificació'].strip(), ''])
