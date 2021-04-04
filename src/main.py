# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import re
from envar import RAW_DATA

pattern_comments = re.compile(r'\(+(.*?)\)+')
pattern_ray = re.compile(r'\s*-+\s*')
pattern_multiple_dot = re.compile(r'\.\s(\.\s)+|\.\.+')
pattern_parentesis = re.compile(r'\(+|\)+')
pattenr_claudators = re.compile(r'\[+|\]+')
pattern_dot_space = re.compile(r'\.')
pattern_dot_m_space = re.compile(r'\.\s+')
pattern_m_spaces = re.compile(r'\s+')
pattern_question_marks = re.compile(r'\?+')


def noise_removal(txt):
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
    txt = txt.replace('/ RÈTOL/', ' ')
    txt = txt.replace('nan', '')

    if txt.isupper() or type(txt) != str or txt.isnumeric():
        txt = np.NAN
    else:
        txt = txt.strip()
        txt_aux = txt.lower()
        if 'suport directe' in txt_aux or 'suport directe amb' in txt_aux:
            txt = np.NAN

    return txt


if __name__ == "__main__":
    df = pd.read_excel(RAW_DATA)
    print("Mostres abans del preprocessament: {}".format(len(df)))

    df['Description'] = df['Description'].apply(noise_removal)
    df['Description'].replace('', np.NAN, inplace=True)
    df.dropna(subset=['Description'], inplace=True)
    df.dropna(subset=['Classificació'], inplace=True)
    df['Description'] = df['Description'].apply(noise_removal)
    df.drop_duplicates(inplace=True)
    df.drop(df.columns.difference(
        ['Description', 'Classificació']), 1, inplace=True)

    unique_cat = []
    aux = set()

    for s in df['Classificació']:
        unique_cat += (s.split('|'))

    for s in unique_cat:
        aux.add(s.strip())

    print(len(df['Classificació'].unique()))

    print("Mostres després del preprocessament: {}".format(len(df)))
    print("Nombre total de clases: {}".format(len(aux)))

    with open('res/clases.txt', 'w') as write_file:
        for clase in aux:
            write_file.write("{}\n".format(clase))
    df.to_csv('res/data.csv', index=False)
