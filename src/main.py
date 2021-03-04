# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import re

pattern_comments = re.compile(r'\(+(.*?)\)+')
pattern_comments_exhaustive = re.compile(r'\(+([\S\s]*)\)+')
pattern_ray = re.compile(r'\s*-+\s*')
pattern_whitespace = re.compile(r'\s{2,}')
pattern_multiple_dot = re.compile(r'\.\s(\.\s)+|\.\.+')
pattern_parentesis = re.compile(r'\(+|\)+')

unique_txt = set()


def noise_removal(txt):
    txt = str(txt).strip()
    txt = txt.replace('\n', ' ').replace('\r', '')
    txt = re.sub(pattern_comments, '', txt)
    txt = re.sub(pattern_ray, '', txt)
    txt = re.sub(pattern_whitespace, '', txt)
    txt = re.sub(pattern_multiple_dot, '', txt)
    txt = re.sub(pattern_parentesis, ' ', txt)

    if txt.isupper() or txt in unique_txt or type(txt) != str or txt.isnumeric():
        txt = np.nan
    else:
        unique_txt.add(txt)

    return txt


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise(ValueError("Numero de argumentos incorrecto"))
    else:
        df = pd.read_excel(sys.argv[1])
        print("Mostres abans del preprocessament: {}".format(len(df)))

        df['Description'] = df['Description'].apply(noise_removal)
        df['Description'].replace('', np.nan, inplace=True)
        df.dropna(subset=['Description'], inplace=True)
        df.dropna(subset=['Classificació'], inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop(df.columns.difference(
            ['Description', 'Classificació']), 1, inplace=True)

        unique_cat = []
        aux = set()

        for s in df['Classificació']:
          unique_cat += (s.split('|'))

        for s in unique_cat:
          aux.add(s.strip())

        print("Mostres després del preprocessament: {}".format(len(df)))
        print("Nombre total de clases: {}".format(len(aux)))

        df.to_csv('res/data_01.csv')
