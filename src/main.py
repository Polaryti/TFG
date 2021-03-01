# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import re

pattern_comments = re.compile(r'\(+(.*?)\)+')
pattern_ray = re.compile(r'\s*-+\s*')
pattern_whitespace = re.compile(r'\s{2,}')
pattern_dot = re.compile(r'\.\s*')

def noise_removal(txt):
    txt = str(txt).strip()
    txt = txt.replace('\n', ' ').replace('\r', '')
    txt = re.sub(pattern_comments, '', txt)
    txt = re.sub(pattern_ray, '', txt)
    txt = re.sub(pattern_whitespace, '', txt)
    txt = re.sub(pattern_dot, '. ', txt)
    return txt

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise(Exception("Numero de argumentos incorrecto"))
    else:
        df = pd.read_excel(sys.argv[1])
        df['Description'] = df['Description'].apply(noise_removal)
        # df['Description'].replace('', np.nan, inplace=True)
        # df.dropna(subset=['Description'], inplace=True)

        for d in df['Description'].values:
            print(d)


        
