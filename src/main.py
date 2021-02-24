from input import Input
import pandas as pd
import sys
import re

def clean_description(df):
    pattern = re.compile(r'\{(.*?)\}')
    nombre_columna = 'Description'
    return df[nombre_columna].apply(re.sub(pattern, ' ',))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise(Exception("Numero de argumentos incorrecto"))
    else:
        df = pd.read_excel(sys.argv[1])
        df = clean_description(df)


        