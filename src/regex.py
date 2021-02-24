import re
import pandas as pd

def clean():
    pattern = re.compile('\{(.*?)\}')
    ruta = ''
    df = pd.read_excel(ruta)
    nombre_columna = ''
    df[nombre_columna] = df[nombre_columna].apply(replace(pattern, ' '))

    df.to_excel("clean_" + ruta)
