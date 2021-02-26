from input import Input
import pandas as pd
import sys
import re

pattern_comments = re.compile(r'\(+(.*?)\)+')
pattern_ray = re.compile(r'-+')

def clean_description(txt):
    return re.sub(pattern, '-', str(txt))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise(Exception("Numero de argumentos incorrecto"))
    else:
        df = pd.read_excel(sys.argv[1])
        df['Description'].apply(clean_description)


        
