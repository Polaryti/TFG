import re
from envar import CATALAN_STOPWORDS, CATALAN_APOSTROFS
from string import digits
import numpy as np
import copy

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



def apostrofs_clean(sample):
    for apostrof in CATALAN_APOSTROFS:
        sample = sample.replace(apostrof, '')

    return sample


def stopwords_removal(sample):
    sample = sample.split()
    for stopword in CATALAN_STOPWORDS:
        sample = list(filter((stopword).__ne__, sample))

    return ' '.join(sample)


def preprocess_data(data, apostrofs: bool, stopwords: bool):
    if type(data) is list:
        processed_data = []
        processed_sample = None
        for sample in data:
            processed_sample = preprocess(sample, apostrofs, stopwords)
            if processed_sample != None:
                processed_data.append(processed_sample)

        return processed_data
    else:
        return preprocess(data, apostrofs, stopwords)


def preprocess(sample, apostrofs: False, stopwords: False):
    if type(sample) is np.NaN or type(sample) is not str or sample.isnumeric():
        return None
    processed_sample = copy.deepcopy(sample)
    processed_sample = processed_sample.strip().lower()

    if apostrofs:
        processed_sample = apostrofs_clean(processed_sample)
    if stopwords:
        processed_sample = stopwords_removal(processed_sample)

    return processed_sample


print(preprocess_data("Hola que tal hi estàs? El meu nom hi és l'Mario del condats, dels Arago'ns. Doncs bé, adeu-hi.", True, True))