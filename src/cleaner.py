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
    if type(sample) != str or sample.isnumeric():
        sample = np.NaN

    for apostrof in CATALAN_APOSTROFS:
        sample = sample.replace(apostrof, '')

    return sample


def stopwords_removal(sample):
    if type(sample) != str or sample.isnumeric():
        sample = np.NaN

    sample = sample.split()
    for stopword in CATALAN_STOPWORDS:
        sample = list(filter((stopword).__ne__, sample))

    return ' '.join(sample)


def noise_clean(sample):
    if type(sample) != str or sample.isnumeric():
        sample = np.NaN

    sample = str(sample).strip()
    sample = sample.replace('\n', ' ').replace('\r', '')
    sample = re.sub(pattern_comments, ' ', sample)
    sample = re.sub(pattern_ray, ' ', sample)
    sample = re.sub(pattern_multiple_dot, '', sample)
    sample = re.sub(pattern_parentesis, ' ', sample)
    sample = re.sub(pattenr_claudators, ' ', sample)
    sample = re.sub(pattern_dot_space, '. ', sample)
    sample = re.sub(pattern_dot_m_space, '. ', sample)
    sample = re.sub(pattern_m_spaces, ' ', sample)
    sample = re.sub(pattern_question_marks, '?', sample)
    sample = re.sub(pattern_signs, '', sample)
    sample = sample.replace('/ RÈTOL/', ' ')
    sample = re.sub(pattern_news, '', sample)
    sample = sample.replace('persona cargo informatius', '')
    sample = sample.replace('persona cargo esports', '')
    sample = sample.replace('new persona cargo', '')
    sample = sample.replace('cargo informatius new', '')

    if type(sample) != str or sample.isnumeric():
        sample = np.NaN
    else:
        if 'suport directe' in sample or 'suport directe amb' in sample or 'fals directe' in sample:
            sample = np.NaN
        if sample == "dnc":
            sample = np.NaN

    return sample


def preprocess_data(data, apostrofs: bool, stopwords: bool, noise: bool):
    if type(data) is list:
        processed_data = []
        processed_sample = None
        for sample in data:
            processed_sample = preprocess(sample, apostrofs, stopwords, noise)
            if processed_sample is not None:
                processed_data.append(processed_sample)

        return processed_data
    else:
        return preprocess(data, apostrofs, stopwords, noise)


def preprocess(sample, apostrofs: True, stopwords: True, noise: True):
    if type(sample) is np.NaN or type(sample) is not str or sample.isnumeric():
        return None
    processed_sample = copy.deepcopy(sample)
    processed_sample = processed_sample.strip().lower()

    if apostrofs:
        processed_sample = apostrofs_clean(processed_sample)
    if stopwords:
        processed_sample = stopwords_removal(processed_sample)
    if noise:
        processed_sample = noise_clean(processed_sample)

    return processed_sample


def clean_all(sample):
    return preprocess_data(sample, True, True, True)


def clean_no_stopwords(sample):
    return preprocess_data(sample, True, False, True)
