import re
from envar import CATALAN_STOPWORDS, CATALAN_APOSTROFS
from string import digits
import numpy as np

pattern_comments = re.compile(r'\(+(.*?)\)+')
pattern_ray = re.compile(r'\s*-+\s*')
pattern_multiple_dot = re.compile(r'\.\s(\.\s)+|\.\.+')
pattern_parentesis = re.compile(r'\(+|\)+')
pattenr_claudators = re.compile(r'\[+|\]+')
pattern_dot_space = re.compile(r'\.')
pattern_dot_m_space = re.compile(r'\.\s+')
pattern_m_spaces = re.compile(r'\s+')
pattern_question_marks = re.compile(r'\?+')
pattern_signs = re.compile(r"[¿?¡!@*+-/\\#$\.,'%&:\"|]")
pattern_news = re.compile(r'$\s*new .*$')
pattern_html = re.compile(r'<.*?>')

remove_digits = str.maketrans('', '', digits)


def apostrofs_clean(sample):
    if type(sample) != str or sample.isnumeric():
        return None

    for apostrof in CATALAN_APOSTROFS:
        sample = sample.replace(apostrof, '')

    sample = sample.strip()
    if sample == '':
        return None
    else:
        return sample


def stopwords_removal(sample):
    if type(sample) != str or sample.isnumeric():
        return None

    sample = sample.split()
    for stopword in CATALAN_STOPWORDS:
        sample = list(filter((stopword).__ne__, sample))

    sample = ' '.join(sample)

    sample = sample.strip()
    if sample == '':
        return None
    else:
        return sample


def noise_clean(sample):
    if type(sample) != str or sample.isnumeric():
        return None

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
    sample = re.sub(pattern_html, '', sample)
    sample = re.sub(pattern_signs, '', sample)
    sample = sample.translate(remove_digits)
    sample = sample.replace('/ rètol/', ' ')
    sample = sample.replace('persona cargo informatius', '')
    sample = sample.replace('persona cargo esports', '')
    sample = sample.replace('new persona cargo', '')
    sample = sample.replace('cargo informatius new', '')
    sample = sample.replace('entradeta de seguretat', '')
    sample = re.sub(pattern_news, '', sample)
    sample = re.sub(pattern_m_spaces, ' ', sample)

    if 'suport directe' in sample or 'suport directe amb' in sample or 'fals directe' in sample or sample == 'dnc':
        return None

    sample = sample.strip()
    if sample == '':
        return None
    else:
        return sample


def preprocess_data(data, apostrofs: bool, stopwords: bool, noise: bool):
    if type(data) is list:
        processed_data = []
        processed_sample = None
        for sample in data:
            processed_sample = preprocess(sample, apostrofs, stopwords, noise)
            if processed_sample is not None:
                processed_data.append(processed_sample.strip())

        return processed_data
    else:
        return preprocess(data, apostrofs, stopwords, noise)


def preprocess(sample, apostrofs: True, stopwords: True, noise: True):
    if type(sample) is not str or sample.isnumeric():
        return np.NaN
    else:
        processed_sample = sample.strip().lower()

    if apostrofs:
        processed_sample = apostrofs_clean(processed_sample)
        if processed_sample is None:
            return np.NaN
    if stopwords:
        processed_sample = stopwords_removal(processed_sample)
        if processed_sample is None:
            return np.NaN
    if noise:
        processed_sample = noise_clean(processed_sample)
        if processed_sample is None:
            return np.NaN

    processed_sample = processed_sample.strip()
    if processed_sample == '':
        return np.NaN
    else:
        return processed_sample


def clean_all(sample):
    return preprocess_data(sample, True, True, True)


def clean_no_stopwords(sample):
    return preprocess_data(sample, True, False, True)
