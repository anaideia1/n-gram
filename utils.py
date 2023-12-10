from ua_gec import Corpus, AnnotationLayer
import re

regex = "[А-ЯІЄЇҐа-яієїґ\']+"


def get_corpus_data(data_type='all'):
    corpus = Corpus(data_type, AnnotationLayer.GecOnly)
    return [re.findall(regex, doc.target.lower()) for doc in corpus]


def get_all_corpus_data():
    return get_corpus_data('all')


def get_train_corpus_data():
    return get_corpus_data('train')


def get_test_corpus_data():
    return get_corpus_data('test')

