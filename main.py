from ua_gec import Corpus, AnnotationLayer
import re

regex = "[а-яієїґ\']+"

if __name__ == '__main__':
    corpus = Corpus('train', AnnotationLayer.GecOnly)

    for doc in corpus:
        items = re.findall(regex, doc.target)
        print(items)
        break