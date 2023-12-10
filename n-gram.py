import random

from nltk import trigrams
from collections import defaultdict

from utils import get_all_corpus_data


class NGramModel:
    def __init__(self):
        # Create a placeholder for model
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

    def fit(self, data):
        # Count frequency of co-occurance
        for sentence in data:
            for w1, w2, w3 in trigrams(sentence, pad_right=True,
                                       pad_left=True):
                self.model[(w1, w2)][w3] += 1

        # Let's transform the counts to probabilities
        for w1_w2 in self.model:
            total_count = float(sum(self.model[w1_w2].values()))
            for w3 in self.model[w1_w2]:
                self.model[w1_w2][w3] /= total_count

    def probability_choices(self, first: str, second: str) -> dict:
        third = self.model[first, second]
        return dict(third)

    def generate(self, text: list) -> str:
        sentence_finished = False

        while not sentence_finished:
            # select a random probability threshold
            r = random.random()
            accumulator = .0

            for word in self.model[tuple(text[-2:])].keys():
                accumulator += self.model[tuple(text[-2:])][word]
                # select words that are above the probability threshold
                if accumulator >= r:
                    text.append(word)
                    break

            if text[-2:] == [None, None]:
                sentence_finished = True

        return ' '.join([t for t in text if t])


def main():
    data = get_all_corpus_data()
    model = NGramModel()
    model.fit(data)

    test_data = '_'

    print('Програма генерації наступного слово побудована на методі N-Gram')
    print('Для того, щоб вийти введіть пусте значення')
    while test_data:
        test_data = input('Введіть початкове значення: ')
        test_data = test_data.split()
        if len(test_data) >= 2:
            print("Можливі вірогідності наступного слова:")
            res = model.probability_choices(test_data[-2], test_data[-1])
            if res:
                print(res)
                gen_sentence = model.generate(test_data)
                print(gen_sentence)
            else:
                print('На жаль, такої комбінації не було в тестових даних :(')


        elif test_data:
            print('Значення повинно складатися як мінімум з двох слів.')
        else:
            print('До побачення!')


if __name__ == '__main__':
    main()