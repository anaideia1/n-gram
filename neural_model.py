import numpy as np
import pandas as pd
import os

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from utils import get_all_corpus_data


class NeuralModel:
    MODEL_NAME = 'word_generator'
    SEQ_LENGTH = 30
    N_GEN_CHARS = 20

    def __init__(self, data_text):
        self.model = None
        self.mapping = None
        x_tr, x_val, y_tr, y_val, vocab = self.prepare_data(data_text)
        self.vocab = vocab
        self.create_and_train_model(x_tr, x_val, y_tr, y_val)



    def prepare_data(self, data_text):
        # preprocess the text
        data_text = [word for sentence in data_text for word in sentence]
        data_new = self.text_cleaner(data_text)
        text = ' '.join(data_new)
        # create sequences
        sequences = self.create_seq(text)

        # create a character mapping index
        chars = sorted(list(set(text)))
        self.mapping = dict((c, i) for i, c in enumerate(chars))
        # encode the sequences
        sequences = self.encode_seq(sequences)

        # vocabulary size
        vocab = len(self.mapping)
        sequences = np.array(sequences)
        # create X and y
        x, y = sequences[:, :-1], sequences[:, -1]
        # one hot encode y
        y = to_categorical(y, num_classes=vocab)
        # create train and validation sets
        x_tr, x_val, y_tr, y_val = train_test_split(
            x, y, test_size=0.1, random_state=42
        )
        print('Train shape:', x_tr.shape, 'Val shape:', x_val.shape)
        return x_tr, x_val, y_tr, y_val, vocab


    @staticmethod
    def text_cleaner(text_data):
        long_words = []
        # remove short word
        for i in text_data:
            if len(i) >= 3:
                long_words.append(i)
        return long_words

    @staticmethod
    def create_seq(text):
        length = 30
        sequences = list()
        for i in range(length, len(text)):
            # select sequence of tokens
            seq = text[i - length:i + 1]
            # store
            sequences.append(seq)
        print('Total Sequences: %d' % len(sequences))
        return sequences

    def encode_seq(self, seq):
        sequences = list()
        for line in seq:
            # integer encode line
            encoded_seq = [self.mapping[char] for char in line]
            # store
            sequences.append(encoded_seq)
        return sequences


    def create_and_train_model(self, x_tr, x_val, y_tr, y_val):
        # define model
        if os.path.exists(self.MODEL_NAME):
            self.model = load_model(self.MODEL_NAME)
        else:
            self.model = Sequential()
            self.model.add(Embedding(self.vocab, 50, input_length=30, trainable=True))
            self.model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
            self.model.add(Dense(self.vocab, activation='softmax'))
            print(self.model.summary())

            # compile the model
            self.model.compile(
                loss='categorical_crossentropy', metrics=['acc'],optimizer='adam'
            )

            self.model.fit(
                x_tr, y_tr, epochs=10, verbose=2, validation_data=(x_val, y_val)
            )
            self.model.save(self.MODEL_NAME)


    def generate_seq(self, seed_text):
        in_text = seed_text
        # generate a fixed number of characters
        for _ in range(self.N_GEN_CHARS):
            # encode the characters as integers
            encoded = [self.mapping[char] for char in in_text]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=self.SEQ_LENGTH,
                                    truncating='pre')
            # predict character
            # yhat = self.model.predict_classes(encoded, verbose=0)
            yhat = np.argmax(self.model.predict(encoded) , axis=1)
            # reverse map integer to character
            out_char = ''
            for char, index in self.mapping.items():
                if index == yhat:
                    out_char = char
                    break
            # append to input
            in_text += out_char
        return in_text


def main():
    data = get_all_corpus_data()
    my_model = NeuralModel(data)

    test_data = '_'
    print('Програма генерації наступних символів побудована на натренованії нейронній мережі')
    print('Для того, щоб вийти введіть пусте значення')
    while test_data:
        test_data = input('Введіть початкове значення: ')
        if len(test_data) >= 5:
            print("Можливі вірогідності наступного слова:")
            res = my_model.generate_seq(test_data)
            print(res)
        elif test_data:
            print('Значення повинно складатися як мінімум з 5 символів.')
        else:
            print('До побачення!')


if __name__ == '__main__':
    main()