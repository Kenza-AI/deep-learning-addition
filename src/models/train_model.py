# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, unicode_literals

import logging
import os

import click
from keras.models import Sequential
from keras import layers
import numpy as np

from src.settings import HIDDEN_SIZE, MAXLEN, RNN, DIGITS, LAYERS, BATCH_SIZE, REVERSE, CHARS, \
    CTABLE

logger = logging.getLogger(__name__)


class Colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


def _build_model():
    print('Build model...')
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(CHARS))))
    # As the decoder RNN's input, repeatedly provide with the last hidden state of
    # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    model.add(layers.RepeatVector(DIGITS + 1))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(LAYERS):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    model.add(layers.TimeDistributed(layers.Dense(len(CHARS))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    return model


def _train_model(model, x_train, y_train, x_val, y_val):
    # Train the model each generation and show predictions against the validation
    # dataset.
    for iteration in range(1, 15):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=1,
                  validation_data=(x_val, y_val))
        # Select 10 samples from the validation set at random so we can visualize
        # errors.
        for i in range(10):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowx, verbose=0)
            query = CTABLE.decode(rowx[0])
            correct = CTABLE.decode(rowy[0])
            guess = CTABLE.decode(preds[0], calc_argmax=False)
            print('Q', query[::-1] if REVERSE else query, end=' ')
            print('T', correct, end=' ')
            if correct == guess:
                print(Colors.ok + '☑' + Colors.close, end=' ')
            else:
                print(Colors.fail + '☒' + Colors.close, end=' ')
            print(guess)

    return model


def train(input_path, output_path):
    x_train = np.load(os.path.join(input_path, 'x_train.npy'))
    y_train = np.load(os.path.join(input_path, 'y_train.npy'))
    x_val = np.load(os.path.join(input_path, 'x_val.npy'))
    y_val = np.load(os.path.join(input_path, 'y_val.npy'))

    model = _build_model()

    trained_model = _train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val
    )

    trained_model.save(os.path.join(output_path, 'model.h5'))


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(input_path, output_path):
    train(input_path, output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
