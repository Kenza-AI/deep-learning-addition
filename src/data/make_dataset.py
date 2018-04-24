# -*- coding: utf-8 -*-
from __future__ import unicode_literals, absolute_import

import logging
import os

import click
import numpy as np

from src.models.train_model import CHARS
from src.settings import TRAINING_SIZE, DIGITS, REVERSE, MAXLEN, CTABLE

logger = logging.getLogger(__name__)


def _generate_data():
    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < TRAINING_SIZE:
        def func():
            return int(''.join(np.random.choice(list('0123456789'))
                               for _ in range(np.random.randint(1, DIGITS + 1))))

        a, b = func(), func()
        # Skip any addition questions we've already seen
        # Also skip any such that x+Y == Y+x (hence the sorting).
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        # Pad the data with spaces such that it is always MAXLEN.
        q = '{}+{}'.format(a, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a + b)
        # Answers can be of maximum size DIGITS + 1.
        ans += ' ' * (DIGITS + 1 - len(ans))
        if REVERSE:
            # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
            # space used for padding.)
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))

    return questions, expected


def _vectorization(questions, expected):
    print('Vectorization...')

    x = np.zeros((len(questions), MAXLEN, len(CHARS)), dtype=np.bool)
    y = np.zeros((len(questions), DIGITS + 1, len(CHARS)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = CTABLE.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = CTABLE.encode(sentence, DIGITS + 1)

    # Shuffle (x, y) in unison as the later parts of x will almost all be larger
    # digits.
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)

    return x_train, y_train, x_val, y_val


@click.command()
@click.argument('output_path', type=click.Path())
def main(output_path):
    """ Runs data processing scripts to save data in ../processed.
    """
    logger.info('making final data set')

    questions, expected = _generate_data()

    x_train, y_train, x_val, y_val = _vectorization(questions, expected)

    np.save(os.path.join(output_path, 'x_train.npy'), x_train)
    np.save(os.path.join(output_path, 'y_train.npy'), y_train)

    np.save(os.path.join(output_path, 'x_val.npy'), x_val)
    np.save(os.path.join(output_path, 'y_val.npy'), y_val)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
