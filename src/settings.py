# Parameters for the model and dataset.
from keras import layers

from src.character_encoder import CharacterTable

TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS

RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

# All the numbers, plus sign and space for padding.
CHARS = '0123456789+ '
CTABLE = CharacterTable(CHARS)
