from src.settings import CTABLE, MAXLEN


def encode_query(input_string):
    """
    Encode a query addition string
    :param input_string: [str], input query string, i.e. '123+456'
    :return: [str], encoded query ready to be used in model.predict(...)
    """
    output = CTABLE.encode(input_string[::-1], MAXLEN)

    return output.reshape((1, output.shape[0], output.shape[1]))


def decode_prediction(input_array):
    """
    Decode model prediction
    :param input_array: [numpy.array], input numpy array
    :return: [str], decoded array to string
    """
    return CTABLE.decode(input_array[0])
