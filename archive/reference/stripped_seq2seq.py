from __future__ import absolute_import
from recurrentshop import LSTMCell, RecurrentContainer
from keras.models import Sequential
from keras.layers import Dropout, Embedding


def StrippedSimpleSeq2Seq(output_dim, output_length, **kwargs):
    """"
    Simple model for sequence to sequence learning.
    The encoder encodes the input sequence to vector (called context vector)
    The decoder decodes the context vector in to a sequence of vectors.
    There is no one on one relation between the input and output sequence elements.
    The input sequence and output sequence may differ in length.

    Arguments:
        output_dim : Required output dimension.
        hidden_dim : The dimension of the internal representations of the model.
        output_length : Length of the required output sequence.
    """
    assert('input_dim' in kwargs)
    shape = (None, None, kwargs['input_dim'])
    del kwargs['input_dim']

    hidden_dim = output_dim

    encoder = RecurrentContainer(unroll=False, stateful=False, input_length=3)
    encoder.add(LSTMCell(hidden_dim, input_shape=(3, 64), **kwargs))

    decoder = RecurrentContainer(unroll=False, stateful=False, decode=True, output_length=output_length, input_length=3)
    decoder.add(Dropout(0., batch_input_shape=(shape[0], hidden_dim)))
    decoder.add(LSTMCell(output_dim, **kwargs))

    model = Sequential()
    # TODO: change [currently fixed] args [vocab_size, embed_dim, input_length] to variables!
    model.add(Embedding(1000, 64, input_length=3))
    model.add(encoder)
    model.add(decoder)
    return model
