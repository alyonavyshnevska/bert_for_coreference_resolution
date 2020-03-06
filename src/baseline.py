from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Embedding
from keras_self_attention import SeqSelfAttention
import numpy as np


def baseline_model(input_embeddings, kernel_size=3, attention_width=15):
    '''

    Implementation of a baseline model for coreference resolution.
    Span Embeddings are input.
    A 2d convolutional layer is applied.
    Self-attention later is applied.
    The output of the model is ready to be fed into a FFNN probing model.

    Self-attention library:
    https://pypi.org/project/keras-self-attention/

    :param input_embeddings: 4D tensor with shape: (batch, channels, rows, cols) if data_format is "channels_first"
    or 4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last" (second option is default)

    :param kernel_size: kernel size of the convolutional layer kernel. Choose between 3 and 5.

    :param attention_width: The global context may be too broad for one piece of data. The parameter attention_width controls the width of the local context

    :return: model

    '''

    num_rows = input_embeddings.shape[1]
    num_cols = input_embeddings.shape[2]
    channels = input_embeddings.shape[3]

    model = Sequential()
    model.add(Conv2D(14, kernel_size, strides=(2, 2), padding='valid', input_shape = (num_rows, num_cols, channels)))
    model.add(SeqSelfAttention(attention_width=attention_width,
                               attention_activation='sigmoid'))
    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])
    model.summary()
    return model


if __name__ == "__main__":
    batch_size = 16
    input_embeddings = np.random.rand(batch_size, 50, 30, 5)   #randomly chosen input data (batch, rows, cols, channels)
    model = baseline_model(input_embeddings, kernel_size=3, attention_width=15)