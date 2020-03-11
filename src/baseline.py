from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Embedding
from keras_self_attention import SeqSelfAttention
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger, Callback
from sklearn.metrics import f1_score
import numpy as np
import h5py

import argparse



class ComputeTestF1(Callback):
    """Custom callback to calculate F1 score"""
    def on_epoch_end(self, epochs, logs):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_target = self.validation_data[1]
        logs['val_f1'] = f1_score(val_target, val_predict)


def get_args():
    parser = argparse.ArgumentParser(description='Run probing experiment for c2f-coref with BERT embeddings')
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    args = parser.parse_args()
    return args



def train_baseline_model(x_train, y_train, x_val, y_val, x_test, y_test, kernel_size=3):
    '''

        :param input_file: arrays of
    [all wordpieces of mention 1, all wordpieces of mention 2, distances, gold_label]
    The dimension of x_train is 948 x 46082.

    Each mention has dimension of 30*768

    Cnn + self-attention + probing model: trained end-to-end
    Example [mention1, mention2]:
    1. CNN is applied to mention 1 first then passed through self-attention.
    2. Same case with mention 2
    3. the headwords are input into the probing model

    Implementation of a baseline model for coreference resolution.

    Each mention has dimension of 30*768.

    Self-attention library:
    https://pypi.org/project/keras-self-attention/

    :param input_embeddings: all wordpieces of one mention of size 30*768

    :param kernel_size: kernel size of the convolutional layer kernel. Choose between 3 and 5.

    :param attention_width: The global context may be too broad for one piece of data.
    This parameter attention_width controls the width of the local context

    :return: mention_encoding

    '''

    mention1 = x_train[0]
    mention2 = x_train[1]

    # transpose (mention1,mention2)
    transposed = K.transpose([mention1, mention2])

    num_rows = transposed[0]
    num_cols = transposed[1]

    model = Sequential()
    model.add(Conv1D(14, kernel_size, strides=(1), padding='same', input_shape = (num_rows, num_cols)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dense(units=1024, activation='relu', input_dim=x_train.shape[1], use_bias=True, kernel_initializer='he_normal',
              bias_initializer='zeros'))  # still need to check whether 1024 is correct, little details about this.
    model.add(Dense(units=1, activation='sigmoid'))

    opt = optimizers.Adam(lr=0.001)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto'), ComputeTestF1(),
                 CSVLogger(log_name, separator='\t')]

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_val, y_val), callbacks=callbacks)
    val_loss_and_metrics = model.evaluate(x_val, y_val, batch_size=x_val.shape[0])
    print(val_loss_and_metrics)
    if args.test_data is not None:
        test_loss_and_metrics = model.evaluate(x_test, y_test, batch_size=x_test.shape[0])
        print(test_loss_and_metrics)



if __name__ == "__main__":

    x_test, y_test = None, None

    args = get_args()
    log_name = args.exp_name + '.log'

    with h5py.File(args.train_data, 'r') as f:
        train_data = f.get('span_representations').value
        x_train = train_data[:, :-2]
        y_train = train_data[:, -1].astype(int)

    with h5py.File(args.val_data, 'r') as f:
        val_data = f.get('span_representations').value
        x_val = val_data[:, :-2]
        y_val = val_data[:, -1].astype(int)

    if args.test_data is not None:
        with h5py.File(args.test_data, 'r') as f:
            test_data = f.get('span_representations').value
            x_test = test_data[:, :-2]
            y_test = test_data[:, -1].astype(int)


    train_baseline_model(x_train, y_train, x_val, y_val, x_test, y_test, kernel_size=3)