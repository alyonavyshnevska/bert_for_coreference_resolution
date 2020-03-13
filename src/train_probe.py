import h5py
import sys
import tensorflow as tf
import numpy as np
import argparse
import glob

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger, Callback
from sklearn.metrics import f1_score


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


if __name__ == "__main__":
    args = get_args()
    log_name = args.exp_name + '.log'

    filenames = glob.glob(args.train_data + "/*.h5")
    train_data = []
    print(filenames)
    for fn in filenames:
        with h5py.File(fn, 'r') as f:
            train_data.append(f.get('span_representations').value)
    train_data = np.concatenate(train_data, axis=0)
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

    # Probing model implementation using keras, following hyperparameters described in Liu's paper, can finetune later.
    model = Sequential()
    model.add(Dense(units=1024, activation='relu', input_dim=x_train.shape[1], use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')) # still need to check whether 1024 is correct, little details about this.
    model.add(Dense(units=1, activation='sigmoid'))
    opt = optimizers.Adam(lr=0.001)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto'), ComputeTestF1(), CSVLogger(log_name, separator='\t')]

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_val, y_val), callbacks=callbacks)
    val_loss_and_metrics = model.evaluate(x_val, y_val, batch_size=x_val.shape[0])
    print(val_loss_and_metrics)
    if args.test_data is not None:
        test_loss_and_metrics = model.evaluate(x_test, y_test, batch_size=x_test.shape[0])
        print(test_loss_and_metrics)
