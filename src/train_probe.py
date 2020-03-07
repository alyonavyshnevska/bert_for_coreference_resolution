import h5py
import sys
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping

if __name__ == "__main__":
    train_filename = sys.argv[1]
    val_filename = sys.argv[2]
    with h5py.File(train_filename, 'r') as f:
        train_data = f.get('span_representations').value

    with h5py.File(val_filename, 'r') as f:
        val_data = f.get('span_representations').value

    x_train = train_data[:,:-2]
    y_train = train_data[:,-1].astype(int)
    x_val = val_data[:,:-2]
    y_val = val_data[:,-1].astype(int)

    # Probing model implementation using keras, following hyperparameters described in Liu's paper, can finetune later.
    model = Sequential()
    model.add(Dense(units=1024, activation='relu', input_dim=x_train.shape[1], use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')) # still need to check whether 1024 is correct, little details about this.
    model.add(Dense(units=1, activation='sigmoid'))
    opt = optimizers.Adam(lr=0.001)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')]

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_val, y_val), callbacks=callbacks)
    loss_and_metrics = model.evaluate(x_val, y_val, batch_size=x_val.shape[0])
    print(loss_and_metrics)