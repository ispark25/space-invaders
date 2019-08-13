import math
import numpy as np

from keras.layers import Input, LSTM, Dense, TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import losses
from keras.optimizers import Adam

import tensorflow as tf

import mdn
# for tensorboard
from datetime import datetime
from keras.callbacks import TensorBoard
import json


Z_DIM = 64 # 32
ACTION_DIM = 6 # 3

HIDDEN_UNITS = 256
GAUSSIAN_MIXTURES = 5

Z_FACTOR = 1
REWARD_FACTOR = 1
#RESTART_FACTOR = 0

LEARNING_RATE = 0.001
# MIN_LEARNING_RATE = 0.001
# DECAY_RATE = 1.0
LOGDIR = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
HIST = "logs/history/"


class RNN():
    def __init__(self): #, learning_rate = 0.001

    self.z_dim = Z_DIM
    self.action_dim = ACTION_DIM
    self.latent_dims = Z_DIM + ACTION_DIM + 1 # z(64), action(6), reward(1)
    self.output_dims = Z_DIM + ACTION_DIM + 1 + 1 # z(64), action(6), reward(1), done(1)
    self.hidden_units = HIDDEN_UNITS
    self.gaussian_mixtures = GAUSSIAN_MIXTURES
    #self.restart_factor = RESTART_FACTOR
    self.reward_factor = REWARD_FACTOR
    self.learning_rate = LEARNING_RATE

    self.models = self._build()
    self.model = self.models[0]
    self.decoder = self.models[1]
    self.tensorboard = TensorBoard(log_dir=LOGDIR)
    self.history = []

    # self.forward = self.models[1]


    def _build(self):

        inputs = Input(shape=(None, self.latent_dims), name = 'inputs')
        # # rescale input to be between 0 and 1
        # s =
        lstm_out = LSTM(self.hidden_units, name = 'lstm', return_sequences = True)(s)
        mdn_out = TimeDistributed(mdn.MDN(self.output_dims, GAUSSIAN_MIXTURES, name='mdn_outputs'), name = 'td-mdn')(lstm_out)

        modelM = Model(inputs = inputs, outputs = mdn_out)
        modelM.compile(loss = mdn.get_mixture_loss_func(self.output_dims, GAUSSIAN_MIXTURES), optimizer='adam')
        modelM.summary()

        decoder = keras.Sequential()
        decoder.add(keras.layers.LSTM(self.hidden_units, batch_input_shape=(1,1,self.output_dims), return_sequences = True, stateful = True))
        decoder.add(mdn.MDN(self.output_dims, GAUSSIAN_MIXTURES, name='mdn_outputs'))
        # decoder.compile(loss = mdn.get_mixture_loss_func(self.output_dims, GAUSSIAN_MIXTURES), optimizer='adam')
        # decoder.summary()

        # #### THE MODEL THAT WILL BE TRAINED
        # rnn_x = Input(shape=(None, Z_DIM + ACTION_DIM + 1))
        # lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state = True)

        # lstm_output_model, _ , _ = lstm(rnn_x)
        # mdn = Dense(GAUSSIAN_MIXTURES * (3*Z_DIM) + 1)

        # mdn_model = mdn(lstm_output_model)

        # model = Model(rnn_x, mdn_model)

        # #### THE MODEL USED DURING PREDICTION
        # state_input_h = Input(shape=(HIDDEN_UNITS,))
        # state_input_c = Input(shape=(HIDDEN_UNITS,))

        # lstm_output_forward , state_h, state_c = lstm(rnn_x, initial_state = [state_input_h, state_input_c])

        # mdn_forward = mdn(lstm_output_forward)

        # forward = Model([rnn_x] + [state_input_h, state_input_c], [mdn_forward, state_h, state_c])

        #### LOSS FUNCTION

        # def rnn_z_loss(y_true, y_pred):

        # 	z_true, rew_true = self.get_responses(y_true)

        # 	d = GAUSSIAN_MIXTURES * Z_DIM
        # 	z_pred = y_pred[:,:,:(3*d)]
        # 	z_pred = K.reshape(z_pred, [-1, GAUSSIAN_MIXTURES * 3])

        # 	log_pi, mu, log_sigma = self.get_mixture_coef(z_pred)

        # 	flat_z_true = K.reshape(z_true,[-1, 1])

        # 	z_loss = log_pi + self.tf_lognormal(flat_z_true, mu, log_sigma)
        # 	z_loss = -K.log(K.sum(K.exp(z_loss), 1, keepdims=True))

        # 	z_loss = K.mean(z_loss)

        # 	return z_loss

        # def rnn_rew_loss(y_true, y_pred):

        # 	z_true, rew_true = self.get_responses(y_true) #, done_true

        # 	d = GAUSSIAN_MIXTURES * Z_DIM
        # 	reward_pred = y_pred[:,:,-1]

        # 	rew_loss =  K.binary_crossentropy(rew_true, reward_pred, from_logits = True)

        # 	rew_loss = K.mean(rew_loss)

        # 	return rew_loss

        # def rnn_loss(y_true, y_pred):

        # 	z_loss = rnn_z_loss(y_true, y_pred)
        # 	rew_loss = rnn_rew_loss(y_true, y_pred)

        # 	return Z_FACTOR * z_loss + REWARD_FACTOR * rew_loss

        # opti = Adam(lr=LEARNING_RATE)
        # model.compile(loss=rnn_loss, optimizer=opti, metrics = [rnn_z_loss, rnn_rew_loss]) #, rnn_done_loss
        # model.compile(loss=rnn_loss, optimizer='rmsprop', metrics = [rnn_z_loss, rnn_rew_loss, rnn_done_loss])

        # return (model,forward)

        return (modelM, decoder)

    def build_decoder(self, rnn_weights='world_model/rnn/weights.h5'):
        self.decoder.compile(loss = mdn.get_mixture_loss_func(self.output_dims, GAUSSIAN_MIXTURES), optimizer='adam')
        self.decoder.summary()
        self.decoder.load_weights(rnn_weights)

    def sample_decoder(self, input, temp, sigma_temp):
        params = self.decoder.predict(input)
        return mdn.sample_from_output(params[0], self.output_dims, self.gaussian_mixtures, temp=temp, sigma_temp=sigma_temp)

    def rest_decoder_states(self):
        self.decoder.reset_states()

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, rnn_input, rnn_output,vrnn_input, vrnn_output, step):

        checkpoint = ModelCheckpoint(LOGDIR+'/step_{}.h5'.format(step), save_weights_only=True, verbose=1, save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

        hist = self.model.fit(rnn_input, rnn_output,
        shuffle=False,
        epochs=1,
        batch_size=len(rnn_input),
        callbacks=[self.tensorboard, checkpoint, early_stopping],
        validation_data=(vrnn_input, vrnn_output))

        # print(hist.history.keys())
        print(hist.history)
        self.history.append(hist.history)


    def save_history(self, filepath):
        with open(filepath, "w") as outfile:
        json.dump(self.history, outfile)


    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def get_responses(self, y_true):

        z_true = y_true[:,:,:Z_DIM]
        rew_true = y_true[:,:,-1]
        done_true = y_true[:,:,(Z_DIM + 1):]

        return z_true, rew_true, done_true


    def get_mixture_coef(self, z_pred):

        log_pi, mu, log_sigma = tf.split(z_pred, 3, 1)
        log_pi = log_pi - K.log(K.sum(K.exp(log_pi), axis = 1, keepdims = True)) # axis 1 is the mixture axis

        return log_pi, mu, log_sigma


    def tf_lognormal(self, z_true, mu, log_sigma):

        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        return -0.5 * ((z_true - mu) / K.exp(log_sigma)) ** 2 - log_sigma - logSqrtTwoPI
