# taken from this example notebook: https://github.com/cpmpercussion/keras-mdn-layer/blob/master/notebooks/MDN-RNN-time-distributed-MDN-training.ipynb
# set up MDN RNN
# inputs = keras.layers.Input(shape=(SEQ_LEN,OUTPUT_DIMENSION), name='inputs')
# lstm1_out = keras.layers.LSTM(HIDDEN_UNITS, name='lstm1', return_sequences=True)(inputs)
# lstm2_out = keras.layers.LSTM(HIDDEN_UNITS, name='lstm2', return_sequences=True)(lstm1_out)
# mdn_out = keras.layers.TimeDistributed(mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES, name='mdn_outputs'), name='td_mdn')(lstm2_out)
#
# model = keras.models.Model(inputs=inputs, outputs=mdn_out)
# model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,NUMBER_MIXTURES), optimizer='adam')
# model.summary()

# Decoding Model
# # Same as training model except for dimension and mixtures.
#
# decoder = keras.Sequential()
# decoder.add(keras.layers.LSTM(HIDDEN_UNITS, batch_input_shape=(1,1,OUTPUT_DIMENSION), return_sequences=True, stateful=True))
# decoder.add(keras.layers.LSTM(HIDDEN_UNITS, stateful=True))
# decoder.add(mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES))
# decoder.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,NUMBER_MIXTURES), optimizer=keras.optimizers.Adam())
# decoder.summary()
#
# #decoder.load_weights('kanji_mdnrnn_model_time_distributed.h5') # load weights independently from file
# #decoder.load_weights('kanji_mdnrnn-99.hdf5')
# decoder.load_weights('kanji_mdnrnn_model_time_distributed.h5')

our case:
		inputs = Input(shape=(None, self.latent_dims), name = 'inputs')
		lstm_out = LSTM(self.hidden_units, name = 'lstm', return_sequences = True)(inputs)
		mdn_out = TimeDistributed(mdn.MDN(self.output_dims, GAUSSIAN_MIXTURES, name='mdn_outputs'), name = 'td-mdn')(lstm_out)

		modelM = Model(inputs = inputs, outputs = mdn_out)
		modelM.compile(loss = mdn.get_mixture_loss_func(self.output_dims, GAUSSIAN_MIXTURES), optimizer='adam')
		modelM.summary()

decoder:
        decoder = keras.Sequential()
        decoder.add(keras.layers.LSTM(self.hidden_units, batch_input_shape=(1,1,self.output_dims), return_sequences = True, stateful = True))
        decoder.add(mdn.MDN(self.output_dims, GAUSSIAN_MIXTURES, name='mdn_outputs'))
        decoder.compile(loss = mdn.get_mixture_loss_func(self.output_dims, GAUSSIAN_MIXTURES), optimizer='adam')
        decoder.summary()
        decoder.load_weights('path/to/weights_4000data_b10_s38.h5')
		inputs = Input(shape=(None, self.latent_dims), name = 'inputs')
		lstm_out = LSTM(self.hidden_units, name = 'lstm', return_sequences = True)(inputs)
		mdn_out = TimeDistributed(mdn.MDN(self.output_dims, GAUSSIAN_MIXTURES, name='mdn_outputs'), name = 'td-mdn')(lstm_out)

		# modelM = Model(inputs = inputs, outputs = mdn_out)
		modelM.compile(loss = mdn.get_mixture_loss_func(self.output_dims, GAUSSIAN_MIXTURES), optimizer='adam')
		modelM.summary()
