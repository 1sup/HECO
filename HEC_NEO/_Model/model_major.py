import time
from keras.callbacks import Callback
from keras.models import Input, Model, Sequential                     #### TCN
from keras.layers import Dense, Activation, Conv1D, LSTM, Dropout, Reshape, Bidirectional, Flatten, Add, Concatenate, MaxPool1D, LeakyReLU, GRU
import keras.backend as K

from _Support.support_NLSTM import NestedLSTM


class TimeHistory(Callback):
	def on_train_begin(self, logs={}):
		self.times = []
		self.totaltime = time.time()
	
	def on_train_end(self, logs={}):
		self.totaltime = time.time() - self.totaltime
	
	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()
	
	def on_epoch_end(self, batch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)


def build_GRU(timestep):
	
	batch_size, timesteps, input_dim = None, timestep, 1
	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	
	x = GRU(64)(i)
	x = Dense(16, activation='linear')(x)
	o = Dense(1, activation="linear")(x)
	
	model = Model(inputs=[i], outputs=[o])
	model.compile(optimizer='rmsprop', loss='mse', )
	model.summary()
	
	return model


def build_LSTM(timestep):
	
	batch_size, timesteps, input_dim = None, timestep, 1
	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	
	x = LSTM(64)(i)
	x = Dense(16, activation='linear')(x)
	o = Dense(1, activation="linear")(x)
	
	model = Model(inputs=[i], outputs=[o])
	model.compile(optimizer='rmsprop', loss='mse', )
	model.summary()
	
	return model


def build_SLSTM(timestep):
	batch_size, timesteps, input_dim = None, timestep, 1
	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	
	x = Reshape((-1, 1))(i)
	x = LSTM(64, return_sequences=True)(x)
	x = LSTM(64, return_sequences=True)(x)
	x = LSTM(64)(x)
	x = Dense(16, activation='linear')(x)
	o = Dense(1, activation="linear")(x)
	
	model = Model(inputs=[i], outputs=[o])
	
	model.compile(optimizer='rmsprop', loss='mse', )
	model.summary()
	return model


def build_BiLSTM(timestep):
	
	batch_size, timesteps, input_dim = None, timestep, 1
	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	
	x = Bidirectional(LSTM(64), merge_mode='concat')(i)
	x = Dense(16, activation='linear')(x)
	o = Dense(1, activation="linear")(x)
	
	model = Model(inputs=[i], outputs=[o])
	
	model.compile(optimizer='Adam', loss='mse', )
	model.summary()
	
	return model


def build_NLSTM(timestep):
	
	batch_size, timesteps, input_dim = None, timestep, 1
	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	
	x = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(i)
	x = Dense(16, activation='linear')(x)
	o = Dense(1, activation="linear")(x)
	
	model = Model(inputs=[i], outputs=[o])
	model.compile(optimizer='rmsprop', loss='mse', )
	model.summary()
	
	return model
