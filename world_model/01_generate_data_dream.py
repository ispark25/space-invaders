import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize, rescale
from skimage.io import imsave
import random
import gym
from gym import spaces

from world_model.vae.arch_surprise_lower_tolerance import VAE, load_vae
from world_model.rnn.arch_v5_cumrew_only_no_action import RNN

import argparse
import os

import matplotlib.pyplot as plt

ENV_NAME = 'SpaceInvaders-v0'
DIR_NAME = './world_model/' # './data/vae_food_3/'
# MIN_LENGTH = 400 # agent should survive for at least this many frames

DEATH_OFFSET = 40 #32-45
DEATH_PENALTY = -500

TEMPERATURE = 0.8

SEED = 0
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
from numpy.random import seed as npseed
npseed(SEED) #set seed for numpy
import tensorflow as tf
tf.set_random_seed(SEED) #set random seed for keras
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
# https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras

# Custom Space Invaders Environment made from our VAE+MDNRNN models
# help from source code WM: https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/doomrnn.py
class SpaceInvadersRNNEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
	}
	def __init__(self, render_mode = False):
		self.render_mode = render_mode

		# get initial mu and initial lv
		with open(DIR_NAME + "data/initial_z.npz", 'rb') as f:
			inits = np.load(f)
			self.initial_mu_logvar = [list(elem) for elem in zip(inits["init_mus"], inits["init_lvs"])]

		# initialise and load VAE and RNN
		self.vae = load_vae("world_model/vae/weights/arch_surprise_medium_tolerance.h5")
		self.rnn = RNN()
		# self.rnn.decoder.load_weights("world_model/rnn/weights_final_20190816-121335.h5")
		# self.rnn.decoder.load_weights("world_model/rnn/weights.h5step_7400.h5")
		self.rnn.decoder.load_weights("world_model/rnn/renders-hist-20190818-042041.json/weights.h5")

		self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
		self.out_width = self.rnn.output_dims
		self.obs_size = self.out_width + self.rnn.hidden_units * 2 # (includes C and H)
		self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(self.obs_size,), dtype=np.float32)

		self.zero_state = np.zeros(self.rnn.hidden_units)

		self.rnn_hidden = None
		self.rnn_cell = None
		self.z =None
		self.done = None
		self.cum_reward = 0
		self.temperature = None

		self.frame_count = None
		self.max_frame = 2100

		self.viewer = None

		self.reset()

	def _sample_init_z(self):
		idx = np.random.randint(0, len(self.initial_mu_logvar))
		init_mu, init_logvar = self.initial_mu_logvar[idx]
		print("mu:{}, lv:{}".format(init_mu, init_logvar))
		# init_mu = np.divide(np.array(init_mu),10000.)
		# init_logvar = np.divide(np.array(init_logvar),10000.)
		init_sb = np.shape(init_logvar)
		init_z = init_mu + np.exp(init_logvar/2.0) * np.random.randn(*init_sb)
		# init_z = init_mu + np.exp(init_logvar/2.0) * np.random.randn(*init_logvar.shape)
		return init_z

		# 	preprocessed = downsample(observation)
		# 	mu, lv = vae.encoder_mu_log_var.predict(np.array([preprocessed]))
		# 	sb = np.shape(lv)
		# 	next_z = mu + np.exp(lv/2.0) * np.random.randn(*sb)

	def _current_state(self):
		return np.concatenate([self.z, self.rnn_hidden.flatten(), self.rnn_cell.flatten()], axis=0)

	def step(self, action):
		self.frame_count += 1

		prev_z = np.zeros((1,self.rnn.z_dim))
		prev_z[0] = self.z.reshape((1,self.rnn.z_dim))

		prev_action = np.zeros((1,2))
		prev_action[0][0] = action[0]
		prev_action[0][1] = action[1]

		prev_cum_reward = np.ones((1, 1))
		prev_cum_reward[0] = self.cum_reward

		temperature = self.temperature
		print("After expanding, shapes are: z={}, action={}, reward={}".format(np.shape(prev_z[:,:64]),np.shape(prev_action),np.shape(prev_cum_reward)))

		print("z:{}".format(prev_z))
		print("action:{}".format(prev_action))
		print("r:{}".format(prev_cum_reward))
		input_to_rnn = [np.array([np.concatenate([prev_z[:,:64], prev_action, prev_cum_reward], axis=1)]), np.array([self.rnn_hidden]), np.array([self.rnn_cell])]
		next_z, next_rew, hidden, cell, next_done = self.rnn.sample_decoder(input_to_rnn, TEMPERATURE)
		print("After pred, shapes are: nz={} nreward={}".format(np.shape(next_z),np.shape(next_rew)))

		print("predictions: done: {}, reward: {}".format(next_done, next_rew))
		# if next_done > 0:
		# 	done = True
		# else:
		# 	done = False
		done = False

		reward = next_rew - prev_cum_reward

		self.z = next_z
		self.done = next_done
		self.rnn_state_hidden = hidden
		self.rnn_state_cell = cell
		self.cum_reward = next_rew

		if self.frame_count >= self.max_frame:
		  done = True

		return self._current_state(), reward, done, {}

	def reset(self):
		self.z = self._sample_init_z()
		self.done = 0
		self.frame_count = 0
		self.reward = 0
		self.cum_reward = 0
		self.temperature = TEMPERATURE
		self.rnn_hidden = self.zero_state
		self.rnn_cell = self.zero_state
		return self._current_state()

	def _get_image(self, upsize = False):
		img = self.vae.decoder.predict(self.z.reshape(1, 64))
		print(img)
		img = (img*255).astype(np.uint8)
		img = img.reshape(64, 64, 3)
		print(img)
		# if upsize:
		# 	img = rescale(img, 2.5, anti_aliasing=False)
		return img

	def render(self, mode='human', close=False):
		if not self.render_mode:
			return

		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		if mode == 'rbg_array':
			img = self._get_image(upsize = True)

		elif mode == 'human':
			img = self._get_image(upsize=True)
			from gym.envs.classic_control import rendering
			if self.viewer is None:
				self.viewer = rendering.SimpleImageViewer()
			self.viewer.imshow(img)
			imsave("screenshot1_{}.png".format(self.frame_count), img)
			print("saved {}.".format(self.frame_count))



def main(args):
	nb_episodes = args.nb_episodes
	time_steps  = args.time_steps
	render      = args.render
	# env = gym.make(ENV_NAME) # <1>
	env = SpaceInvadersRNNEnv(render_mode = True)

	if env.render_mode:
		from pyglet.window import key

	a = np.array([0.0, 0.5])

	def key_press(k, mod):
		global overwrite
		overwrite = True
		if k==key.LEFT:
			a[0] = -1.0
			print('human key left.')
		if k==key.RIGHT:
			a[0] = +1.0
			print('human key right.')
		if k==key.UP:
			a[1] = +1.0
			print('human key up (shoot).')


	def key_release(k, mod):
		if k in [key.LEFT, key.RIGHT]:
			a[0] = 0.
		elif k == key.UP:
			a[1] = 0.

	if env.render_mode:
		env.render()
		env.viewer.window.on_key_press = key_press
		env.viewer.window.on_key_release = key_release

	reward_list = []

	print("Generating data for env {}".format(ENV_NAME))
	s = 0

	for i in range(40):
		print("game:{}".format(i))
		env.reset()
		total_reward = 0.0
		steps = 0

		repeat = np.random.randint(1, 11)
		obs_list = []
		z_list = []
		obs = env.reset()
		obs_list.append(obs)
		z_list.append(obs[0:64])

		overwrite = True

		while True:
			print("step: {}".format(steps))
			if steps % repeat == 0:
				action1 = np.random.rand() * 2.0 - 1.0
				action2 = np.random.rand()
				repeat = np.random.randint(1, 11)
			action = 0.0
			if overwrite:
				action1 = a[0]
				action2 = a[1]
			obs, reward, done, info = env.step(np.array([action1, action2]))
			obs_list.append(obs)
			z_list.append(obs[0:64])
			total_reward += reward
			steps += 1

			if env.render_mode:
				env.render()
			if done:
				break

		reward_list.append(total_reward)

		print('cumulative reward', total_reward)
	print('average reward', np.mean(reward_list))
	env.close()

	# rnn = RNN()
	# rnn.decoder.load_weights("world_model/rnn/weights_final_20190816-121335.h5")
	#
	# vae = load_vae("world_model/vae/weights/arch_surprise_medium_tolerance.h5")
	#
	# def rollout(controller):
	#     obs = env.reset()
	#     h = rnn.initial_state()
	#     done = False
	#     cumulative_reward = 0
	#     while not done:
	#         z = vae.encode(obs)
	#         a = controller.action([z,h])
	#         obs, reward, done = env.step(a)
	#         cumulative_reward += reward
	#         h = rnn.forward([z,a,h])
	#
	# env = gym.make(ENV_NAME) # <1>
	#
	# s=0
	# while s < nb_episodes:
	# 	real = {"obs":[], "z":[], "action":[], "reward":[], "done":[]}
	# 	dream = {"obs":[], "z":[], "action":[], "reward":[], "done":[]}
	#
	# 	# obs_seq    = []
	# 	# action_seq = []
	# 	# reward_seq = []
	# 	# done_seq   = []
	#
	# 	observation = env.reset()
	# 	reward = 0
	# 	cumulative_reward = 0
	# 	done = False
	#
	# 	dobs = None
	# 	dreward = 0
	# 	dcumulative_reward = 0
	# 	ddone = 0
	#
	# 	hidden = np.zeros(rnn.hidden_units)
	# 	cell = np.zeros(rnn.hidden_units)
	#
	# 	# Player starts off with 3 lives
	# 	prev_lives = 3
	#
	# 	repeat = np.random.randint(1, 11)
	# 	t = 0
	#
	# 	#initial rollout
	# 	action = generate_data_action(t, env)  # <2>
	# 	repeat = np.random.randint(1, 11)
	#
	# 	real["obs"].append(observation)
	# 	real["action"].append(action)
	# 	real["reward"].append(reward)
	# 	real["done"].append(done)
	#
	# 	preprocessed = downsample(observation)
	# 	mu, lv = vae.encoder_mu_log_var.predict(np.array([preprocessed]))
	# 	np.savez_compressed('./world_model/data/' + 'initial_z.npz', init_mus=mu, init_lvs=lv) # TODO: what for?
	# 	sb = np.shape(lv)
	# 	next_z = mu + np.exp(lv/2.0) * np.random.randn(*sb)
	#
	# 	print(next_z)
	# 	print(np.shape(hidden))
	# 	print(np.shape(cell))
	# 	t+=1
	#
	# 	# for rollout test
	# 	h = rnn
	# 	while t < time_steps:
	# 		print("t={}".format(t))
	#
	# 		if t % repeat == 0:
	# 			action = generate_data_action(t, env)  # <2>
	# 			repeat = np.random.randint(1, 11)
	#
	# 		real["obs"].append(observation)
	# 		real["action"].append(action)
	# 		real["reward"].append(reward)
	# 		real["done"].append(done)
	#
	# 		if done:
	# 			print("REACHED DONE STATE")
	# 			break
	#
	# 		# Next
	# 		observation, reward, done, info = env.step(action)  # <4>
	# 		# Extra code to penalise for death
	# 		curr_lives = info['ale.lives']
	# 		if curr_lives < prev_lives:
	# 			reward_seq[max(0, t - DEATH_OFFSET)] += DEATH_PENALTY
	# 			cumulative_reward += DEATH_PENALTY
	# 		prev_lives = curr_lives
	#
	# 		# preprocessed = downsample(observation)
	# 		# # print(preprocessed)
	# 		#
	# 		# mu, lv = vae.encoder_mu_log_var.predict(np.array([preprocessed]))
	# 		# sb = np.shape(lv)
	# 		# next_z = mu + np.exp(lv/2.0) * np.random.randn(*sb)
	#
	# 		cumulative_reward += reward
	# 		def convert_y(y):
	# 			if y == 0 or y == 2 or y == 3:
	# 				return 0
	# 			else:
	# 				return 1
	#
	# 		def convert_x(x):
	# 			if x <= 1:
	# 				return 0
	# 			elif x%2 == 1:
	# 				return -1
	# 			else: #x%2 == 0
	# 				return 1
	# 		cont_action = [convert_x(int(action)),convert_y(int(action))]
	# 		# print("After expanding, shapes are: z={}, action={}, reward={}, done={}".format(np.shape(z[0]),np.shape(cont_action),np.shape([reward]),np.shape(done)))
	# 		input_to_rnn = [np.array([[np.concatenate([next_z[0], cont_action, [cumulative_reward]])]]), np.array([hidden]), np.array([cell])]
	# 		next_z, next_rew, hidden, cell, next_done = rnn.sample_decoder(input_to_rnn, 1)
	# 		# print("shape: {}".format(np.shape(next_z)))
	# 		print("predictions: done: {}, reward: {}".format(next_done, next_rew))
	# 		print("true:        done: {}, reward: {}".format(done, cumulative_reward))
	#
	#
	# 		if render:
	# 			env.render()
	# 		t = t + 1
	#
	# 	# Save episode data if it featured commandership
	# 	if t < 600: # commanders do not appear earlier than 600 steps
	# 		print("DISCARDED: Episode {} finished after {} timesteps".format(s, t))
	# 		continue
	#
	# 	obs_seq = np.array(obs_seq)
	# 	downsampled = batch_downsample(obs_seq)
	#
	# 	if downsampled is None:
	# 		print("DISCARDED: Episode {} finished after {} timesteps".format(s, t))
	# 		continue
	#
	# 	# Randomise filename to minimise conflicts even when running multiple times and across multiple machines
	# 	episode_id = random.randint(0, 2**31 - 1)
	# 	filename = f'{DIR_NAME}{episode_id}.npz'
	# 	np.savez_compressed(filename, obs=downsampled, action=action_seq, reward=reward_seq, done=done_seq) # <4>
	#
	# 	print("SAVED: Episode {} finished after {} timesteps".format(s, t))
	# 	s = s + 1
	# env.close()

# 0: do nothing
# 1: shoot
# 2: right
# 3: left
# 4: shoot + right
# 5: shoot + left
actions = np.array([0, 1, 2, 3, 4, 5])
def generate_data_action(t, env):
	return np.random.choice(actions, p=[0.1, 0.18, 0.18, 0.18, 0.18, 0.18])

def batch_downsample(batch):
	batch_size = len(batch)
	crop = batch[:, 9:-9, 16:-16]
	blocky = block_reduce(crop, (1,3,2,1), np.max)

	# Select top five rows of each frame
	partial = blocky[:, :5, :, :]

	# Count the number of non-zero red (in RGB) elements in top section of each frame
	nonzero = np.count_nonzero(partial[:, :, :, 0], axis=(1,2))
	scoreboard = nonzero > 48
	commander  = (~scoreboard) & (nonzero > 0)

	# if commander does not exist, do not process further
	if np.count_nonzero(commander) == 0:
		return None

	# smoother resizing method
	blurry = resize(crop, (batch_size, 64, 64, 3), mode='constant')

	# remove scoreboard
	blurry[scoreboard, :5] = 0

	# whiten bullets
	blurry[blocky[:,:,:,0] == 142] = 1.0

	# whiten commandership
	blurry[commander, :5] = np.where(blocky[commander, :5] > 0, 1.0, 0.0)

	return blurry


def downsample(frame):
	crop = frame[9:-9, 16:-16]
	blocky = block_reduce(crop, (3,2,1), np.max)

	# Select top five rows of the frame
	partial = blocky[:5, :, :]

	# Count the number of non-zero red (in RGB) elements in top section of each frame
	nonzero = np.count_nonzero(partial[:, :, 0])

	# smoother resizing method
	blurry = resize(crop, (64, 64, 3), mode='constant')

	# remove scoreboard
	if nonzero > 48:
		blurry[:5] = 0
	elif nonzero > 0:
		blurry[:5] = np.where(blocky[:5] > 0, 1.0, 0.0)

	# whiten bullets
	blurry[blocky[:,:,0] == 142] = 1.0
	return blurry

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=('Create new training data'))
	parser.add_argument('--nb_episodes', type=int, default=2000,
						help='total number of episodes to generate per worker')
	parser.add_argument('--time_steps', type=int, default=100000,
						help='how many timesteps at start of episode?')
	parser.add_argument('--render', default=0, type=int,
						help='render the env as data is generated')
	# extra hyperparameters
	parser.add_argument('--z_factor', default = 1)
	parser.add_argument('--action_factor', default = 1)
	parser.add_argument('--reward_factor', default = 1)
	parser.add_argument('--done_factor', default = 10)
	parser.add_argument('--hidden_units', default = 256)
	parser.add_argument('--grad_clip', default = 1.0)
	parser.add_argument('--num_mixture', default = 5)
	parser.add_argument('--restart_factor', default = 10)
	parser.add_argument('--learning_rate', default = 0.001)
	parser.add_argument('--decay_rate', default = 0.99999)
	parser.add_argument('--min_learning_rate', default = 0.00001)
	parser.add_argument('--use_layer_norm', default = 0)
	parser.add_argument('--use_recurrent_dropout', default = 0)
	parser.add_argument('--recurrent_dropout_prob', default = 0.90)
	parser.add_argument('--use_input_dropout', default = 0)
	parser.add_argument('--input_dropout_prob', default=0.90)
	parser.add_argument('--use_output_dropout', default = 0)
	parser.add_argument('--output_dropout_prob', default = 0.90)

	args = parser.parse_args()
	main(args)
