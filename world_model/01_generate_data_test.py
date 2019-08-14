import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize
import random
import gym

from world_model.vae.arch_surprise_lower_tolerance import VAE, load_vae
from world_model.rnn.arch_v2 import RNN

import argparse

ENV_NAME = 'SpaceInvaders-v0'
DIR_NAME = './world_model/data/test/' # './data/vae_food_3/'
# MIN_LENGTH = 400 # agent should survive for at least this many frames

DEATH_OFFSET = 40 #32-45
DEATH_PENALTY = -500

def main(args):
	nb_episodes = args.nb_episodes
	time_steps  = args.time_steps
	render      = args.render

	env = gym.make(ENV_NAME) # <1>
	print("Generating data for env {}".format(ENV_NAME))
	s = 0

	rnn = RNN()
	rnn.decoder.load_weights("world_model/rnn/weights/weights_arch2_b1_s630.h5")

	vae = load_vae("world_model/vae/weights/arch_surprise_medium_tolerance.h5")

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

	while s < nb_episodes:
		obs_seq    = []
		action_seq = []
		reward_seq = []
		done_seq   = []

		observation = env.reset()
		hidden = np.zeros(rnn.hidden_units)
		cell = np.zeros(rnn.hidden_units)
		reward = 0
		cumulative_reward = 0
		done = False

		# Player starts off with 3 lives
		prev_lives = 3

		repeat = np.random.randint(1, 11)
		t = 0

		# for rollout test
		h = rnn
		while t < time_steps:

			if t % repeat == 0:
				action = generate_data_action(t, env)  # <2>
				repeat = np.random.randint(1, 11)

			obs_seq.append(observation)
			action_seq.append(action)
			reward_seq.append(reward)
			done_seq.append(done)

			if done:
				print("REACHED DONE STATE")
				break

			# Next
			observation, reward, done, info = env.step(action)  # <4>
			# Extra code to penalise for death
			curr_lives = info['ale.lives']
			if curr_lives < prev_lives:
				reward_seq[max(0, t - DEATH_OFFSET)] += DEATH_PENALTY
			prev_lives = curr_lives

			preprocessed = downsample(observation)
			# print(preprocessed)

			mu, lv = vae.encoder_mu_log_var.predict(np.array([preprocessed]))
			sb = np.shape(lv)
			z = mu + np.exp(lv/2.0) * np.random.randn(*sb)

			cumulative_reward += reward
			def convert_y(y):
				if y == 0 or y == 2 or y == 3:
					return 0
				else:
					return 1

			def convert_x(x):
				if x <= 1:
					return 0
				elif x%2 == 1:
					return -1
				else: #x%2 == 0
					return 1
			cont_action = [convert_x(int(action)),convert_y(int(action))]
			print("shapes are: z={}, action={}, reward={}, done={}".format(np.shape(z),np.shape([cont_action]),np.shape(reward),np.shape([done])))
			# z = np.expand_dims(z, axis = 0)
			# action = np.expand_dims(action, axis = 0)
			# action = np.expand_dims(action, axis = 0)
			# reward = [[[reward]]]
			# done = [[[int(done)]]]
			# done = np.expand_dims(np.array(done), axis=0)
			print("After expanding, shapes are: z={}, action={}, reward={}, done={}".format(np.shape(z[0]),np.shape(cont_action),np.shape(reward),np.shape(done)))
			# action = np.array(list(list(a) for a in zip([convert_x(int(a)) for a in [action]], [convert_y(int(a)) for a in [action]])))
			input_to_rnn = [np.array([[np.concatenate([z[0], cont_action, [reward]])]]), np.array([hidden]), np.array([cell])]


			out = rnn.decoder.predict(input_to_rnn)
			# print(out)
			y_pred = out[0][0][0]
			next_z = rnn.sample_decoder(y_pred, 1, 1)
			hidden = out[1][0]
			cell = out[2][0]

			print("next_z: {}".format(np.shape(next_z)))
			loss = (np.square(z[0] - next_z[0, :-(4)])).mean(axis=None)
			print(loss)
			print("y_pred: {}, hidden:{}, cell:{}".format(np.shape(y_pred), np.shape(hidden), np.shape(cell)))

			if render:
				env.render()
			t = t + 1

		# Save episode data if it featured commandership
		if t < 600: # commanders do not appear earlier than 600 steps
			print("DISCARDED: Episode {} finished after {} timesteps".format(s, t))
			continue

		obs_seq = np.array(obs_seq)
		downsampled = batch_downsample(obs_seq)

		if downsampled is None:
			print("DISCARDED: Episode {} finished after {} timesteps".format(s, t))
			continue

		# Randomise filename to minimise conflicts even when running multiple times and across multiple machines
		episode_id = random.randint(0, 2**31 - 1)
		filename = f'{DIR_NAME}{episode_id}.npz'
		np.savez_compressed(filename, obs=downsampled, action=action_seq, reward=reward_seq, done=done_seq) # <4>

		print("SAVED: Episode {} finished after {} timesteps".format(s, t))
		s = s + 1
	env.close()

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

	args = parser.parse_args()
	main(args)
