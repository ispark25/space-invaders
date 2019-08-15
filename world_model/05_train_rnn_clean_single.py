#python 04_train_rnn.py --new_model --batch_size 200
# python 04_train_rnn.py --new_model --batch_size 100

from world_model.rnn.arch_v3 import RNN
import argparse
import numpy as np
import os

from datetime import datetime

# test command:
# python -m world_model.05_train_rnn --new_model --N=100 --batch_size=2 --root_dir_name=./world_model/data/test/ --rnn_weights=./world_model/rnn/weights.h5

# actual command:
# python -m world_model.05_train_rnn --new_model --batch_size=1 --root_dir_name=./world_model/data/ --rnn_weights=./world_model/rnn/weights.h5
def get_filelist(N):
	filelist = os.listdir(args.series_dir_name)
	filelist = [x for x in filelist if x != '.DS_Store']
	filelist.sort()
	length_filelist = len(filelist)

	if length_filelist > N:
	  filelist = filelist[:N]

	if length_filelist < N:

	  N = length_filelist

	return filelist, N


def random_batch(filelist, batch_size, validation=False, VR_RATIO = 10.0):
	N_data = len(filelist)
	if validation:
		indices = np.random.permutation(int(N_data/VR_RATIO))[0:batch_size]
	else:
		indices = np.random.permutation(np.arange(int(N_data/VR_RATIO),N_data))[0:batch_size]

	z_list = []
	action_list = []
	rew_list = []
	done_list = []

	for i in indices:
		new_data = np.load(args.series_dir_name + str(filelist[i]))

		mu = new_data['mu']
		log_var = new_data['lv']
		action = new_data['action']
		reward = new_data['reward']
		done = new_data['done']

		reward = np.expand_dims(reward, axis=2)
		done = np.expand_dims(done, axis=2)

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

		action = np.array(list(list(a) for a in zip([convert_x(int(a)) for a in action], [convert_y(int(a)) for a in action])))

		s = log_var.shape
		z = mu + np.exp(log_var/2.0) * np.random.randn(*s)

		z_list.append(z)
		action_list.append(action)
		rew_list.append(reward)
		done_list.append(done)

	z_list = np.array(z_list)
	action_list = np.array(action_list)
	rew_list = np.array(rew_list)
	done_list = np.array(done_list)

	return z_list, action_list, rew_list, done_list

def main(args):

	new_model = args.new_model
	N = int(args.N)
	steps = int(args.steps)
	batch_size = int(args.batch_size)
	seq_length = int(args.seq_length)

	rnn = RNN() #learning_rate = LEARNING_RATE

	if not new_model:
		try:
			rnn.set_weights(args.rnn_weights)
		except:
			print("Either set --new_model or ensure ./rnn/weights.h5 exists")
			raise


	filelist, N = get_filelist(N)

	for step in range(steps):
		print('STEP ' + str(step))

		z, action, rew, done = random_batch(filelist, batch_size)
		print("shapes are: z={}, action={}, reward={}, done={}".format(np.shape(z),np.shape(action),np.shape(rew),np.shape(done)))
		print("rnnin shapes are: z={}, action={}, reward={}, done={}".format(np.shape(z[:, :-1,:]),np.shape(action[:, :-1,:]),np.shape(rew[:, :-1,:]),np.shape(done[:, :-1,:])))
		rnn_input = np.concatenate([z[:, :-1, :], action[:, :-1, :], rew[:, :-1, :]], axis = 2)
		rnn_output = np.concatenate([z[:, 1:, :], action[:, 1:,:], rew[:, 1:, :], done[:, 1:,:]], axis = 2) # ,action[:, 1:], done[:, 1:]
		print("Shape of rnnin: {}, rnnout:{}".format(np.shape(rnn_input), np.shape(rnn_output)))

		# validation set
		vz, vaction, vrew , vdone = random_batch(filelist, batch_size, True)
		print("shapes are: z={}, action={}, reward={}, done={}".format(np.shape(vz),np.shape(vaction),np.shape(vrew),np.shape(vdone)))
		vrnn_input = np.concatenate([vz[:, :-1, :], vaction[:, :-1, :], vrew[:, :-1, :]], axis = 2)
		vrnn_output = np.concatenate([vz[:, 1:, :], vaction[:, 1:,:], vrew[:, 1:, :], vdone[:, 1:,:]], axis = 2) # ,action[:, 1:], done[:, 1:]

		if step == 0:
			np.savez_compressed(args.root_dir_name + 'rnn_files.npz', rnn_input = rnn_input, rnn_output = rnn_output)

		rnn.train(rnn_input, rnn_output, vrnn_input, vrnn_output, step)

		if step % 100 == 0:
			rnn.model.save_weights(args.rnn_weights + 'step_{}.h5'.format(step))
			rnn.save_history(args.hist_dir_name)

		# if step % 100 == 0:
		# 	rnn.save_history(HIST_DIR_NAME)

	rnn.model.save_weights('./world_model/rnn/weights.h5')




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=('Train RNN'))
	parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
	parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
	parser.add_argument('--steps', default = 10000, help='how many rnn batches to train over')
	parser.add_argument('--batch_size', default = 1, help='how many episodes in a batch? (currently online learning)')
	parser.add_argument('--seq_length', default = 300, help='how long is a sequence?')
	parser.add_argument('--root_dir_name', default='./world_model/data/',
						help='Directory name of root.')
	parser.add_argument('--series_dir_name', default='./world_model/data/rnn_food/',
						help='Directory name of series.')
	parser.add_argument('--rnn_weights', default='./world_model/rnn/weights.h5',
						help="Directory name for rnn weights")
	parser.add_argument('--hist_dir_name', default='./world_model/rnn/history/hist-{}.json'.format(datetime.now().strftime("%Y%m%d-%H%M%S")),
						help="")
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
