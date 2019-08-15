#python 04_train_rnn.py --new_model --batch_size 200
# python 04_train_rnn.py --new_model --batch_size 100

from world_model.rnn.arch import RNN
import argparse
import numpy as np
import os

from datetime import datetime



# ROOT_DIR_NAME = './world_model/data/'
# SERIES_DIR_NAME = './world_model/data/series/'
HIST_DIR_NAME = './world_model/rnn/history/hist-{}.json'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))


def get_filelist(N):

    ROOT_DIR_NAME = args.root_dir_name
    SERIES_DIR_NAME = args.series_dir_name

    filelist = os.listdir(SERIES_DIR_NAME)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)

    if length_filelist > N:
      filelist = filelist[:N]

    if length_filelist < N:
      N = length_filelist

    def get_frame_count(filelist, N):
        # get frame count
        frames = 0
        for i in range(N):
        # for f in filelist:
            data = np.load(SERIES_DIR_NAME + filelist[i])
            frames += len(data["action"])
        return frames

    batch_size = int(args.batch_size)
    seq_length = int(args.seq_length)

    # calculate the number of frames in total, and how many batches
    num_frames = get_frame_count(filelist, N)
    num_batches = int(num_frames/(batch_size*seq_length))
    frames_adjusted = num_batches*batch_size*seq_length
    # randomize the files first
    indices = np.random.permutation(N) # len(filelist))
    num_frames = get_frame_count(filelist, N)

    # initialise zero array for each variable of size num_frames
    mu = np.zeros((num_frames, 64), dtype=np.float16) # width of z
    logvar = np.zeros((num_frames, 64), dtype=np.float16) # width of z
    action = np.zeros(num_frames, dtype=np.float16)
    rew = np.zeros(num_frames, dtype=np.float16)
    done = np.zeros(num_frames, dtype=np.uint8)

    # populate data
    idx = 0
    for i in indices:
        data = np.load(SERIES_DIR_NAME + filelist[i])
        N = len(data["action"])
        mu[idx:idx+N] = data["mu"].reshape(N, 64)
        logvar[idx:idx+N] = data["lv"].reshape(N, 64)
        action[idx:idx+N] = data["action"].reshape(N)
        rew[idx:idx+N] = data["reward"].reshape(N)
        done[idx+N-1] = 1

        # crude process check
        idx += N
        if idx % 100000 < 1000:
            print("batch processing: frame {}".format(idx))

    # crop data into integer batches
    mu = mu[0:frames_adjusted]
    logvar = logvar[0:frames_adjusted]
    action = action[0:frames_adjusted]
    rew = rew[0:frames_adjusted]
    done = done[0:frames_adjusted]

    # reshape data
    mu = mu.reshape(batch_size, -1, 64)         # shape = (batch_size, seq_length, z_dimension)
    logvar = logvar.reshape(batch_size, -1, 64) # shape = (batch_size, seq_length, z_dimension)
    rew = rew.reshape(batch_size, -1)                 # shape = (batch_size, seq_length)
    done = done.reshape(batch_size, -1)                # shape = (batch_size, seq_length)
    # one hot encode the action, turn [0-5] into [0, 0, 0, 0, 0, 1]
    action = action.astype(int)
    one_hot_action = np.zeros((action.size,6))  # a.max()+1))
    one_hot_action[np.arange(action.size), action] = 1
    action = one_hot_action.reshape(batch_size, -1, 6) # shape = (batch_size, seq_length, a_dimension)

    print("shapes are: mu={},lv={}, action={}, reward={}, done={}".format(np.shape(mu),np.shape(logvar),np.shape(action),np.shape(rew),np.shape(done)))


    # split into train and test
    VALIDATION = int((num_batches/10.0)*seq_length)
    vmu = mu[:, :VALIDATION, :]
    vlogvar = logvar[:, :VALIDATION, :]
    vaction = action[:, :VALIDATION, :]
    vrew = rew[:, :VALIDATION]
    vdone = done[:, :VALIDATION]

    mu = np.split(mu[:, VALIDATION:, :], num_batches, 1)
    logvar = np.split(logvar[:, VALIDATION:, :], num_batches, 1)
    action = np.split(action[:, VALIDATION:, :], num_batches, 1)
    rew = np.split(rew[:, VALIDATION:], num_batches, 1)
    done = np.split(done[:, VALIDATION:], num_batches, 1)

    # mu = np.split(mu.reshape(batch_size, -1, 64), num_batches, 1)
    # logvar = np.split(logvar.reshape(batch_size, -1, 64), num_batches, 1)

    # action = action.astype(int)
    # one_hot_action = np.zeros((action.size,6))  # a.max()+1))
    # one_hot_action[np.arange(action.size), action] = 1
    # # action = np.split(action.reshape(batch_size, -1), num_batches, 1)
    # action = np.split(one_hot_action.reshape(batch_size, -1, 6), num_batches, 1)

    # rew = np.split(rew.reshape(batch_size, -1), num_batches, 1)
    # done = np.split(done.reshape(batch_size, -1), num_batches, 1)

    train_data = (mu, logvar, action, rew, done)
    test_data = (vmu, vlogvar, vaction, vrew, vdone)

    return filelist, train_data, test_data, N, num_batches


def random_batch(filelist, train_data, test_data, num_batches, batch_size=100, seq_length=300, validation=False):

    ROOT_DIR_NAME = args.root_dir_name
    SERIES_DIR_NAME = args.series_dir_name

    if validation:
        # batch_index = np.random.randint(int(num_batches/10.0))
        batch_index = int(num_batches/10.0)
        batch_mu, batch_logvar, action_list, rew_list, done_list = test_data
        # for i in range(int(num_batches/10.0)):
        # batch_mu = mu[:batch_index]
        # batch_logvar = logvar[:batch_index]
        # action_list = action[:batch_index]
        # rew_list = rew[:batch_index]
        # done_list = done[:batch_index]
        batch_s = batch_logvar.shape
        z_list = batch_mu + np.exp(batch_logvar/2.0) * np.random.randn(*batch_s)
    else:
        batch_index = np.random.randint(high=num_batches, low=int(num_batches/10))
        mu, logvar, action, rew, done = train_data
        batch_mu = mu[batch_index]
        batch_logvar = logvar[batch_index]
        action_list = action[batch_index]
        rew_list = rew[batch_index]
        done_list = done[batch_index]
        batch_s = batch_logvar.shape
        z_list = batch_mu + np.exp(batch_logvar/2.0) * np.random.randn(*batch_s)

    # mu, logvar, action, rew, done = total_data
    # batch_mu = mu[batch_index]
    # batch_logvar = logvar[batch_index]
    # action_list = action[batch_index]
    # rew_list = rew[batch_index]
    # done_list = done[batch_index]
    # batch_s = batch_logvar.shape
    # z_list = batch_mu + np.exp(batch_logvar/2.0) * np.random.randn(*batch_s)

    # N_data = len(filelist)
    # indices = np.random.permutation(N_data)[0:batch_size]

    # print(indices)

    # z_list = []
    # action_list =[]
    # rew_list = []
    # done_list = []

    # for i in indices:
    #     try:
    #         new_data = np.load(SERIES_DIR_NAME + filelist[i])

    #         mu = new_data['mu']
    #         log_var = new_data['lv'] # 'log_var is not a file in the archive'
    #         action = new_data['action']
    #         reward = new_data['reward']
    #         done = new_data['done']

    #         action = np.expand_dims(action, axis=2)
    #         reward = np.expand_dims(reward, axis=2)
    #         done = np.expand_dims(done, axis=2)

    #         print("mu:{}, lv:{}, action:{}, reward:{}, done:{}.".format(np.shape(mu),np.shape(log_var),np.shape(action),np.shape(reward),np.shape(done)))

    #         s = log_var.shape

    #         z = mu + np.exp(log_var/2.0) * np.random.randn(*s)

    #         print("z:{}, action:{}, reward:{}, done:{}.".format(np.shape(z),np.shape(action),np.shape(rew),np.shape(done)))

    #         # batch into pieces seq_length long with zero padding
    #         dmu = np.zeros((num_fram))


    #         z_list.append(z)
    #         action_list.append(action)
    #         rew_list.append(reward)
    #         done_list.append(done)

    #     except:
    #         print("an error occured for {}".format(i))
    #         pass

    # z_list = np.array(z_list)
    # action_list = np.array(action_list)
    # rew_list = np.array(rew_list)
    # done_list = np.array(done_list)

    # print("z:{}, action:{}, reward:{}, done:{}.".format(np.shape(z_list),np.shape(action_list),np.shape(rew_list),np.shape(done_list)))


    return z_list, action_list, rew_list, done_list

def main(args):

    ROOT_DIR_NAME = args.root_dir_name
    SERIES_DIR_NAME = args.series_dir_name

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


    filelist, train_data, test_data, N, num_batches= get_filelist(N)


    for step in range(steps):
        print('STEP ' + str(step))

        z, action, rew ,done = random_batch(filelist, train_data, test_data, num_batches, batch_size, seq_length, False)

        print("shapes are: z={}, action={}, reward={}, done={}".format(np.shape(z),np.shape(action),np.shape(rew),np.shape(done)))

        # action = np.expand_dims(action, axis=2)
        rew = np.expand_dims(rew, axis=2)
        done = np.expand_dims(done,axis=2)

        print("After expanding, shapes are: z={}, action={}, reward={}, done={}".format(np.shape(z),np.shape(action),np.shape(rew),np.shape(done)))


        rnn_input = np.concatenate([z[:, :-1, :], action[:, :-1, :], rew[:, :-1, :]], axis = 2)
        rnn_output = np.concatenate([z[:, 1:, :], action[:, 1:,:], rew[:, 1:, :], done[:, 1:,:]], axis = 2) # ,action[:, 1:], done[:, 1:]

        print("Shape of rnnin: {}, rnnout:{}".format(np.shape(rnn_input), np.shape(rnn_output)))

        # validation set
        vz, vaction, vrew , vdone = random_batch(filelist, train_data, test_data, num_batches, batch_size, seq_length, True)
        vrew = np.expand_dims(vrew, axis=2)
        vdone = np.expand_dims(vdone,axis=2)
        vrnn_input = np.concatenate([vz[:, :-1, :], vaction[:, :-1, :], vrew[:, :-1, :]], axis = 2)
        vrnn_output = np.concatenate([vz[:, 1:, :], vaction[:, 1:,:], vrew[:, 1:, :], vdone[:, 1:,:]], axis = 2) # ,action[:, 1:], done[:, 1:]

        if step == 0:
            np.savez_compressed(ROOT_DIR_NAME + 'rnn_files.npz', rnn_input = rnn_input, rnn_output = rnn_output)

        rnn.train(rnn_input, rnn_output, vrnn_input, vrnn_output, step)

        if step % 10 == 0:

            rnn.model.save_weights('./world_model/rnn/weights.h5')

        if step % 100 == 0:



            # rnn.model.test(rnn_input, rnn_output)
            rnn.save_history(HIST_DIR_NAME)

    rnn.model.save_weights('./world_model/rnn/weights.h5')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train RNN'))
    parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
    parser.add_argument('--steps', default = 4000, help='how many rnn batches to train over')
    parser.add_argument('--batch_size', default = 100, help='how many episodes in a batch?')
    parser.add_argument('--seq_length', default = 300, help='how long is a sequence?')
    parser.add_argument('--root_dir_name', default='./world_model/data/',
                        help='Directory name of root.')
    parser.add_argument('--series_dir_name', default='./world_model/data/rnn_food/',
                        help='Directory name of series.')
    parser.add_argument('--rnn_weights', default='./world_model/rnn/weights.h5',
                        help="Directory name for rnn weights")

    args = parser.parse_args()

    main(args)
