#python 04_train_rnn.py --new_model --batch_size 200
# python 04_train_rnn.py --new_model --batch_size 100

from world_model.rnn.arch import RNN
import argparse
import numpy as np
import os

# ROOT_DIR_NAME = './world_model/data/'
# SERIES_DIR_NAME = './world_model/data/series/'


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

    def get_frame_count(filelist):
        # get frame count
        frames = 0
        for f in filelist:
            data = np.load(SERIES_DIR_NAME + f)
            frames += len(data["action"])
        return frames

    batch_size = int(args.batch_size)
    seq_length = int(args.seq_length)
    # def create_batches(filelist, batch_size=10, seq_length=500):

    num_frames = get_frame_count(filelist)
    num_batches = int(num_frames/(batch_size*seq_length))
    frames_adjusted = num_batches*batch_size*seq_length
    # indices = np.random.permutation(len(filelist))

    # num_frames = get_frame_count(filelist)

    mu = np.zeros((num_frames, 64), dtype=np.float16) # width of z
    logvar = np.zeros((num_frames, 64), dtype=np.float16) # width of z
    action = np.zeros(num_frames, dtype=np.float16)
    rew = np.zeros(num_frames, dtype=np.float16)
    done = np.zeros(num_frames, dtype=np.uint8)
    idx = 0
    for i in filelist:
        data = np.load(SERIES_DIR_NAME + i)
        N = len(data["action"])
        mu[idx:idx+N] = data["mu"].reshape(N, 64)
        logvar[idx:idx+N] = data["lv"].reshape(N, 64)
        action[idx:idx+N] = data["action"].reshape(N)
        rew[idx:idx+N] = data["reward"].reshape(N)
        done[idx] = 1

        idx += N
        if idx % 100000 < 1000:
            print("batch processing: frame {}".format(idx))
    

    mu = mu[0:frames_adjusted]
    logvar = logvar[0:frames_adjusted]
    action = action[0:frames_adjusted]
    rew = rew[0:frames_adjusted]
    done = done[0:frames_adjusted]

    mu = np.split(mu.reshape(batch_size, -1, 64), num_batches, 1)
    logvar = np.split(logvar.reshape(batch_size, -1, 64), num_batches, 1)

    print(action)
    action = action.astype(int)
    print(action)
    print(np.shape(action))

    one_hot_action = np.zeros((action.size,6))  # a.max()+1))
    one_hot_action[np.arange(action.size), action] = 1
    # action = np.split(action.reshape(batch_size, -1), num_batches, 1)
    action = np.split(one_hot_action.reshape(batch_size, -1, 6), num_batches, 1)
    
    rew = np.split(rew.reshape(batch_size, -1), num_batches, 1)
    done = np.split(done.reshape(batch_size, -1), num_batches, 1)

    total_data = (mu, logvar, action, rew, done)

    # create_batches(filelist, , )

    return filelist, total_data, N, num_batches


def random_batch(filelist, total_data, num_batches, batch_size=100, seq_length=300):

    ROOT_DIR_NAME = args.root_dir_name
    SERIES_DIR_NAME = args.series_dir_name

    batch_index = np.random.randint(num_batches)
    mu, logvar, action, rew, done = total_data
    batch_mu = mu[batch_index]
    batch_logvar = logvar[batch_index]
    action_list = action[batch_index]
    rew_list = rew[batch_index]
    done_list = done[batch_index]
    batch_s = batch_logvar.shape
    z_list = batch_mu + np.exp(batch_logvar/2.0) * np.random.randn(*batch_s)

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


    filelist, total_data, N, num_batches= get_filelist(N)


    for step in range(steps):
        print('STEP ' + str(step))

        z, action, rew ,done = random_batch(filelist, total_data, num_batches, batch_size, seq_length)

        print("shapes are: z={}, action={}, reward={}, done={}".format(np.shape(z),np.shape(action),np.shape(rew),np.shape(done)))

        # action = np.expand_dims(action, axis=2)
        rew = np.expand_dims(rew, axis=2)
        done = np.expand_dims(done,axis=2)

        print("After expanding, shapes are: z={}, action={}, reward={}, done={}".format(np.shape(z),np.shape(action),np.shape(rew),np.shape(done)))


        rnn_input = np.concatenate([z[:, :-1, :], action[:, :-1, :], rew[:, :-1, :]], axis = 2)
        rnn_output = np.concatenate([z[:, 1:, :], action[:, 1:,:], rew[:, 1:, :], done[:, 1:,:]], axis = 2) # ,action[:, 1:], done[:, 1:]

        print("Shape of rnnin: {}, rnnout:{}".format(np.shape(rnn_input), np.shape(rnn_output)))

        if step == 0:
            np.savez_compressed(ROOT_DIR_NAME + 'rnn_files.npz', rnn_input = rnn_input, rnn_output = rnn_output)

        rnn.train(rnn_input, rnn_output)

        if step % 10 == 0:

            rnn.model.save_weights('./world_model/rnn/weights.h5')

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
