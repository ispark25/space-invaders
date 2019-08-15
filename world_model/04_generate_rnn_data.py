#python 03_generate_rnn_data.py

from world_model.vae.arch_surprise import VAE, load_vae
import numpy as np
import argparse
import os

# ROOT_DIR_NAME = './world_model/data/'
# INPUT_DIR_NAME = './world_model/data/vae_food/'
# OUTPUT_DIR_NAME = './world_model/data/rnn_food/'

# VAE_WEIGHT_FILE_NAME = './world_model/vae/weight/...'

def main(args):

    ROOT_DIR_NAME = args.root_dir_name
    INPUT_DIR_NAME = args.input_dir_name
    OUTPUT_DIR_NAME = args.output_dir_name

    VAE_WEIGHT_FILE_NAME = args.vae_weight_file_name

    vae = load_vae(VAE_WEIGHT_FILE_NAME)

    files = os.listdir(INPUT_DIR_NAME)
    nb_files = len(files)
    nb_encoded = 0
    print(f'Encoded {nb_encoded} / {nb_files} episodes')

    init_mus = []
    init_lvs = []

    for filename in files:
        if nb_encoded < 5300:
            pass
        else:
            episode_data = np.load(INPUT_DIR_NAME + filename)

            mu, lv = vae.encoder_mu_log_var.predict(episode_data['obs'])

            init_mus.append(mu[0, :]) # mu[i, :] = mu vector for i-th episode
            init_lvs.append(lv[0, :]) # lv[i, :] = log_var vector for i-th episode

            np.savez_compressed(OUTPUT_DIR_NAME + filename,
                mu=mu,
                lv=lv,
                action = episode_data['action'],
                reward = episode_data['reward'],
                done = episode_data['done'].astype(int)) # TODO: why?

        # Log progress
        nb_encoded += 1
        if nb_encoded % 100 == 0:
            print(f'Encoded {nb_encoded} / {nb_files} episodes')
    print(f'Encoded {nb_encoded} / {nb_files} episodes')

    init_mus = np.array(init_mus)
    init_lvs = np.array(init_lvs)

    print('ONE MU SHAPE = {}'.format(mu.shape))
    print('INITIAL MU SHAPE = {}'.format(init_mus.shape))

    np.savez_compressed(ROOT_DIR_NAME + 'initial_z.npz', init_mus=init_mus, init_lvs=init_lvs) # TODO: what for?


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--root_dir_name', default = './world_model/data/', help='root Directory.')
    parser.add_argument('--input_dir_name', default='./world_model/data/vae_food/',
                            help='Directory name of vae input.')
    parser.add_argument('--output_dir_name', default='./world_model/data/rnn_food/',
                            help='Directory name of rnn input.')
    parser.add_argument('--vae_weight_file_name', default='./world_model/vae/weights/...',
                            help='weight file for vae.')
    args = parser.parse_args()

    main(args)
