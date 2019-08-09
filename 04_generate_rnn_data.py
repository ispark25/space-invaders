#python 03_generate_rnn_data.py

from vae.arch import VAE, load_vae
import numpy as np
import os

ROOT_DIR_NAME = './data/'
INPUT_DIR_NAME = './data/vae_food/'
OUTPUT_DIR_NAME = './data/rnn_food/'

VAE_WEIGHT_FILE_NAME = './vae/weight/...'

def main(args):
    vae = load_vae(VAE_WEIGHT_FILE_NAME)

    files = os.listdir(INPUT_DIR_NAME)
    nb_files = len(files)
    nb_encoded = 0

    init_mus = []
    init_lvs = []

    for filename in files:
        episode_data = np.load(INPUT_DIR_NAME + filename)

        mu, lv = vae.encoder_mu_log_var.predict(episode_data['obs'])

        init_mus.append(mu[0, :]) # mu[i, :] = mu vector for i-th episode
        init_lvs.append(lv[0, :]) # lv[i, :] = log_var vector for i-th episode

        np.savez_compressed(OUTPUT_DIR_NAME + file, 
            mu=mu, 
            lv=lv, 
            action = episode_data['action'], 
            reward = episode_data['reward'], 
            done = episode_data['done'].astype(int)) # TODO: why?

        # Log progress
        nb_encoded += 1
        if nb_encoded % 100 == 0:
            print(f'Encoded {count} / {nb_files} episodes')
    print(f'Encoded {count} / {nb_files} episodes')

    init_mus = np.array(init_mus)
    init_lvs = np.array(init_lvs)

    print('ONE MU SHAPE = {}'.format(mu.shape))
    print('INITIAL MU SHAPE = {}'.format(initial_mus.shape))

    np.savez_compressed(ROOT_DIR_NAME + 'initial_z.npz', init_mus=init_mus, init_lvs=init_lvs) # TODO: what for?

    
if __name__ == '__main__':
    main()
