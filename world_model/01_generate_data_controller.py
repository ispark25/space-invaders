import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize
import random
import gym

import argparse

ENV_NAME = 'SpaceInvaders-v0'
DIR_NAME = './data/vae_food_3/'
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
    while s < nb_episodes:
        obs_seq    = []
        action_seq = []
        reward_seq = []
        done_seq   = []

        observation = env.reset()
        reward = 0
        done = False

        # Player starts off with 3 lives
        prev_lives = 3

        repeat = np.random.randint(1, 11)
        t = 0
        while t < time_steps:
            if t % repeat == 0:
                action = generate_data_action(t, env)  # <2>
                repeat = np.random.randint(1, 11)

            obs_seq.append(observation)
            action_seq.append(action)
            reward_seq.append(reward)
            done_seq.append(done)

            if done:
                break

            # Next
            observation, reward, done, info = env.step(action)  # <4>

            # Extra code to penalise for death
            curr_lives = info['ale.lives']
            if curr_lives < prev_lives:
                reward_seq[max(0, t - DEATH_OFFSET)] += DEATH_PENALTY
            prev_lives = curr_lives

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


