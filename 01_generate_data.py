import numpy as np
import random
import gym

import argparse

ENV_NAME = 'SpaceInvaders-v0'
DIR_NAME = './data/rollout/'
# MIN_LENGTH = 400 # agent should survive for at least this many frames

DEATH_OFFSET = 40 #32-45
DEATH_PENALTY = -500

def main(args):    
    nb_episodes = args.nb_episodes
    time_steps  = args.time_steps
    render      = args.render

    env = gym.make(ENV_NAME) # <1>
    print("Generating data for env {}".format(ENV_NAME))

    for s in range(nb_episodes):

        episode_id = random.randint(0, 2**31 - 1) # To minimise conflicts even when running multiple times
        filename = f'{DIR_NAME}{episode_id}.npz'

        obs_seq    = []
        action_seq = []
        reward_seq = []
        done_seq   = []

        observation = env.reset()
        reward = 0 # TODO: base reward per frame: e.g. -0.1? possibly bad
        done = False

        # Player starts off with 3 lives
        prev_lives = 3

        repeat = np.random.randint(1, 11)
        t = 0
        while t < time_steps and not done:
            if t % repeat == 0:
                action = generate_data_action(t, env)  # <2>
                repeat = np.random.randint(1, 11)

            # observation = cfg.adjust_obs(observation)  # <3>
            # reward      = cfg.adjust_reward(reward)

            obs_seq.append(observation)
            action_seq.append(action)
            reward_seq.append(reward)
            done_seq.append(done)

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

        print("Episode {} finished after {} timesteps".format(s, t))

        # Save episode data
        np.savez_compressed(filename, obs=obs_seq, action=action_seq, reward=reward_seq, done=done_seq) # <4>

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

# def adjust_obs(obs):
#     # apply cropping etc.
#     # return obs.astype('float32') / 255.0
#     return obs

# def adjust_reward(reward):
#     return reward / 10.0

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


