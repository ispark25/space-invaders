import numpy as np
import os
import sys
from skimage.measure import block_reduce
from skimage.transform import resize
import argparse

def preprocess(root, dest, a, b):
    for filename in os.listdir(root):
        
        number = to_number(filename)
        if not (a <= number and number < b):
            continue
        
        samples = np.load(root + filename)
        obs = batch_downsample(samples['obs'])

        np.savez_compressed(dest + filename, 
                            obs=obs,
                            action=samples['action'],
                            reward=samples['reward'],
                            done=samples['done'])
        print(f'Saved {filename}')
    print('Finished!')

def batch_downsample(batch):
    batch_size = len(batch)
    crop = batch[:, 9:-9, 16:-16]
    
    blurry = resize(crop, (batch_size, 64, 64, 3), mode='constant')
    blocky = block_reduce(crop, (1,3,2,1), np.max)
    
    for i in range(batch_size):
        if np.count_nonzero(blurry[i, :4, 32:, 0]) > 48:
            blurry[i, :4, :] = 0
    
    blurry[blocky[:,:,:,0] == 142] = 1.0
    return blurry

def to_number(filename):
    return int(filename[:-4])

if __name__ == '__main__':
    # 2^31 -1 = 2,147,483,647 --> 500,000,000
    parser = argparse.ArgumentParser(description=('Preprocess data'))
    parser.add_argument('--start', type=int, default=0,
                        help='where to start preprocessing from')
    parser.add_argument('--end', type=int, default=3000000000,
                        help='where to stop preprocessing (exclusive)')
    parser.add_argument('--dir_root', default='./world_model/data/raw_food/',
                        help='Directory name of root.')
    parser.add_argument('--dir_dest', default='./world_model/data/vae_food/',
                        help='Directory name of destinations.')
    args = parser.parse_args()
    preprocess(args.dir_root, args.dir_dest, args.start, args.end)