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

def to_number(filename):
    return int(filename[:-4])

if __name__ == '__main__':
    # 2^31 -1 = 2,147,483,647 --> 500,000,000
    parser = argparse.ArgumentParser(description=('Preprocess data'))
    parser.add_argument('--start', type=int, default=0,
                        help='where to start preprocessing from')
    parser.add_argument('--end', type=int, default=3000000000,
                        help='where to stop preprocessing (exclusive)')
    args = parser.parse_args()
    preprocess('data/raw_food/', 'data/vae_food/', args.start, args.end)