import sys
import numpy as np
import os
from skimage.measure import block_reduce
from skimage.transform import resize
from skimage.color import rgb2gray
from scipy import signal
import argparse


b = np.array([[-1, 1,-1], 
[-1,-1,-1]])
max_peak = np.prod(b.shape)

def bullet_encoder(bullet_mask):
    a = np.empty((64, 64), dtype=np.int8)
    a[:, :] = -1
    a[bullet_mask] = 1

    c = signal.correlate(a, b, 'valid')

    ys, xs = np.where(c == max_peak)
    nb_bullets = min(8, len(ys))

    # centre of the bulllet, xs + 1, ranges from 1-62
    # xs = ((xs + 1) - 31.5) / (62.0 / 2)
    xs = np.clip((xs - 30.5) / 31.0, -1, 1)
    ys = np.clip((ys - 31.5) / 31.5, -1, 1)

    encoded = np.full((24,), -1, dtype=np.float32)
    encoded[0:nb_bullets*3:3] = xs[:nb_bullets]
    encoded[1:nb_bullets*3:3] = ys[:nb_bullets]
    encoded[2:nb_bullets*3:3] = 1
    return encoded

def batch_downsample(batch):
    batch_size = len(batch)
    crop = batch[:, 9:-9, 16:-16]
    
    blurry = resize(crop, (batch_size, 64, 64, 3), mode='constant')
    blocky = block_reduce(crop, (1,3,2,1), np.max)
    bullet_mask = blocky[:,:,:,0] == 142
    
    bullets = []
    for i in range(batch_size):
        if np.count_nonzero(blurry[i, :4, 32:, 0]) > 48:
            blurry[i, :4, :] = 0
        bullets.append(bullet_encoder(bullet_mask[i]))
    
    removed = blurry.copy()
    removed[bullet_mask] = 0    
    blurry[bullet_mask] = 1.0
    return blurry, removed, np.array(bullets)

def preprocess(root, dest, a, b):
    for filename in os.listdir(root):
        
        number = to_number(filename)
        if not (a <= number and number < b):
            continue
        
        samples = np.load(root + filename)
        emphasised, removed, bullets = batch_downsample(samples['obs'])

        np.savez_compressed(dest + filename, 
                            bullet_emphasised=emphasised,
                            bullet_removed=removed,
                            bullets=bullets,
                            action=samples['action'],
                            reward=samples['reward'],
                            done=samples['done'])
        print(f'Saved {filename}')
    print('Finished!')

def to_number(filename):
    return int(filename[:-4])

if __name__ == '__main__':
    # 2^31 -1 = 2,147,483,647
    # 500,000,000
    parser = argparse.ArgumentParser(description=('Preprocess data'))
    parser.add_argument('--start', type=int, default=0,
                        help='where to start preprocessing from')
    parser.add_argument('--end', type=int, default=3000000000,
                        help='where to stop preprocessing (exclusive)')
    args = parser.parse_args()
    preprocess('data/rollout/', 'data/rollout_processed/', args.start, args.end)