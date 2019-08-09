#python 02_train_vae.py --new_model

from vae.arch2 import VAE
import argparse
import numpy as np
import os
from vae.data_loader import get_generators

DIR_NAME = './data/rollout_processed/'
WEIGHT_FILE = './vae/adam_64_heU_plus_bigR_0Tol.h5'
SCREEN_SIZE_X = 64
SCREEN_SIZE_Y = 64
TV_RATIO = 0.2

from datetime import datetime
from keras.callbacks import TensorBoard
def main(args):
  exec_time = datetime.now().strftime('%Y%m%d-%H%M%S')
  tensorboard = TensorBoard(log_dir=f'log/vae/{exec_time}', update_freq='batch')

  new_model = args.new_model
  epochs = int(args.epochs)
  steps = int(args.steps)

  vae = VAE()

  if not new_model:
    try:
      vae.set_weights(WEIGHT_FILE)
    except:
      print(f'Either set --new_model or ensure {WEIGHT_FILE} exists')
      raise
  
  t_gen, v_gen = get_generators(DIR_NAME, TV_RATIO)
  #steps per epoch
  vae.train(t_gen, v_gen, 
    epochs=epochs, 
    steps_per_epoch=steps,
    validation_steps=int(steps * TV_RATIO),
    workers=10,
    callbacks=[tensorboard])
  
  vae.save_weights(WEIGHT_FILE)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--epochs', default = 10, help='number of epochs to train for')
  parser.add_argument('--steps', default = 1000, help='number of steps per epoch')
  args = parser.parse_args()

  main(args)
