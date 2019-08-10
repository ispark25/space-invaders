#python 02_train_vae.py --new_model

from world_model.vae.arch import VAE, load_vae
import argparse
import numpy as np
import os
from datetime import datetime
from keras.callbacks import TensorBoard
from world_model.vae.data_loader import get_generators

# INPUT_DIR_NAME = './world_model/data/vae_food/'
# WEIGHT_FILE_NAME = './world_model/vae/weights/adam_64_heU_plus_bigR_0Tol.h5'
TV_RATIO = 0.2 # training and validation set split ratio


def main(args):
  exec_time = datetime.now().strftime('%Y%m%d-%H%M%S')
  tensorboard = TensorBoard(log_dir=f'log/vae/{exec_time}', update_freq='batch')

  new_model = args.new_model
  epochs = int(args.epochs)
  steps = int(args.steps)

  INPUT_DIR_NAME = args.input_dir_name
  WEIGHT_FILE_NAME = args.weight_file_name

  # instantiate VAE
  vae = VAE() if new_model else load_vae(WEIGHT_FILE_NAME)

  # get training set and validation set generators
  t_gen, v_gen = get_generators(INPUT_DIR_NAME, TV_RATIO)

  # start training!
  vae.train(t_gen, v_gen, 
    epochs=epochs, 
    steps_per_epoch=steps,
    validation_steps=int(steps * TV_RATIO),
    workers=10,
    callbacks=[tensorboard])
  
  # save model weights
  vae.save_weights(WEIGHT_FILE_NAME)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--epochs', default = 10, type=int, help='number of epochs to train for')
  parser.add_argument('--steps', default = 1000, type=int, help='number of steps per epoch')
  parser.add_argument('--input_dir_name', default='./world_model/data/vae_food/',
                        help='Directory name of rollouts.')
  parser.add_argument('--weight_file_name', default='./world_model/vae/weights/adam_64_heU_plus_bigR_0Tol.h5',
                        help='Directory name of rollouts.')
  args = parser.parse_args()

  main(args)
