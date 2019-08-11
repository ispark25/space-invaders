#python 03_train_vae.py --new_model

from vae.arch_surprise import VAE, load_vae
import argparse
import numpy as np
import os
from datetime import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from vae.data_loader import get_generators

INPUT_DIR_NAME = './data/vae_food_prioritised2/'
FNAME = 'arch_surprise_purple.h5'
WEIGHT_FILE_NAME = f'./vae/weight/{FNAME}'
TV_RATIO = 0.2 # training and validation set split ratio


def main(args):
  exec_time = datetime.now().strftime('%Y%m%d-%H%M%S')
  tensorboard = TensorBoard(log_dir=f'log/vae/{exec_time}', update_freq='batch')
  checkpoint = ModelCheckpoint(f'./vae/weight/best/{FNAME}', monitor='val_loss', save_best_only=True)

  new_model = args.new_model
  epochs = int(args.epochs)
  steps = int(args.steps)

  # instantiate VAE
  vae = VAE() if new_model else load_vae(WEIGHT_FILE_NAME)

  # get training set and validation set generators
  t_gen, v_gen = get_generators(INPUT_DIR_NAME, TV_RATIO)

  # start training!
  vae.train(t_gen, v_gen, 
    epochs=epochs, 
    steps_per_epoch=steps,
    validation_steps=int(steps * TV_RATIO), #TODO remove!
    workers=10,
    callbacks=[tensorboard, checkpoint])
  
  # save model weights
  vae.save_weights(WEIGHT_FILE_NAME)

  finish_time = datetime.now().strftime('%Y%m%d-%H%M%S')
  print(f'Started at: {exec_time} | Finished at: {finish_time}')

  # 500 epochs, 100 steps
  # use 80 epochs, 500 steps or 100 epochs, 500 steps
  # for commander: used 50 epochs and 100 steps with arch3 weights as base

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--epochs', default = 100, type=int, help='number of epochs to train for')
  parser.add_argument('--steps', default = 500, type=int, help='number of steps per epoch')
  args = parser.parse_args()

  main(args)
