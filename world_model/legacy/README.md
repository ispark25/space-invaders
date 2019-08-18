## Legacy files

01_generate_data_old: deprecated version of the script, applied quality checks
to the data (more than 600 steps, etc)

01_generate_data_test: a test script which tested the WM infrastructure, from
taking a picture as input from the environment, turning it into a latent vector z
feeding it into the RNN and producing a prediction from which we sample from to
get a latent vector and other predicted variables.

02_preprocess_data: deprecated, was merged into the new 01_generate_data

05_train_rnn: First change: data input was processed into batches of shape
(batch_size, seq_length,latent_dims). However, due to the architecture, we
decided to go back to online learning and process games one at a time

05_train_rnn_old: changed the batch generation to be in one function

05_train_rnn_not_cleaned: conversion of action discrete space into a continuous
one

05_train_rnn_clean_single: online learning, used a different batch learning
algorithm

05_train_rnn_clean_single_cumrew: applied cumulative reward instead of normal
reward as input

05_train_rnn_clean_single_cumrew_only: removed normal reward, trains only with
cumulative reward, in an attempt for training a dreams environment
