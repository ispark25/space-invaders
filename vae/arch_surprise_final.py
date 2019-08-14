import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Activation
from keras.models import Model
from keras import backend as K

INPUT_DIM = (64,64,3)

# TODO for report:
# - compare different initialisations, adam, nadam, rmsprop
# - compare different layer setups: paper, source code, mine
# - mention about selu, elu, glorot uniform/normal, softplus, latent_dim, etc.

C_FILTERS    = [32,64,64,128,128]
C_KERN_SIZES = [3,3,3,3,3] # intuition, different parts of the game aren't really related to each other (localised)
C_STRIDES    = [2,2,2,2,2]
C_ACTIVS     = ['relu','relu','relu','relu','relu']
C_INITS      = ['he_uniform','he_uniform','he_uniform','he_uniform','he_uniform']
C_PADDINGS   = ['same', 'same', 'same', 'same', 'same'] # intuition: important for commandership as it lies on the edge

DENSE_SIZE = 1024

D_FILTERS    = [128,64,64,32,3]
D_KERN_SIZES = [3,3,3,3,3]
D_STRIDES    = [2,2,2,2,2]
D_ACTIVS     = ['relu','relu','relu','relu','sigmoid']
D_INITS      = ['he_uniform','he_uniform','he_uniform','he_uniform','he_uniform']
D_PADDINGS   = ['same', 'same', 'same', 'same', 'same']

Z_DIM = 64
KL_TOLERANCE = 0.5 # TODO: try to reduce this value after training, and try to do it from the start as well

def sampling(args):
    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_sigma), mean=0.,stddev=1.)
    return z_mean + z_sigma * epsilon

def to_sigma(z_log_var):
    return K.exp(z_log_var * 0.5)

class VAE():
    def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder = self.models[1]
        self.encoder_mu_log_var = self.models[2]
        self.decoder = self.models[3]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.kl_tolerance = KL_TOLERANCE


    def _build(self):
        vae_x = Input(shape=INPUT_DIM, name='observation_input')
        vae_c1 = Conv2D(filters=C_FILTERS[0], kernel_size=C_KERN_SIZES[0], strides=C_STRIDES[0], activation=C_ACTIVS[0], kernel_initializer=C_INITS[0], padding=C_PADDINGS[0], name='conv_layer_1')(vae_x)
        vae_c2 = Conv2D(filters=C_FILTERS[1], kernel_size=C_KERN_SIZES[1], strides=C_STRIDES[1], activation=C_ACTIVS[1], kernel_initializer=C_INITS[1], padding=C_PADDINGS[1], name='conv_layer_2')(vae_c1)
        vae_c3 = Conv2D(filters=C_FILTERS[2], kernel_size=C_KERN_SIZES[2], strides=C_STRIDES[2], activation=C_ACTIVS[2], kernel_initializer=C_INITS[2], padding=C_PADDINGS[2], name='conv_layer_3')(vae_c2)
        vae_c4 = Conv2D(filters=C_FILTERS[3], kernel_size=C_KERN_SIZES[3], strides=C_STRIDES[3], activation=C_ACTIVS[3], kernel_initializer=C_INITS[3], padding=C_PADDINGS[3], name='conv_layer_4')(vae_c3)
        vae_c5 = Conv2D(filters=C_FILTERS[4], kernel_size=C_KERN_SIZES[4], strides=C_STRIDES[4], activation=C_ACTIVS[4], kernel_initializer=C_INITS[4], padding=C_PADDINGS[4], name='conv_layer_5')(vae_c4)

        vae_z_in = Flatten()(vae_c5)

        vae_z_mean = Dense(Z_DIM, name='mu')(vae_z_in)
        vae_z_log_var = Dense(Z_DIM, name='log_var')(vae_z_in)
        vae_z_sigma = Lambda(to_sigma, name='sigma')(vae_z_log_var) # TODO mention!

        vae_z = Lambda(sampling, name='z')([vae_z_mean, vae_z_sigma])
        
        vae_z_input = Input(shape=(Z_DIM,), name='z_input')

        #### DECODER: we instantiate these layers separately so as to reuse them later
        d_shape = K.int_shape(vae_c5)

        vae_dense = Dense(d_shape[1] * d_shape[2] * d_shape[3], name='dense_layer')
        vae_z_out = Reshape((d_shape[1], d_shape[2], d_shape[3]), name='unflatten')
        vae_d1 = Conv2DTranspose(filters=D_FILTERS[0], kernel_size=D_KERN_SIZES[0], strides=D_STRIDES[0], activation=D_ACTIVS[0], kernel_initializer=D_INITS[0], padding=D_PADDINGS[0], name='deconv_layer_1')
        vae_d2 = Conv2DTranspose(filters=D_FILTERS[1], kernel_size=D_KERN_SIZES[1], strides=D_STRIDES[1], activation=D_ACTIVS[1], kernel_initializer=D_INITS[1], padding=D_PADDINGS[1], name='deconv_layer_2')
        vae_d3 = Conv2DTranspose(filters=D_FILTERS[2], kernel_size=D_KERN_SIZES[2], strides=D_STRIDES[2], activation=D_ACTIVS[2], kernel_initializer=D_INITS[2], padding=D_PADDINGS[2], name='deconv_layer_3')
        vae_d4 = Conv2DTranspose(filters=D_FILTERS[3], kernel_size=D_KERN_SIZES[3], strides=D_STRIDES[3], activation=D_ACTIVS[3], kernel_initializer=D_INITS[3], padding=D_PADDINGS[3], name='deconv_layer_4')
        vae_d5 = Conv2DTranspose(filters=D_FILTERS[4], kernel_size=D_KERN_SIZES[4], strides=D_STRIDES[4], activation=D_ACTIVS[4], kernel_initializer=D_INITS[4], padding=D_PADDINGS[4], name='deconv_layer_5')
        vae_d5 = Conv2DTranspose(filters=D_FILTERS[4], kernel_size=D_KERN_SIZES[4], strides=D_STRIDES[4], activation=D_ACTIVS[4], kernel_initializer=D_INITS[4], padding=D_PADDINGS[4], name='deconv_layer_5')
        
        #### DECODER IN FULL MODEL
        vae_dense_model = vae_dense(vae_z)
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4_model = vae_d4(vae_d3_model)
        vae_d5_model = vae_d5(vae_d4_model)

        #### DECODER ONLY
        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)
        vae_d5_decoder = vae_d5(vae_d4_decoder)

        #### MODELS
        vae_full = Model(vae_x, vae_d5_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_encoder_mu_log_var = Model(vae_x, (vae_z_mean, vae_z_log_var))
        vae_decoder = Model(vae_z_input, vae_d5_decoder)

        def vae_r_loss(y_true, y_pred):
            # int_shape(y_true) = (batch_size, 64, 64, 3)
            # surprise_squared = K.square(y_true - y_pred) * K.square(y_true - K.mean(y_true, axis=0))
            surprise_squared = K.square(y_true - y_pred) * K.exp(K.square(y_true - K.mean(y_true, axis=0)))
            r_loss = K.sum(surprise_squared, axis = [1,2,3]) * (64 * 64) # TODO mention!
            return r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = 1)
            kl_loss = K.maximum(kl_loss, KL_TOLERANCE * Z_DIM)
            return kl_loss

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)
        
        vae_full.compile(optimizer='adam', loss=vae_loss,  metrics=[vae_r_loss, vae_kl_loss]) 
        return (vae_full, vae_encoder, vae_encoder_mu_log_var, vae_decoder)
    
    def train(self, t_gen, v_gen, **kwargs):
        self.full_model.summary()
        self.full_model.fit_generator(t_gen, validation_data=v_gen, shuffle=True, **kwargs) # TODO mention!

    def set_weights(self, filepath):
        self.full_model.load_weights(filepath)

    def save_weights(self, filepath):
        self.full_model.save_weights(filepath)

    def save_encoder_weights(self, filepath):
        self.encoder.save_weights(filepath)

def load_vae(filename):
    vae = VAE()
    try:
        vae.set_weights(filename)
    except:
        print(f'ERROR: {filename} does not exist')
        raise
    return vae