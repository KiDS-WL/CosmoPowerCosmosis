import numpy as np
from cosmopower import cosmopower_NN
import tensorflow as tf

# checking that we are using a GPU
device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu' #
print('using', device, 'device \n')

# setting the seed for reproducibility
np.random.seed(1)
tf.random.set_seed(2)

##load paramters and spectra
params_linear = np.load('outputs/train_parameter_linear.npz')
params_boost = np.load('outputs/train_parameter_boost.npz')

linear_matter = np.load('outputs/linear_matter.npz')
non_linear_boost = np.load('outputs/non_linear_boost.npz')

log10_linear_matter = np.load('outputs/log10_linear_matter.npz')
log10_non_linear_boost = np.load('outputs/log10_non_linear_boost.npz')

# Define the k-modes and features
k_modes = log10_linear_matter['modes']
training_features_spectra = log10_linear_matter['features']
training_features_boost = non_linear_boost['features']



# instantiate NN class
cp_nn = cosmopower_NN(parameters=params_linear.files, 
                      modes=k_modes, 
                      n_hidden = [512, 512, 512, 512], # 4 hidden layers, each with 512 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

with tf.device(device):
    # train
    cp_nn.train(training_parameters=params_linear,
                training_features=training_features_spectra,
                filename_saved_model='outputs/lin_matter_power_emulator',
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[200, 200, 200, 200, 200],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000],
                )



# instantiate NN class
cp_nn = cosmopower_NN(parameters=params_boost.files, 
                      modes=k_modes, 
                      n_hidden = [512, 512, 512, 512], # 4 hidden layers, each with 512 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

with tf.device(device):
    # train
    cp_nn.train(training_parameters=params_boost,
                training_features=training_features_boost,
                filename_saved_model='outputs/boost_matter_power_emulator',
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[200, 200, 200, 200, 200],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000],
                )