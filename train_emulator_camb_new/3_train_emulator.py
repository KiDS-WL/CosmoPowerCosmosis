import numpy as np
import sys, os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from cosmopower import cosmopower_NN
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"


power_spectra_version = 'mead2020_feedback'

# checking that we are using a GPU
device = 'gpu:0' if tf.config.list_physical_devices('GPU') else 'cpu' #
print('using', device, 'device \n')

# setting the seed for reproducibility
np.random.seed(1)
tf.random.set_seed(2)

##load paramters and spectra
params_linear = np.load('outputs/train_parameter_linear.npz')
params_nonlinear = np.load('outputs/train_'+power_spectra_version+'_parameter_nonlinear.npz')


# Define the k-modes and features
k_modes = np.load('outputs/log10_linear_matter_'+power_spectra_version+'.npz')['modes']
training_features_linear_spectra = np.load('outputs/log10_linear_matter_'+power_spectra_version+'.npz')['features']
training_features_nonlinear_spectra = np.load('outputs/log10_non_linear_matter_'+power_spectra_version+'.npz')['features']


#load reference spectra
reference_linear_spectra=np.load('outputs/center_linear_matter_'+power_spectra_version+'.npz')['features']
reference_non_linear_spectra=np.load('outputs/center_non_linear_matter_'+power_spectra_version+'.npz')['features']


# instantiate NN class
cp_nn = cosmopower_NN(parameters=params_linear.files, 
                      modes=k_modes, 
                      n_hidden = [1048, 1048, 1048, 1048], # 4 hidden layers, each with 1048 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

with tf.device(device):
    # train
    cp_nn.train(training_parameters=params_linear,
                training_features=training_features_linear_spectra,
                filename_saved_model='outputs/lin_matter_power_emulator_'+power_spectra_version,
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                batch_sizes=[1000, 1000, 1000, 1000, 1000, 1000],
                gradient_accumulation_steps = [1, 1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000,1000],
                )


# instantiate NN class
cp_nn = cosmopower_NN(parameters=params_linear.files, 
                      modes=k_modes, 
                      n_hidden = [1048, 1048, 1048, 1048], # 4 hidden layers, each with 1048 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

training_features = training_features_linear_spectra-np.log10(reference_linear_spectra)
with tf.device(device):
    # train
    cp_nn.train(training_parameters=params_linear,
                training_features=training_features,
                filename_saved_model='outputs/log10_ratio_lin_matter_power_emulator_'+power_spectra_version,
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                batch_sizes=[1000, 1000, 1000, 1000, 1000, 1000],
                gradient_accumulation_steps = [1, 1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000,1000],
                )



# instantiate NN class
cp_nn = cosmopower_NN(parameters=params_nonlinear.files, 
                      modes=k_modes, 
                      n_hidden = [1048, 1048, 1048, 1048], # 4 hidden layers, each with 1048 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

with tf.device(device):
    # train
    cp_nn.train(training_parameters=params_nonlinear,
                training_features=training_features_nonlinear_spectra,
                filename_saved_model='outputs/nonlin_matter_power_emulator_'+power_spectra_version,
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                batch_sizes=[1000, 1000, 1000, 1000, 1000, 1000],
                gradient_accumulation_steps = [1, 1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000,1000],
 )

# instantiate NN class
cp_nn = cosmopower_NN(parameters=params_nonlinear.files, 
                      modes=k_modes, 
                      n_hidden = [1048, 1048, 1048, 1048], # 4 hidden layers, each with 1048 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

training_features = training_features_nonlinear_spectra-np.log10(reference_non_linear_spectra)
with tf.device(device):
    # train
    cp_nn.train(training_parameters=params_nonlinear,
                training_features=training_features,
                filename_saved_model='outputs/log10_variance_nonlin_matter_power_emulator_'+power_spectra_version,
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                batch_sizes=[1000, 1000, 1000, 1000, 1000, 1000],
                gradient_accumulation_steps = [1, 1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000,1000],
                )


