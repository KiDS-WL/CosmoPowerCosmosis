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


# Define the k-modes and features
k_modes = np.load('outputs/linear_matter_'+power_spectra_version+'.npz')['modes']
available_indices = np.load('outputs/linear_matter_'+power_spectra_version+'.npz')['available_indices']
training_features_linear_spectra = np.load('outputs/linear_matter_'+power_spectra_version+'.npz')['features']
training_features_nonlinear_spectra = np.load('outputs/non_linear_matter_'+power_spectra_version+'.npz')['features']

print(available_indices.shape)

#load reference spectra
reference_linear_spectra=np.load('outputs/center_linear_matter_'+power_spectra_version+'.npz')['features']
reference_non_linear_spectra=np.load('outputs/center_non_linear_matter_'+power_spectra_version+'.npz')['features']

##load paramters and spectra
params = np.load('outputs/train_'+power_spectra_version+'_parameter.npz')

param_names_lin = ['omch2', 'obh2', 'h', 'n_s', 'sigma8', 'z']
params_lin = {}
for name in param_names_lin:
    params_lin[name]=params[name][available_indices]

# instantiate NN class
cp_nn = cosmopower_NN(parameters=param_names_lin, 
                      modes=k_modes, 
                      n_hidden = [1048, 1048, 1048, 1048], # 4 hidden layers, each with 1048 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

training_features = np.log10(training_features_linear_spectra)-np.log10(reference_linear_spectra)
with tf.device(device):
    # train
    cp_nn.train(training_parameters=params_lin,
                training_features=training_features,
                filename_saved_model='outputs/log10_reference_lin_matter_power_emulator_'+power_spectra_version,
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                batch_sizes=[1000, 1000, 1000, 1000, 1000, 1000],
                gradient_accumulation_steps = [1, 1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000,1000],
                )


param_names_nonlin = ['omch2', 'obh2', 'h', 'n_s', 'sigma8', 'z', 'log_T_AGN']
params_nonlin = {}
for name in param_names_nonlin:
    params_nonlin[name]=params[name][available_indices]

# instantiate NN class
cp_nn = cosmopower_NN(parameters=param_names_nonlin, 
                      modes=k_modes, 
                      n_hidden = [1048, 1048, 1048, 1048], # 4 hidden layers, each with 1048 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

training_features = np.log10(training_features_nonlinear_spectra)-np.log10(reference_non_linear_spectra)
with tf.device(device):
    # train
    cp_nn.train(training_parameters=params_nonlin,
                training_features=training_features,
                filename_saved_model='outputs/log10_reference_non_lin_matter_power_emulator_'+power_spectra_version,
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                batch_sizes=[1000, 1000, 1000, 1000, 1000, 1000],
                gradient_accumulation_steps = [1, 1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000,1000],
                )
