import sys, os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from cosmopower import cosmopower_NN
from matplotlib import cm

power_spectra_version = 'mead2020_feedback'
## load emulators
cp_nn_lin = cosmopower_NN(restore=True, restore_filename='outputs/lin_matter_power_emulator_'+power_spectra_version)
cp_nn_log10_ratio_lin = cosmopower_NN(restore=True, restore_filename='outputs/log10_ratio_lin_matter_power_emulator_'+power_spectra_version)

cp_nn_non_lin = cosmopower_NN(restore=True, restore_filename='outputs/nonlin_matter_power_emulator_'+power_spectra_version)
cp_nn_log10_ratio_nonlin = cosmopower_NN(restore=True, restore_filename='outputs/log10_ratio_nonlin_matter_power_emulator_'+power_spectra_version)


#load reference spectra
reference_linear_spectra=np.load('outputs/center_linear_matter_'+power_spectra_version+'.npz')['features']
reference_non_linear_spectra=np.load('outputs/center_non_linear_matter_'+power_spectra_version+'.npz')['features']

## emulate test spectra
params_linear = np.load('outputs/test_parameter_linear.npz')
params_nonlinear = np.load('outputs/test_'+power_spectra_version+'_parameter_nonlinear.npz')

#variance scaling
training_features_linear_spectra = np.load('outputs/log10_linear_matter_'+power_spectra_version+'.npz')['features']
training_features_nonlinear_spectra = np.load('outputs/log10_non_linear_matter_'+power_spectra_version+'.npz')['features']
variance_linear = np.std(10**(training_features_linear_spectra),axis=0)
variance_nonlinear = np.std(10**(training_features_nonlinear_spectra),axis=0)

emulated_testing_linear_spectra = cp_nn_lin.ten_to_predictions_np(params_linear)
emulated_testing_linear_spectra_log10_ratio = 10**(cp_nn_log10_ratio_lin.predictions_np(params_linear)+np.log10(reference_linear_spectra))

emulated_testing_non_linear_spectra = cp_nn_non_lin.ten_to_predictions_np(params_nonlinear)
emulated_testing_non_linear_spectra_log10_ratio = 10**(cp_nn_log10_ratio_nonlin.predictions_np(params_nonlinear)+np.log10(reference_non_linear_spectra))

#load test spectra
testing_linear_spectra = np.load('outputs/linear_matter_test_'+power_spectra_version+'.npz')
testing_non_linear_spectra = np.load('outputs/non_linear_matter_test_'+power_spectra_version+'.npz')
testing_boost = np.load('outputs/non_linear_boost_test_'+power_spectra_version+'.npz')
k_modes = testing_linear_spectra['modes']

diff=np.abs((emulated_testing_linear_spectra-testing_linear_spectra['features'])/testing_linear_spectra['features'])
percentiles = np.zeros((4, diff.shape[1]))
percentiles[0] = np.percentile(diff, 68, axis = 0)
percentiles[1] = np.percentile(diff, 95, axis = 0)
percentiles[2] = np.percentile(diff, 99, axis = 0)
percentiles[3] = np.percentile(diff, 99.9, axis = 0)
plt.figure(figsize=(12, 9))
plt.fill_between(k_modes, 0, percentiles[2,:], color = 'salmon', label = '99%', alpha=0.8)
plt.fill_between(k_modes, 0, percentiles[1,:], color = 'red', label = '95%', alpha = 0.7)
plt.fill_between(k_modes, 0, percentiles[0,:], color = 'darkred', label = '68%', alpha = 1)
plt.xscale('log')
plt.ylim(0,0.003)
plt.legend(frameon=False, fontsize=30, loc='upper left')
plt.ylabel(r'$\frac{| P(k,z)^{\mathrm{emulated}}_\mathrm{lin} -P(k,z)^{\mathrm{true}}_\mathrm{lin} |} {P(k,z)^{\mathrm{true}}_\mathrm{lin} }$', fontsize=30)
plt.xlabel(r'$k [Mpc^{-1}]$',  fontsize=30)
plt.title(r'$\log_{10}P(k,z)$',  fontsize=30)
plt.savefig('plots/linear_power_difference_'+power_spectra_version+'.jpg',dpi=200,bbox_inches='tight')
plt.close()



diff=np.abs((emulated_testing_linear_spectra_log10_ratio-testing_linear_spectra['features'])/testing_linear_spectra['features'])
percentiles = np.zeros((4, diff.shape[1]))
percentiles[0] = np.percentile(diff, 68, axis = 0)
percentiles[1] = np.percentile(diff, 95, axis = 0)
percentiles[2] = np.percentile(diff, 99, axis = 0)
percentiles[3] = np.percentile(diff, 99.9, axis = 0)
plt.figure(figsize=(12, 9))
plt.fill_between(k_modes, 0, percentiles[2,:], color = 'salmon', label = '99%', alpha=0.8)
plt.fill_between(k_modes, 0, percentiles[1,:], color = 'red', label = '95%', alpha = 0.7)
plt.fill_between(k_modes, 0, percentiles[0,:], color = 'darkred', label = '68%', alpha = 1)
plt.xscale('log')
plt.ylim(0,0.003)
plt.legend(frameon=False, fontsize=30, loc='upper left')
plt.ylabel(r'$\frac{| P(k,z)^{\mathrm{emulated}} -P(k,z)^{\mathrm{true}} |} {P(k,z)^{\mathrm{true}} }$', fontsize=30)
plt.xlabel(r'$k [Mpc^{-1}]$',  fontsize=30)
plt.title(r'$\log_{10}P_\mathrm{lin}(k,z)- \log_{10}P^{center}(k,z)$',  fontsize=30)
plt.savefig('plots/linear_power_log10_ratio_difference_'+power_spectra_version+'.jpg',dpi=200,bbox_inches='tight')
plt.close()

#####
### non linear
#####


diff=np.abs((emulated_testing_non_linear_spectra-testing_non_linear_spectra['features'])/testing_non_linear_spectra['features'])
percentiles = np.zeros((4, diff.shape[1]))
percentiles[0] = np.percentile(diff, 68, axis = 0)
percentiles[1] = np.percentile(diff, 95, axis = 0)
percentiles[2] = np.percentile(diff, 99, axis = 0)
percentiles[3] = np.percentile(diff, 99.9, axis = 0)
plt.figure(figsize=(12, 9))
plt.fill_between(k_modes, 0, percentiles[2,:], color = 'salmon', label = '99%', alpha=0.8)
plt.fill_between(k_modes, 0, percentiles[1,:], color = 'red', label = '95%', alpha = 0.7)
plt.fill_between(k_modes, 0, percentiles[0,:], color = 'darkred', label = '68%', alpha = 1)
plt.xscale('log')
plt.ylim(0,0.003)
plt.legend(frameon=False, fontsize=30, loc='upper left')
plt.ylabel(r'$\frac{| P(k,z)^{\mathrm{emulated}} -P(k,z)^{\mathrm{true}} |} {P(k,z)^{\mathrm{true}} }$', fontsize=30)
plt.xlabel(r'$k [Mpc^{-1}]$',  fontsize=30)
plt.title(r'$\log_{10}P(k,z)$',  fontsize=30)
plt.savefig('plots/non_linear_power_difference_'+power_spectra_version+'.jpg',dpi=200,bbox_inches='tight')
plt.close()


diff=np.abs((emulated_testing_non_linear_spectra_log10_ratio-testing_non_linear_spectra['features'])/testing_non_linear_spectra['features'])
percentiles = np.zeros((4, diff.shape[1]))
percentiles[0] = np.percentile(diff, 68, axis = 0)
percentiles[1] = np.percentile(diff, 95, axis = 0)
percentiles[2] = np.percentile(diff, 99, axis = 0)
percentiles[3] = np.percentile(diff, 99.9, axis = 0)
plt.figure(figsize=(12, 9))
plt.fill_between(k_modes, 0, percentiles[2,:], color = 'salmon', label = '99%', alpha=0.8)
plt.fill_between(k_modes, 0, percentiles[1,:], color = 'red', label = '95%', alpha = 0.7)
plt.fill_between(k_modes, 0, percentiles[0,:], color = 'darkred', label = '68%', alpha = 1)
plt.xscale('log')
plt.ylim(0,0.003)
plt.legend(frameon=False, fontsize=30, loc='upper left')
plt.ylabel(r'$\frac{| P(k,z)^{\mathrm{emulated}} -P(k,z)^{\mathrm{true}} |} {P(k,z)^{\mathrm{true}} }$', fontsize=30)
plt.xlabel(r'$k [Mpc^{-1}]$',  fontsize=30)
plt.title(r'$\log_{10}P(k,z)- \log_{10}P^{center}(k,z)$',  fontsize=30)
plt.savefig('plots/non_linear_power_log10_ratio_difference_'+power_spectra_version+'.jpg',dpi=200,bbox_inches='tight')
plt.close()

