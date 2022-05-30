import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cosmopower import cosmopower_NN


cp_nn_lin = cosmopower_NN(restore=True, restore_filename='outputs/lin_matter_power_emulator')
cp_nn_boost = cosmopower_NN(restore=True, restore_filename='outputs/boost_matter_power_emulator')

params_linear = np.load('outputs/test_parameter_linear.npz')
params_boost = np.load('outputs/test_parameter_boost.npz')

predicted_testing_spectra = cp_nn_lin.ten_to_predictions_np(params_linear)
predicted_testing_boost = cp_nn_boost.predictions_np(params_boost)

testing_spectra = np.load('outputs/linear_matter_test.npz')
testing_boost = np.load('outputs/non_linear_boost_test.npz')

k_modes = testing_spectra['modes']

diff=np.abs((testing_spectra['features']-predicted_testing_spectra)/predicted_testing_spectra)

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

plt.legend(frameon=False, fontsize=30, loc='upper left')
plt.ylabel(r'$\frac{| P(k,z)^{\rm{true}} - P(k,z)^{\rm{emulated}} |} {P(k,z)^{\rm{emulated}} }$', fontsize=30)
plt.xlabel(r'$k [Mpc^{-1}]$',  fontsize=30)

plt.savefig('plots/linear_power_difference.jpg',dpi=200,bbox_inches='tight')
plt.close()





diff=np.abs((testing_boost['features']-predicted_testing_boost)/predicted_testing_boost)

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

plt.legend(frameon=False, fontsize=30, loc='upper left')
plt.ylabel(r'$\frac{| P(k,z)^{\rm{true}} - P(k,z)^{\rm{emulated}} |} {P(k,z)^{\rm{emulated}} }$', fontsize=30)
plt.xlabel(r'$k [Mpc^{-1}]$',  fontsize=30)

plt.savefig('plots/boost_difference.jpg',dpi=200,bbox_inches='tight')
plt.close()