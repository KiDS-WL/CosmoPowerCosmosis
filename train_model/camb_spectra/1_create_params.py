#!/usr/bin/env/python

# Author: A. Spurio Mancini


import numpy as np
import pyDOE as pyDOE

# number of parameters and samples

n_params = 7
n_samples = 180000

# parameter ranges
omch2 =     np.linspace(0.051,    0.255,   n_samples)

obh2 =      np.linspace(0.019, 0.026, n_samples)

h =         np.linspace(0.64,    0.82,    n_samples)

ns =        np.linspace(0.84,    1.1,    n_samples)

lnAs =      np.linspace(1.61,    3.91,    n_samples)

log_T_AGN =      np.linspace(7.3,      8.3,    n_samples)

z = np.linspace(0.0,      6.0,    n_samples)

# LHS grid

AllParams = np.vstack([omch2, obh2, h, ns, lnAs, log_T_AGN, z])
lhd = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
idx = (lhd * n_samples).astype(int)

AllCombinations = np.zeros((n_samples, n_params))
for i in range(n_params):
    AllCombinations[:, i] = AllParams[i][idx[:, i]]

# saving

params = {'omch2': AllCombinations[:, 0],
          'obh2': AllCombinations[:, 1],
          'h': AllCombinations[:, 2],
          'n_s': AllCombinations[:, 3],
          'ln10^{10}A_s': AllCombinations[:, 4],
          'z': AllCombinations[:, 6],
           }

np.savez('outputs/train_mead2020_feedback_parameter_linear.npz', **params)

params = {'omch2': AllCombinations[:, 0],
          'obh2': AllCombinations[:, 1],
          'h': AllCombinations[:, 2],
          'n_s': AllCombinations[:, 3],
          'ln10^{10}A_s': AllCombinations[:, 4],
          'log_T_AGN': AllCombinations[:, 5],
          'z': AllCombinations[:, 6],
           }

np.savez('outputs/train_mead2020_feedback_parameter_boost.npz', **params)







n_samples = 20000

# parameter ranges
omch2 =     np.linspace(0.051,    0.255,   n_samples)

obh2 =      np.linspace(0.019, 0.026, n_samples)

h =         np.linspace(0.64,    0.82,    n_samples)

ns =        np.linspace(0.84,    1.1,    n_samples)

lnAs =      np.linspace(1.61,    3.91,    n_samples)

log_T_AGN =      np.linspace(7.3,      8.3,    n_samples)

z = np.linspace(0.0,      6.0,    n_samples)

# LHS grid

AllParams = np.vstack([omch2, obh2, h, ns, lnAs, log_T_AGN, z])
lhd = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
idx = (lhd * n_samples).astype(int)

AllCombinations = np.zeros((n_samples, n_params))
for i in range(n_params):
    AllCombinations[:, i] = AllParams[i][idx[:, i]]

# saving

params = {'omch2': AllCombinations[:, 0],
          'obh2': AllCombinations[:, 1],
          'h': AllCombinations[:, 2],
          'n_s': AllCombinations[:, 3],
          'ln10^{10}A_s': AllCombinations[:, 4],
          'z': AllCombinations[:, 6],
           }

np.savez('outputs/test_mead2020_feedback_parameter_linear.npz', **params)

params = {'omch2': AllCombinations[:, 0],
          'obh2': AllCombinations[:, 1],
          'h': AllCombinations[:, 2],
          'n_s': AllCombinations[:, 3],
          'ln10^{10}A_s': AllCombinations[:, 4],
          'log_T_AGN': AllCombinations[:, 5],
          'z': AllCombinations[:, 6],
           }

np.savez('outputs/test_mead2020_feedback_parameter_boost.npz', **params)
