#!/usr/bin/env/python
from cosmosis.runtime.pipeline import LikelihoodPipeline
from cosmosis.runtime.config import Inifile
import numpy as np
from tqdm import tqdm

#################################################
# Read in the cosmosis configuration file pipeline.ini 
inifile = 'cosmosis_modules_4_training/powerspectra.ini'
ini = Inifile(inifile)  
#################################################

##################################################
# get the value of modules in the [pipeline] section
ini.get("pipeline","modules")

ini.set("pipeline","modules","sample_S8 sigma8toAs camb")
####################################################

####################################################
# setup the pipeline and give it the ini values
pipeline = LikelihoodPipeline(ini) 

# Get the fiducial values
params = pipeline.start_vector()
block = pipeline.run_parameters(params)
k  = block["matter_power_lin","k"]
print(params)

def spectra_generation(i):
    try:
        params[0] = params_linear['omch2'][i]
        params[1] = params_linear['obh2'][i]
        params[2] = params_linear['h'][i]
        params[3] = params_linear['n_s'][i]
        params[4] = params_linear['S8'][i]
        params[5] = params_linear['z'][i]
        params[6] = params_boost['log_T_AGN'][i]

        block    = pipeline.run_parameters(params)
        
        p_k_lin = block["matter_power_lin","p_k"].T
        p_k_nl = block["matter_power_nl","p_k"].T
        
        return p_k_lin[0],p_k_nl[0]/p_k_lin[0]

    except:
        print('something wrong with cosmosis')



params_linear = np.load('outputs/train_mead2020_feedback_parameter_linear.npz')
params_boost = np.load('outputs/train_mead2020_feedback_parameter_boost.npz')

P_lin = []
nl_boost = []
for i in tqdm(range(len(params_linear['omch2']))):
    spectra=spectra_generation(i)
    P_lin.append(spectra[0])
    nl_boost.append(spectra[1])


np.savez('outputs/linear_matter_mead2020_feedback', modes = k, features = P_lin)
np.savez('outputs/non_linear_boost_mead2020_feedback', modes = k, features = nl_boost)

np.savez('outputs/log10_linear_matter_mead2020_feedback', modes = k, features = np.log10(P_lin))
np.savez('outputs/log10_non_linear_boost_mead2020_feedback', modes = k, features = np.log10(nl_boost))




params_linear = np.load('outputs/test_mead2020_feedback_parameter_linear.npz')
params_boost = np.load('outputs/test_mead2020_feedback_parameter_boost.npz')

P_lin = []
nl_boost = []
for i in tqdm(range(len(params_linear['omch2']))):
    spectra=spectra_generation(i)
    P_lin.append(spectra[0])
    nl_boost.append(spectra[1])


np.savez('outputs/linear_matter_test_mead2020_feedback', modes = k, features = P_lin)
np.savez('outputs/non_linear_boost_test_mead2020_feedback', modes = k, features = nl_boost)

