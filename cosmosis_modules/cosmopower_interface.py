from builtins import str
import os
from cosmosis.datablock import names, option_section
import sys
import traceback
from scipy.interpolate import CubicSpline
from scipy.interpolate import RectBivariateSpline 
import cosmopower as cp


import cosmopower
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to use CPU if GPU is avalible otherwise the GPU memory runs out of memory. It also does not slower the predtiction.

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters
distances = names.distances


def get_optional_params(block, section, names):
    """Get values from a datablock from a list of names.
    
    If the entries of names are tuples or lists of length 2, they are assumed
    to correspond to (cosmosis_name, output_name), where cosmosis_name is the 
    datablock key and output_name the params dict key."""
    params = {}    
    for name in names:
        cosmosis_name = name
        output_name = name
        if isinstance(name, (list, tuple)):
            if len(name) == 2 and isinstance(name[1], str):
                # Output name specified
                output_name = name[1]
                cosmosis_name = name[0]
        if block.has_value(section, cosmosis_name):
            params[output_name] = block[section, cosmosis_name]
    return params


def setup(options):

    config = {
        'kmax': options.get_double(option_section, 'kmax', default=10.0),
        'kmin': options.get_double(option_section, 'kmin', default=1e-5),
        'nk': options.get_int(option_section, 'nk', default=200),
        'use_specific_k_modes': options.get_bool(option_section, 'use_specific_k_modes', default=False),
    }

    for _, key in options.keys(option_section):
        if key.startswith('cosmopower_'):
            config[key] = block[option_section, key]
    
    # Create the object that connects to cosmopower
    # load pre-trained NN model: maps cosmological parameters to linear log-P(k)
    path_2_trained_emulator = options.get_string(option_section, 'path_2_trained_emulator')
    config['lin_matter_power_cp'] = cp.cosmopower_NN(restore=True, 
                            restore_filename=os.path.join(path_2_trained_emulator+'/trained_models/CP_paper/PK/PKLIN_NN'))

    config['nl_matter_power_boost_cp'] = cp.cosmopower_NN(restore=True, 
                            restore_filename=os.path.join(path_2_trained_emulator+'/trained_models/CP_paper/PK/PKNLBOOST_NN'))

    # Return all this config information
    return config


def get_cosmopower_inputs(block, z, nz):

    # Get parameters from block and give them the
    # names and form that class expects

    params_lin = {
        'ln10^{10}A_s':  [np.log(block[cosmo, 'A_s']*10**10)]*nz,
        'n_s':       [block[cosmo, 'n_s']]*nz,
        'h':         [block[cosmo, 'h0']]*nz,
        'omega_b':   [block[cosmo, 'ombh2']]*nz,
        'omega_cdm': [block[cosmo, 'omch2']]*nz,
        'z':         z
    }


    print('halo Model:' ,block.get_double(names.halo_model_parameters, 'A', default=2.32),block.get_double(names.halo_model_parameters, 'eta0', default=0.76))
    params_boost = {
        'ln10^{10}A_s':  [np.log(block[cosmo, 'A_s']*10**10)]*nz,
        'n_s':           [block[cosmo, 'n_s']]*nz,
        'h':             [block[cosmo, 'h0']]*nz,
        'omega_b':       [block[cosmo, 'ombh2']]*nz,
        'omega_cdm':     [block[cosmo, 'omch2']]*nz,
        'z':             z,
        'c_min':         [block.get_double(names.halo_model_parameters, 'A', default=2.32)]*nz,
        'eta_0':         [block.get_double(names.halo_model_parameters, 'eta0', default=0.76)]*nz
    }

    return params_lin, params_boost


def execute(block, config):

    h0 = block[cosmo, 'h0']
    
    z = block['NZ_SOURCE', 'z']
    nz = block['NZ_SOURCE', 'nz']

    block[distances, 'z'] = z
    block[distances, 'nz'] = nz

    #use k modes for cosmopower
    k = config['lin_matter_power_cp'].modes
    nk = len(k)
    
    params_lin,params_boost = get_cosmopower_inputs(block, z, nz)
    P_lin = config['lin_matter_power_cp'].ten_to_predictions_np(params_lin).T
    P_nl = P_lin*config['nl_matter_power_boost_cp'].ten_to_predictions_np(params_boost).T
    
    if(config['use_specific_k_modes']):
        k_new = np.logspace(np.log10(config['kmin']), np.log10(config['kmax']),num=config['nk'])
        
        # P_lin_spline = RectBivariateSpline(k,z,P_lin)
        # P_nl_spline = RectBivariateSpline(k,z,P_nl)
        # P_lin = P_lin_spline(k_new,z)
        # P_nl = P_nl_spline(k_new,z)

        P_lin_new = np.zeros(shape=(len(k_new),nz))
        P_nl_new = np.zeros(shape=(len(k_new),nz))
        for i in range(nz):
            P_lin_spline = CubicSpline(k,P_lin[:,i])
            P_nl_spline = CubicSpline(k,P_nl[:,i])
            P_lin_new[:,i] = P_lin_spline(k_new)
            P_nl_new[:,i] = P_nl_spline(k_new)
        P_lin = P_lin_new
        P_nl = P_nl_new

        k = k_new

    # Save matter power as a grid
    block.put_grid("matter_power_lin", "k_h", k / h0, "z", z, "p_k", P_lin * h0**3)
    block.put_grid("matter_power_nl", "k_h", k / h0, "z", z, "p_k", P_nl * h0**3)

    return 0

