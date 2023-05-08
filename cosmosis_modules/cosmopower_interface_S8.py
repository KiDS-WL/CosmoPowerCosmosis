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
                            restore_filename=os.path.join(path_2_trained_emulator+'/log10_reference_lin_matter_power_emulator_mead2020_feedback'))

    config['nonlin_matter_power_cp'] = cp.cosmopower_NN(restore=True, 
                            restore_filename=os.path.join(path_2_trained_emulator+'/log10_reference_non_lin_matter_power_emulator_mead2020_feedback'))

    #load reference spectra
    config['reference_linear_spectra']=np.log10(np.load(path_2_trained_emulator+'/center_linear_matter_mead2020_feedback.npz')['features'])
    config['reference_nonlinear_spectra']=np.log10(np.load(path_2_trained_emulator+'/center_non_linear_matter_mead2020_feedback.npz')['features'])


    config['As_emulator'] = cp.cosmopower_NN(restore=True, 
                            restore_filename=os.path.join(path_2_trained_emulator+'/As_emulator'))


    # Return all this config information
    return config


def get_cosmopower_inputs(block, z, nz):

    # Get parameters from block and give them the
    # names and form that class expects

    if((block[cosmo, 'S_8_input']>1.0)or(block[cosmo, 'S_8_input']<0.5)):
        print('S8 value outside training range',block[cosmo, 'S_8_input'])
        exit()
    if((block[cosmo, 'n_s']>1.1)or(block[cosmo, 'n_s']<0.84)):
        print('n_s value outside training range',block[cosmo, 'n_s'])
        exit()
    if((block[cosmo, 'h0']>0.82)or(block[cosmo, 'h0']<0.64)):
        print('h0 value outside training range',block[cosmo, 'h0'])
        exit()
    if((block[cosmo, 'ombh2']>0.026)or(block[cosmo, 'ombh2']<0.019)):
        print('ombh2 value outside training range',block[cosmo, 'ombh2'])
        exit()
    if((block[cosmo, 'omch2']>0.255)or(block[cosmo, 'omch2']<0.051)):
        print('omch2 value outside training range',block[cosmo, 'omch2'])
        exit()
    if((z[nz-1]>6.0)or(z[0]<0)):
        print('z values are outside training range',z)
        exit()
    if((block.get_double(names.halo_model_parameters, 'logT_AGN')>8.3)or(block.get_double(names.halo_model_parameters, 'logT_AGN')<7.3)):
        print('logT_AGN value outside training range',block.get_double(names.halo_model_parameters, 'logT_AGN'))
        exit()
    if((block[cosmo, 'mnu']!=0.06)or(block[cosmo, 'omega_k']!=0.0)or(block[cosmo, 'w']!=-1.0)or(block[cosmo, 'wa']!=0.0)):
        print('either mnu!=0.06eV, or omega_k!=0.0, or w!=-1, or wa!=0, which were used for the training')
        exit()
              

    print('mnu: ',block[cosmo, 'mnu'])
    print('S8: ',block[cosmo, 'S_8_input'])
    print('n_s: ',block[cosmo, 'n_s'])
    print('h: ',block[cosmo, 'h0'])
    print('ombh2: ',block[cosmo, 'ombh2'])
    print('omch2: ',block[cosmo, 'omch2'])
    params_lin = {
        'S8':  [block[cosmo, 'S_8_input']]*nz,
        'n_s':       [block[cosmo, 'n_s']]*nz,
        'h':         [block[cosmo, 'h0']]*nz,
        'obh2':   [block[cosmo, 'ombh2']]*nz,
        'omch2': [block[cosmo, 'omch2']]*nz,
        'z':         z
    }

    print('halo Model:' ,block.get_double(names.halo_model_parameters, 'logT_AGN'))
    params_nonlin = {
        'S8':  [block[cosmo, 'S_8_input']]*nz,
        'n_s':           [block[cosmo, 'n_s']]*nz,
        'h':             [block[cosmo, 'h0']]*nz,
        'obh2':       [block[cosmo, 'ombh2']]*nz,
        'omch2':     [block[cosmo, 'omch2']]*nz,
        'z':             z,
        'log_T_AGN':     [block.get_double(names.halo_model_parameters, 'logT_AGN')]*nz
    }

    return params_lin, params_nonlin


def execute(block, config):

    h0 = block[cosmo, 'h0']
    
    z = block['NZ_SOURCE', 'z']
    nz = block['NZ_SOURCE', 'nz']

    block[distances, 'z'] = z
    block[distances, 'nz'] = nz

    print('nz',nz)

    #use k modes for cosmopower
    k = config['lin_matter_power_cp'].modes
    nk = len(k)
    
    params_lin,params_nonlin = get_cosmopower_inputs(block, z, nz)

    As=config['As_emulator'].predictions_np(params_nonlin)[0][0]
    #block[cosmo, "A_s"] = As
    print('As:',As,1e-10*np.exp(As))
    if(As>5.0):
        print('ln10^{10}A_s is greater than 5.0 and were impossible to calculate',As)
        exit()

    P_lin = config['lin_matter_power_cp'].predictions_np(params_lin)
    P_nl = config['nonlin_matter_power_cp'].predictions_np(params_nonlin)

    #subtract the reference spectra 
    for i in range(P_lin.shape[0]):
        P_lin[i] = P_lin[i]+config['reference_linear_spectra']
        P_nl[i] = P_nl[i]+config['reference_nonlinear_spectra']
    P_lin = 10**(P_lin)
    P_nl = 10**(P_nl)
    
    print(P_lin.shape)
    print(P_nl.shape)
    
    if(config['use_specific_k_modes']):
        k_new = np.logspace(np.log10(config['kmin']), np.log10(config['kmax']),num=config['nk'])

        P_lin_new = np.zeros(shape=(nz,len(k_new)))
        P_nl_new = np.zeros(shape=(nz,len(k_new)))
        for i in range(nz):
            P_lin_spline = CubicSpline(k,P_lin[i])
            P_nl_spline = CubicSpline(k,P_nl[i])
            P_lin_new[i] = P_lin_spline(k_new)
            P_nl_new[i] = P_nl_spline(k_new)
        P_lin = P_lin_new
        P_nl = P_nl_new

        k = k_new

    np.save('outputs/non_linear_spectrum',P_nl * h0**3)
    np.save('outputs/kh',k / h0)
    np.save('outputs/z', z)

    print(k.shape,z.shape,P_lin.shape)

    # Save matter power as a grid
    block.put_grid("matter_power_lin", "z", z,"k_h", k / h0, "p_k", P_lin * h0**3)
    block.put_grid("matter_power_nl", "z", z, "k_h", k / h0, "p_k" ,P_nl * h0**3)

    return 0

