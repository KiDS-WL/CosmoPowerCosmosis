from builtins import str
import os
from cosmosis.datablock import names, option_section
import sys
import traceback
from scipy.interpolate import CubicSpline
from scipy.interpolate import RectBivariateSpline 
import cosmopower as cp
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to use CPU if GPU is avalible otherwise the GPU memory runs out of memory. It also does not slower the predtiction.

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters

# This interface requires the pre-trained CMB emulator by Alessio Spurio Mancini (https://arxiv.org/abs/2106.03846). 
# See: https://github.com/alessiospuriomancini/cosmopower/tree/main/cosmopower/trained_models/CP_paper/CMB

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

    config = {}
    for _, key in options.keys(option_section):
        if key.startswith('cosmopower_'):
            config[key] = block[option_section, key]
    
    # Create the object that connects to cosmopower
    path_2_trained_emulator = options.get_string(option_section, 'path_2_trained_emulator')
    config['TT'] = cp.cosmopower_NN(restore=True, restore_filename=os.path.join(path_2_trained_emulator+'/cmb_TT_NN'))
    config['TE'] = cp.cosmopower_PCAplusNN(restore=True, restore_filename=os.path.join(path_2_trained_emulator+'/cmb_TE_PCAplusNN'))
    config['EE'] = cp.cosmopower_NN(restore=True, restore_filename=os.path.join(path_2_trained_emulator+'/cmb_EE_NN'))
    config['PP'] = cp.cosmopower_PCAplusNN(restore=True, restore_filename=os.path.join(path_2_trained_emulator+'/cmb_PP_PCAplusNN'))

    # Return all this config information
    return config


def get_cosmopower_inputs(block):

    # Get parameters from block and give them the
    # names and form that cosmopower expects
    params = {
        'omega_b':  [block[cosmo, 'ombh2']],
        'omega_cdm': [block[cosmo, 'omch2']],
        'h':         [block[cosmo, 'h0']],
        'tau_reio':   [block[cosmo, 'tau']],
        'n_s': [block[cosmo, 'n_s']],
        'ln10^{10}A_s': [block[cosmo, 'A_s']]
    }
    # Note: the p(k) emulator writes ln10^{10}A_s to the datablock as 'A_s'. Wonderful naming convention!
    return params


def execute(block, config):

    params = get_cosmopower_inputs(block)
    cmb_unit = (2.7255e6)**2 #muK
    tt_spectra = config['TT'].ten_to_predictions_np(params) * cmb_unit
    te_spectra = config['TE'].predictions_np(params) * cmb_unit
    ee_spectra = config['EE'].ten_to_predictions_np(params) * cmb_unit
    pp_spectra = config['PP'].ten_to_predictions_np(params)
    ell = config['TT'].modes
    # Planck likelihood requires (ell*(ell+1))/(2*pi) C_ell
    block["cmb_cl", "tt"] = tt_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block["cmb_cl", "te"] = te_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block["cmb_cl", "ee"] = ee_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block["cmb_cl", "pp"] = pp_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block["cmb_cl", "ell"] = ell

    return 0

