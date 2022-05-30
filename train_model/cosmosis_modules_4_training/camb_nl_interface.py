from builtins import str
import os
from cosmosis.datablock import names, option_section
import sys
import traceback
import camb
import numpy as np

# add class directory to the path
dirname = os.path.split(__file__)[0]

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters
distances = names.distances

def setup(options):
    # Read options from the ini file which are fixed across
    # the length of the chain
    config = {
        'lmax': options.get_int(option_section, 'lmax', default=2000),
        'zmax': options.get_double(option_section, 'zmax', default=6.0),
        'dz': options.get_double(option_section, 'dz', default=0.05),
        'z_value': options.get_double(option_section, 'z_value', default=1.0),
        'kmax': options.get_double(option_section, 'kmax', default=100.0),
        'kmin': options.get_double(option_section, 'kmin', default=1e-5),
        'nk': options.get_int(option_section, 'nk', default=200),
        'halofit_version': options.get_string(option_section, 'halofit_version', default='mead')
    }

    for _, key in options.keys(option_section):
        if key.startswith('camb_'):
            config[key] = block[option_section, key]


    # Return all this config information
    return config


def get_camb_inputs(block, config):

    # Get parameters from block and give them the
    # names and form that class expects
    params = camb.set_params(ombh2 = block[cosmo, 'ombh2'],
                            omch2 = block[cosmo, 'omch2'],
                            H0 = block[cosmo, 'h0']*100,
                            ns = block[cosmo, 'n_s'],
                            As = block[cosmo, 'A_s'], 
                            omk=block[cosmo, 'omega_k'], 
                            lmax=config["lmax"]*2, 
                            WantTransfer=True,  
                            kmax=config["kmax"]*2,
                            num_massive_neutrinos=0, mnu=0.0, 
                            standard_neutrino_neff=block.get_double(cosmo, 'massless_nu', default=3.046),
                            tau=block.get_double(cosmo, 'tau', default=0.0925),
                            TCMB=block.get_double(cosmo, 't_cmb', default=2.726),
                            YHe=block.get_double(cosmo, 'YHe', default=0.25),
                            redshifts= np.arange(0.0, config['zmax'] + config['dz'], config['dz']).tolist()[::-1],
                            verbose=False,
                            halofit_version='mead',
                            HMCode_A_baryon=block.get_double(names.halo_model_parameters, 'A', default=2.32),
                            HMCode_eta_baryon=block.get_double(names.halo_model_parameters, 'eta0', default=0.76),
                            DoLensing=False
                            )

    return params


def execute(block, config):
    
    # Set input parameters
    params = get_camb_inputs(block, config)
    results = camb.get_results(params)

    # Define k,z we want to sample
    k = np.logspace(np.log10(config['kmin']), np.log10(config['kmax']), config['nk'])

    # Extract (interpolate) P(k,z) at the requested
    # sample points.
    PKcamb_lin = results.get_matter_power_interpolator(nonlinear=False,hubble_units=False,k_hunit=False)
    PKcamb_nl = results.get_matter_power_interpolator(nonlinear=True,hubble_units=False,k_hunit=False)

    # Save matter power as a grid
    z = [block.get_double(cosmo, 'z_value', default=2.0)]
    block.put_grid("matter_power_lin", "k", k, "z",z, "p_k", PKcamb_lin.P(z,k).T)
    block.put_grid("matter_power_nl", "k", k, "z",z, "p_k", PKcamb_nl.P(z,k).T)
    
    
    
    return 0


