from cosmosis.datablock import names, option_section as opt
from cosmosis.datablock.cosmosis_py import errors
import numpy as np
import warnings
import traceback
import contextlib
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline, CubicSpline

# Finally we can now import camb
import camb

import cosmopower as cp
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to use CPU if GPU is avalible otherwise the GPU memory runs out of memory. It also does not slower the predtiction.

cosmo = names.cosmological_parameters

MODE_BG = 'background'
MODE_THERM = 'thermal'
MODE_CMB = 'cmb'
MODE_POWER = 'power'
MODE_ALL = 'all'
MODES = [MODE_BG, MODE_THERM, MODE_CMB, MODE_POWER, MODE_ALL]

DEFAULT_A_S = 2.1e-9

C_KMS = 299792.458

# See this table for description:
#https://camb.readthedocs.io/en/latest/transfer_variables.html#transfer-variables
matter_power_section_names = {
    'delta_cdm': 'dark_matter_power',
    'delta_baryon': 'baryon_power',
    'delta_photon': 'photon_power',
    'delta_neutrino': 'massless_neutrino_power',
    'delta_nu': 'massive_neutrino_power',
    'delta_tot': 'matter_power',
    'delta_nonu': 'cdm_baryon_power',
    'delta_tot_de': 'matter_de_power',
    'weyl': 'weyl_curvature_power',
    'v_newtonian_cdm': 'cdm_velocity_power',
    'v_newtonian_baryon': 'baryon_velocity_power',
    'v_baryon_cdm': 'baryon_cdm_relative_velocity_power',
}

@contextlib.contextmanager
def be_quiet_camb():
    original_feedback_level = camb.config.FeedbackLevel
    try:
        camb.set_feedback_level(0)
        yield
    finally:
        camb.set_feedback_level(original_feedback_level)


def get_optional_params(block, section, names):
    params = {}

    for name in names:
        # For some parameters we just use the Camb name as the parameter
        # name, but for other we specify a simpler one
        if isinstance(name, (list, tuple)):
            cosmosis_name, output_name = name
        else:
            cosmosis_name = name
            output_name = name

        # We don't try to set our own default for these parameters,
        # we just let camb decide them on its own
        if block.has_value(section, cosmosis_name):
            params[output_name] = block[section, cosmosis_name]
    return params

def get_choice(options, name, valid, default=None, prefix=''):
    choice = options.get_string(opt, name, default=default)
    if choice not in valid:
        raise ValueError("Parameter setting '{}' in camb must be one of: {}.  You tried: {}".format(name, valid, choice))
    return prefix + choice


def make_z_for_pk(more_config):
    if 'zmid' in more_config:
        z = np.concatenate((np.linspace(more_config['zmin'], 
                                        more_config['zmid'], 
                                        more_config['nz_mid'], 
                                        endpoint=False),
                            np.linspace(more_config['zmid'], 
                                        more_config['zmax'], 
                                        more_config['nz']-more_config['nz_mid'])))[::-1]
    else:
        z = np.linspace(more_config['zmin'], more_config['zmax'], more_config['nz'])[::-1]

    return z


def setup(options):
    mode = options.get_string(opt, 'mode', default='all')
    if not mode in MODES:
        raise ValueError("Unknown mode {}.  Must be one of: {}".format(mode, MODES))

    # These are parameters for CAMB
    config = {}

    # These are parameters that we do not pass directly to CAMBparams,
    # but use ourselves in some other way
    more_config = {}

    more_config['mode'] = mode
    more_config['max_printed_errors'] = options.get_int(opt, 'max_printed_errors', default=20)
    more_config['n_printed_errors'] = 0
    config['WantCls'] = mode in [MODE_CMB, MODE_ALL]
    config['WantTransfer'] = mode in [MODE_POWER, MODE_ALL]
    config['WantScalars'] = True
    config['WantTensors'] = options.get_bool(opt, 'do_tensors', default=False)
    config['WantVectors'] = options.get_bool(opt, 'do_vectors', default=False)
    config['WantDerivedParameters'] = True
    config['Want_cl_2D_array'] = False
    config['Want_CMB'] = config['WantCls']
    config['DoLensing'] = options.get_bool(opt, 'do_lensing', default=False)
    config['NonLinear'] = get_choice(options, 'nonlinear', ['none', 'pk', 'lens', 'both'], 
                                     default='none' if mode in [MODE_BG, MODE_THERM] else 'both', 
                                     prefix='NonLinear_')

    config['scalar_initial_condition'] = 'initial_' + options.get_string(opt, 'initial', default='adiabatic')
    
    config['want_zdrag'] = mode != MODE_BG
    config['want_zstar'] = config['want_zdrag']

    more_config['want_chistar'] = options.get_bool(opt, 'want_chistar', default=True)
    more_config['n_logz'] = options.get_int(opt, 'n_logz', default=0)
    more_config['zmax_logz'] = options.get_double(opt, 'zmax_logz', default = 1100.)
    
    more_config['lmax_params'] = get_optional_params(options, opt, ['max_eta_k', 'lens_potential_accuracy',
                                                                    'lens_margin', 'k_eta_fac', 'lens_k_eta_reference',
                                                                    #'min_l', 'max_l_tensor', 'Log_lvalues', , 'max_eta_k_tensor'
                                                                     ])
    # lmax is required
    more_config['lmax_params']['lmax'] = options.get_int(opt, 'lmax', default=2600)                                                  
    
    more_config['initial_power_params'] = get_optional_params(options, opt, ['pivot_scalar', 'pivot_tensor'])

    more_config['cosmology_params'] = get_optional_params(options, opt, ['neutrino_hierarchy' ,'theta_H0_range'])

    if 'theta_H0_range' in more_config['cosmology_params']:
        more_config['cosmology_params']['theta_H0_range'] = [float(x) for x in more_config['cosmology_params']['theta_H0_range'].split()]

    more_config['do_reionization'] = options.get_bool(opt, 'do_reionization', default=True)
    more_config['use_optical_depth'] = options.get_bool(opt, 'use_optical_depth', default=True)
    more_config['reionization_params'] = get_optional_params(options, opt, ['include_helium_fullreion', 'tau_solve_accuracy_boost', 
                                                                            ('tau_timestep_boost','timestep_boost'), ('tau_max_redshift', 'max_redshift')])
    
    more_config['use_tabulated_w'] = options.get_bool(opt, 'use_tabulated_w', default=False)
    more_config['use_ppf_w'] = options.get_bool(opt, 'use_ppf_w', default=False)
    more_config['do_bao'] = options.get_bool(opt, 'do_bao', default=True)
    
    more_config['nonlinear_params'] = get_optional_params(options, opt, ['halofit_version', 'Min_kh_nonlinear'])

    halofit_version = more_config['nonlinear_params'].get('halofit_version')
    known_halofit_versions = list(camb.nonlinear.halofit_version_names.keys())
    if halofit_version not in known_halofit_versions:
        raise ValueError("halofit_version must be one of : {}.  You put: {}".format(known_halofit_versions, halofit_version))

    more_config['accuracy_params'] = get_optional_params(options, opt, 
                                                        ['AccuracyBoost', 'lSampleBoost', 'lAccuracyBoost', 'DoLateRadTruncation'])
                                                        #  'TimeStepBoost', 'BackgroundTimeStepBoost', 'IntTolBoost', 
                                                        #  'SourcekAccuracyBoost', 'IntkAccuracyBoost', 'TransferkBoost',
                                                        #  'NonFlatIntAccuracyBoost', 'BessIntBoost', 'LensingBoost',
                                                        #  'NonlinSourceBoost', 'BesselBoost', 'LimberBoost', 'SourceLimberBoost',
                                                        #  'KmaxBoost', 'neutrino_q_boost', 'AccuratePolarization', 'AccurateBB',  
                                                        #  'AccurateReionization'])

    more_config['zmin'] = options.get_double(opt, 'zmin', default=0.0)
    more_config['zmax'] = options.get_double(opt, 'zmax', default=3.01)
    more_config['nz'] = options.get_int(opt, 'nz', default=150)
    more_config.update(get_optional_params(options, opt, ['zmid', 'nz_mid']))

    more_config['zmin_background'] = options.get_double(opt, 'zmin_background', default=more_config['zmin'])
    more_config['zmax_background'] = options.get_double(opt, 'zmax_background', default=more_config['zmax'])
    more_config['nz_background'] = options.get_int(opt, 'nz_background', default=more_config['nz'])

    more_config['transfer_params'] = get_optional_params(options, opt, ['k_per_logint', 'accurate_massive_neutrino_transfers'])
    # Adjust CAMB defaults
    more_config['transfer_params']['kmax'] = options.get_double(opt, 'kmax', default=10.0)
    # more_config['transfer_params']['high_precision'] = options.get_bool(opt, 'high_precision', default=True)

    more_config['kmin'] = options.get_double(opt, 'kmin', default=1e-5)
    more_config['kmax'] = options.get_double(opt, 'kmax', more_config['transfer_params']['kmax'])
    more_config['kmax_extrapolate'] = options.get_double(opt, 'kmax_extrapolate', default=more_config['kmax'])
    more_config['nk'] = options.get_int(opt, 'nk', default=200)
    more_config['use_specific_k_modes'] = options.get_bool(opt, 'use_specific_k_modes', default=False)
    
    # Create the object that connects to cosmopower
    # load pre-trained NN model: maps cosmological parameters to linear log-P(k)
    if config['WantTransfer']:
        more_config['lin_matter_power_cp'] = cp.cosmopower_NN(restore=True, restore_filename=options.get_string(opt, 'lin_matter_power_emulator'))
        more_config['nonlin_matter_power_cp'] = cp.cosmopower_NN(restore=True, restore_filename=options.get_string(opt, 'nonlin_matter_power_emulator')) if config['NonLinear'] != 'NonLinear_none' else None

        if halofit_version == 'mead2020_feedback':
            more_config['reference_linear_spectra'] = np.log10(np.load(options.get_string(opt, 'reference_linear_spectra'))['features'])
            more_config['reference_nonlinear_spectra'] = np.log10(np.load(options.get_string(opt, 'reference_nonlinear_spectra'))['features']) if config['NonLinear'] != 'NonLinear_none' else None
            more_config['As_emulator'] = cp.cosmopower_NN(restore=True, restore_filename=options.get_string(opt, 'As_emulator')) if options.has_value(opt, 'As_emulator') else None
        else:
            more_config['reference_linear_spectra'] = None
            more_config['reference_nonlinear_spectra'] = None
            more_config['As_emulator'] = None
                            
                            
    # CosmoPower cls
    if config['WantCls']:
        more_config['TT'] = cp.cosmopower_NN(restore=True, restore_filename=options.get_string(opt, 'TT_emulator'))
        more_config['TE'] = cp.cosmopower_PCAplusNN(restore=True, restore_filename=options.get_string(opt, 'TE_emulator'))
        more_config['EE'] = cp.cosmopower_NN(restore=True, restore_filename=options.get_string(opt, 'EE_emulator'))
        more_config['PP'] = cp.cosmopower_PCAplusNN(restore=True, restore_filename=options.get_string(opt, 'PP_emulator'))

    more_config['power_spectra'] = options.get_string(opt, 'power_spectra', default='delta_tot').split()
    bad_power = []
    for p in more_config['power_spectra']:
        if p not in matter_power_section_names:
            bad_power.append(p)
    if bad_power:
        bad_power = ", ".join(bad_power)
        good_power = ", ".join(matter_power_section_names.keys())
        raise ValueError("""These matter power types are not known: {}.
            Please use any these (separated by spaces): {}""".format(bad_power, good_power))

    camb.set_feedback_level(level=options.get_int(opt, 'feedback', default=0))
    return [config, more_config]


def get_cosmopower_inputs(block, z, nz, config, more_config):
    version = more_config['nonlinear_params'].get('halofit_version', '')

    def check_range(value, min_val, max_val, name):
        if value < min_val or value > max_val:
            raise Exception(f"{name} value outside training range {value}")

    # Check cosmological parameters
    check_range(block[cosmo, 'mnu'], 0.06, 0.06, "mnu")
    check_range(block[cosmo, 'omega_k'], 0.0, 0.0, "omega_k")
    check_range(block[cosmo, 'w'], -1.0, -1.0, "w")
    check_range(block[cosmo, 'wa'], 0.0, 0.0, "wa")
    check_range(block[cosmo, 'n_s'], 0.84, 1.1, "n_s")
    check_range(block[cosmo, 'h0'], 0.64, 0.82, "h0")
    check_range(block[cosmo, 'ombh2'], 0.019, 0.026, "ombh2")
    check_range(block[cosmo, 'omch2'], 0.051, 0.255, "omch2")
    check_range(z[-1], 0, 6.0, "z")
    check_range(z[0], 0, 6.0,  "z")

    params_lin = params_nonlin = params_boost = None

    if version == 'mead2015':
        if config['NonLinear'] != 'NonLinear_none':
            check_range(block[cosmo, 'A'], 2, 4, "A")
            check_range(block[cosmo, 'eta'], 0.5, 1.0, "eta")

        if block.has_value(cosmo, 'A_s') and not block.has_value(cosmo, 'S_8_input'):
            check_range(np.log(block[cosmo, 'A_s'] * 10**10), 1.61, 3.91, "A_s")
            params_lin = {
                'ln10^{10}A_s': [np.log(block[cosmo, 'A_s'] * 10**10)] * nz,
                'n_s': [block[cosmo, 'n_s']] * nz,
                'h': [block[cosmo, 'h0']] * nz,
                'omega_b': [block[cosmo, 'ombh2']] * nz,
                'omega_cdm': [block[cosmo, 'omch2']] * nz,
                'z': z
            }
            if config['NonLinear'] != 'NonLinear_none':
                params_boost = {
                    'ln10^{10}A_s': [np.log(block[cosmo, 'A_s'] * 10**10)] * nz,
                    'n_s': [block[cosmo, 'n_s']] * nz,
                    'h': [block[cosmo, 'h0']] * nz,
                    'omega_b': [block[cosmo, 'ombh2']] * nz,
                    'omega_cdm': [block[cosmo, 'omch2']] * nz,
                    'z': z,
                    'c_min': [block.get_double(names.halo_model_parameters, 'A')] * nz,
                    'eta_0': [block.get_double(names.halo_model_parameters, 'eta')] * nz
                }

        elif not block.has_value(cosmo, 'A_s') and block.has_value(cosmo, 'S_8_input'):
            check_range(block[cosmo, 'S_8_input'], 0.5, 1.0, "S_8_input")
            params_lin = {
                'S_8': [block[cosmo, 'S_8_input']] * nz,
                'n_s': [block[cosmo, 'n_s']] * nz,
                'h': [block[cosmo, 'h0']] * nz,
                'omega_b': [block[cosmo, 'ombh2']] * nz,
                'omega_cdm': [block[cosmo, 'omch2']] * nz,
                'z': z,
                'c_min': [block.get_double(names.halo_model_parameters, 'A')] * nz,
                'eta_0': [block.get_double(names.halo_model_parameters, 'eta')] * nz
            }
            if config['NonLinear'] != 'NonLinear_none':
                params_boost = {
                    'S_8': [block[cosmo, 'S_8_input']] * nz,
                    'n_s': [block[cosmo, 'n_s']] * nz,
                    'h': [block[cosmo, 'h0']] * nz,
                    'omega_b': [block[cosmo, 'ombh2']] * nz,
                    'omega_cdm': [block[cosmo, 'omch2']] * nz,
                    'z': z,
                    'c_min': [block.get_double(names.halo_model_parameters, 'A')] * nz,
                    'eta_0': [block.get_double(names.halo_model_parameters, 'eta')] * nz
                }

    elif version == 'mead2020_feedback':
        if not block.has_value(cosmo, 'sigma_8') and block.has_value(cosmo, 'S_8_input'):
            check_range(block[cosmo, 'S_8_input'], 0.5, 1.0, "S_8_input")
            if config['NonLinear'] != 'NonLinear_none':
                check_range(block.get_double(names.halo_model_parameters, 'logT_AGN'), 7.3, 8.3, "logT_AGN")
            params_lin = {
                'S8': [block[cosmo, 'S_8_input']] * nz,
                'n_s': [block[cosmo, 'n_s']] * nz,
                'h': [block[cosmo, 'h0']] * nz,
                'obh2': [block[cosmo, 'ombh2']] * nz,
                'omch2': [block[cosmo, 'omch2']] * nz,
                'z': z
            }
            if config['NonLinear'] != 'NonLinear_none':
                params_nonlin = {
                    'S8': [block[cosmo, 'S_8_input']] * nz,
                    'n_s': [block[cosmo, 'n_s']] * nz,
                    'h': [block[cosmo, 'h0']] * nz,
                    'obh2': [block[cosmo, 'ombh2']] * nz,
                    'omch2': [block[cosmo, 'omch2']] * nz,
                    'z': z,
                    'log_T_AGN': [block.get_double(names.halo_model_parameters, 'logT_AGN')] * nz
                }

        elif block.has_value(cosmo, 'sigma_8') and not block.has_value(cosmo, 'S_8_input'):
            check_range(block[cosmo, 'sigma_8'], 0.39, 1.01, "sigma_8")
            if config['NonLinear'] != 'NonLinear_none':
                check_range(block.get_double(names.halo_model_parameters, 'logT_AGN'), 6.5, 9.36, "logT_AGN")
            params_lin = {
                'sigma8': [block[cosmo, 'sigma_8']] * nz,
                'n_s': [block[cosmo, 'n_s']] * nz,
                'h': [block[cosmo, 'h0']] * nz,
                'obh2': [block[cosmo, 'ombh2']] * nz,
                'omch2': [block[cosmo, 'omch2']] * nz,
                'z': z
            }
            if config['NonLinear'] != 'NonLinear_none':
                params_nonlin = {
                    'sigma8': [block[cosmo, 'sigma_8']] * nz,
                    'n_s': [block[cosmo, 'n_s']] * nz,
                    'h': [block[cosmo, 'h0']] * nz,
                    'obh2': [block[cosmo, 'ombh2']] * nz,
                    'omch2': [block[cosmo, 'omch2']] * nz,
                    'z': z,
                    'log_T_AGN': [block.get_double(names.halo_model_parameters, 'logT_AGN')] * nz
                }

    return params_lin, params_nonlin, params_boost


def get_cosmopower_inputs_cls(block,):
    
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


# The extract functions convert from the block to camb parameters
# during the execute function

def extract_recombination_params(block, config, more_config):
    default_recomb = camb.recombination.Recfast()
 
    min_a_evolve_Tm = block.get_double('recfast', 'min_a_evolve_Tm', default=default_recomb.min_a_evolve_Tm)
    RECFAST_fudge = block.get_double('recfast', 'RECFAST_fudge', default=default_recomb.RECFAST_fudge)
    RECFAST_fudge_He = block.get_double('recfast', 'RECFAST_fudge_He', default=default_recomb.RECFAST_fudge_He)
    RECFAST_Heswitch = block.get_int('recfast', 'RECFAST_Heswitch', default=default_recomb.RECFAST_Heswitch)
    RECFAST_Hswitch = block.get_bool('recfast', 'RECFAST_Hswitch', default=default_recomb.RECFAST_Hswitch)
    AGauss1 = block.get_double('recfast', 'AGauss1', default=default_recomb.AGauss1)
    AGauss2 = block.get_double('recfast', 'AGauss2', default=default_recomb.AGauss2)
    zGauss1 = block.get_double('recfast', 'zGauss1', default=default_recomb.zGauss1)
    zGauss2 = block.get_double('recfast', 'zGauss2', default=default_recomb.zGauss2)
    wGauss1 = block.get_double('recfast', 'wGauss1', default=default_recomb.wGauss1)
    wGauss2 = block.get_double('recfast', 'wGauss2', default=default_recomb.wGauss2)
    
    recomb = camb.recombination.Recfast(
        min_a_evolve_Tm = min_a_evolve_Tm, 
        RECFAST_fudge = RECFAST_fudge, 
        RECFAST_fudge_He = RECFAST_fudge_He, 
        RECFAST_Heswitch = RECFAST_Heswitch, 
        RECFAST_Hswitch = RECFAST_Hswitch, 
        AGauss1 = AGauss1, 
        AGauss2 = AGauss2, 
        zGauss1 = zGauss1, 
        zGauss2 = zGauss2, 
        wGauss1 = wGauss1, 
        wGauss2 = wGauss2, 
    )

    #Not yet supporting CosmoRec, but not too hard if needed.

    return recomb

def extract_reionization_params(block, config, more_config):
    reion = camb.reionization.TanhReionization()
    if more_config['do_reionization']:
        if more_config['use_optical_depth']:
            tau = block[cosmo, 'tau']
            reion = camb.reionization.TanhReionization(use_optical_depth=True, optical_depth=tau)
        else:
            sec = 'reionization'
            redshift = block[sec, 'redshift']
            delta_redshift = block[sec, 'delta_redshift']
            reion_params = get_optional_params(block, sec, ['fraction', 'helium_redshift', 'helium_delta_redshift', 'helium_redshiftstart'])
            reion = camb.reionization.TanhReionization(
                use_optical_depth=False,
                redshift = redshift,
                delta_redshift = delta_redshift,
                include_helium_fullreion = include_helium_fullreion,
                **reion_params,
                **more_config['reionization_params'],
            )
    else:
        reion = camb.reionization.TanhReionization()
        reion.Reionization = False
    return reion

def extract_dark_energy_params(block, config, more_config):
    if more_config['use_ppf_w']:
        de_class = camb.dark_energy.DarkEnergyPPF
    else:
        de_class = camb.dark_energy.DarkEnergyFluid

    dark_energy = de_class()
    if more_config['use_tabulated_w']:
        if block.has_value(cosmo, 'consistency_module_was_used') and block.has_value(cosmo, 'cosmomc_theta'):
            raise RuntimeError("You used the consistency module with cosmomc_theta=T but are also"
                               "using a tabulated w(a) in camb. The theta-H0 relation as implemeted"
                               "in the consistency module will not work for models other than w0-wa"
                               )
        a = block[names.de_equation_of_state, 'a']
        w = block[names.de_equation_of_state, 'w']
        dark_energy.set_w_a_table(a, w)
    else:
        w0 = block.get_double(cosmo, 'w', default=-1.0)
        wa = block.get_double(cosmo, 'wa', default=0.0)
        cs2 = block.get_double(cosmo, 'cs2_de', default=1.0)
        dark_energy.set_params(w=w0, wa=wa, cs2=cs2)

    return dark_energy

def extract_initial_power_params(block, config, more_config):
    optional_param_names = ['nrun', 'nrunrun', 'nt', 'ntrun', 'r']
    optional_params = get_optional_params(block, cosmo, optional_param_names)

    init_power = camb.InitialPowerLaw()
    init_power.set_params(
        ns = block[cosmo, 'n_s'],
        As = block[cosmo, 'A_s'],
        **optional_params,
        **more_config['initial_power_params']
    )
    return init_power

def extract_nonlinear_params(block, config, more_config):
    version = more_config['nonlinear_params'].get('halofit_version', '')

    if version == 'mead2015' or version == 'mead2016' or version == 'mead':
        A = block[names.halo_model_parameters, 'A']
        eta0 = block[names.halo_model_parameters, 'eta']
        hmcode_params = {'HMCode_A_baryon': A, 'HMCode_eta_baryon':eta0}
    elif version == 'mead2020_feedback':
        T_AGN = block[names.halo_model_parameters, 'logT_AGN']
        hmcode_params = {'HMCode_logT_AGN': T_AGN}
    else:
        hmcode_params = {}

    return camb.nonlinear.Halofit(
        **more_config['nonlinear_params'],
        **hmcode_params
    )


def extract_camb_params(block, config, more_config):
    want_perturbations = more_config['mode'] not in [MODE_BG, MODE_THERM]
    want_thermal = more_config['mode'] != MODE_BG

    if block.has_value(cosmo, 'sigma_8_input'):
        warnings.warn("Parameter sigma8_input will be deprecated in favour of sigma_8.")
        block[cosmo, 'sigma_8'] = block[cosmo, 'sigma_8_input']

    if block.has_value(cosmo, 'A_s') and block.has_value(cosmo, 'sigma_8'):
        warnings.warn("Parameter A_s is being ignored in favour of sigma_8")

    # Set A_s for now, this gets rescaled later if sigma_8 is provided.
    if not block.has_value(cosmo, 'A_s'):
        block[cosmo, 'A_s'] = DEFAULT_A_S
    
    # if want_perturbations:
    init_power = extract_initial_power_params(block, config, more_config)
    nonlinear = extract_nonlinear_params(block, config, more_config)
    # if want_thermal:
    recomb = extract_recombination_params(block, config, more_config)
    reion = extract_reionization_params(block, config, more_config)
    dark_energy = extract_dark_energy_params(block, config, more_config)

    # Get optional parameters from datablock.
    cosmology_params = get_optional_params(block, cosmo, 
        ['TCMB', 'YHe', 'mnu', 'nnu', 'standard_neutrino_neff', 'num_massive_neutrinos',
         ('A_lens', 'Alens')])

    if block.has_value(cosmo, 'massless_nu'):
        warnings.warn("Parameter massless_nu is being ignored. Set nnu, the effective number of relativistic species in the early Universe.")

    if (block.has_value(cosmo, 'omega_nu') or block.has_value(cosmo, 'omnuh2')) and not (block.has_value(cosmo, 'mnu')):
        warnings.warn("Parameter omega_nu and omnuh2 are being ignored. Set mnu and num_massive_neutrinos instead.")

    # Set h if provided, otherwise look for theta_mc
    if block.has_value(cosmo, 'cosmomc_theta'):
        cosmology_params['cosmomc_theta'] = block[cosmo, 'cosmomc_theta'] / 100
    elif block.has_value(cosmo, 'hubble'):
        cosmology_params['H0'] = block[cosmo, 'hubble']
    else:
        cosmology_params['H0'] = block[cosmo, 'h0']*100

    p = camb.CAMBparams(
        InitPower = init_power,
        Recomb = recomb,
        DarkEnergy = dark_energy,
        #Accuracy = accuracy,
        #Transfer = transfer,
        NonLinearModel=nonlinear,
        **config,
    )
    # Setting up neutrinos by hand is hard. We let CAMB deal with it instead.
    with be_quiet_camb():
        p.set_cosmology(ombh2 = block[cosmo, 'ombh2'],
                        omch2 = block[cosmo, 'omch2'],
                        omk = block[cosmo, 'omega_k'],
                        **more_config['cosmology_params'],
                        **cosmology_params)

    # Fix for CAMB version < 1.0.10
    if np.isclose(p.omnuh2, 0) and 'nnu' in cosmology_params and not np.isclose(cosmology_params['nnu'], p.num_nu_massless): 
        p.num_nu_massless = cosmology_params['nnu']

    # Setting reionization before setting the cosmology can give problems when
    # sampling in cosmomc_theta
    # if want_thermal:
    p.Reion = reion

    p.set_for_lmax(**more_config['lmax_params'])
    p.set_accuracy(**more_config['accuracy_params'])

    if want_perturbations:
        z = make_z_for_pk(more_config)
        p.set_matter_power(redshifts=z, nonlinear=config['NonLinear'] in ['NonLinear_both', 'NonLinear_pk'], **more_config['transfer_params'])

    return p


def save_derived_parameters(r, block):
    p = r.Params
    # Write the default derived parameters to distance section
    derived = r.get_derived_params()
    for k, v in derived.items():
        block[names.distances, k] = v
    block[names.distances, 'rs_zdrag'] = block[names.distances, 'rdrag']
    zstar = derived['zstar']
    shift = r.angular_diameter_distance(zstar) * (1 + zstar) * (p.omegam * p.H0**2)**0.5 / C_KMS
    block[names.distances, 'cmbshift'] = shift
    
    p.omegal = 1 - p.omegam - p.omk
    p.ommh2 = p.omegam * p.h**2

    for cosmosis_name, CAMB_name, scaling in [('h0'               , 'h',               1),
                                              ('hubble'           , 'h',             100),
                                              ('omnuh2'           , 'omnuh2',          1),
                                              ('n_eff'            , 'N_eff',           1),
                                              ('num_nu_massless'  , 'num_nu_massless', 1),
                                              ('num_nu_massive'   , 'num_nu_massive',  1),
                                              ('massive_nu'       , 'num_nu_massive',  1),
                                              ('massless_nu'      , 'num_nu_massless', 1),
                                              ('omega_b'          , 'omegab',          1),
                                              ('omega_c'          , 'omegac',          1),
                                              ('omega_nu'         , 'omeganu',         1),
                                              ('omega_m'          , 'omegam',          1),
                                              ('omega_lambda'     , 'omegal',          1),
                                              ('ommh2'            , 'ommh2',           1),]:

        CAMB_value = getattr(p, CAMB_name)*scaling

        if block.has_value(names.cosmological_parameters, cosmosis_name):
            input_value = block[names.cosmological_parameters, cosmosis_name]
            if not np.isclose(input_value, CAMB_value, rtol=0.002):
                warnings.warn(f"Parameter {cosmosis_name} inconsistent: input was {input_value} but value is now {CAMB_value}.")
        # Either way we save the results
        block[names.cosmological_parameters, cosmosis_name] = CAMB_value


def save_distances(r, block, more_config):
    p = r.Params

    # Evaluate z on a different grid than the spectra, so we can easily extend it further
    z_background = np.linspace(
        more_config['zmin_background'], more_config['zmax_background'], more_config['nz_background'])

    #If desired, append logarithmically distributed redshifts
    log_z = np.geomspace(more_config['zmax_background'], more_config['zmax_logz'], num = more_config['n_logz'])
    z_background = np.append(z_background, log_z[1:])
    
    # Write basic distances and related quantities to datablock
    block[names.distances, 'nz'] = len(z_background)
    block[names.distances, 'z'] = z_background
    block[names.distances, 'a'] = 1/(z_background+1)

    D_C = r.comoving_radial_distance(z_background)
    H = r.h_of_z(z_background)
    D_H = 1 / H[0]

    if p.omk == 0:
        D_M = D_C
    elif p.omk < 0:
        s = np.sqrt(-p.omk)
        D_M = (D_H / s)  * np.sin(s * D_C / D_H)
    else:
        s = np.sqrt(p.omk)
        D_M = (D_H / s) * np.sinh(s * D_C / D_H)

    D_L = D_M * (1 + z_background)
    D_A = D_M / (1 + z_background)
    D_V = ((1 + z_background)**2 * z_background * D_A**2 / H)**(1./3.)

    # Deal with mu(0), which is -np.inf
    mu = np.zeros_like(D_L)
    pos = D_L > 0
    mu[pos] = 5*np.log10(D_L[pos])+25
    mu[~pos] = -np.inf

    block[names.distances, 'D_C'] = D_C
    block[names.distances, 'D_M'] = D_M
    block[names.distances, 'D_L'] = D_L
    block[names.distances, 'D_A'] = D_A
    block[names.distances, 'D_V'] = D_V
    block[names.distances, 'H'] = H
    block[names.distances, 'MU'] = mu

    if more_config['do_bao']:
        rs_DV, _, _, F_AP = r.get_BAO(z_background, p).T
        block[names.distances, 'rs_DV'] = rs_DV
        block[names.distances, 'F_AP'] = F_AP

    if more_config['want_chistar']:
        chistar = (r.conformal_time(0)- r.tau_maxvis)
        block[names.distances, 'CHISTAR'] = chistar


def compute_growth_factor(block, P_tot, k, z, more_config):
    P_kmin = P_tot[:,0]
    D = np.sqrt(P_kmin / P_kmin[0]).squeeze()
    return D

def window(k_mode):
    R = 8 # in units Mpc/h which is correct since k is in h/Mpc
    return 3*(np.sin(k_mode*R)-k_mode*R*np.cos(k_mode*R))/(k_mode*R)**3


def save_matter_power(r, block, config, more_config):
    p = r.Params
    # Grids in k, z on which to save matter power.
    # There are two kmax values - the max one calculated directly,
    # and the max one extrapolated out too.  We output to the larger
    # of these
    kmax_power = max(more_config['kmax'], more_config['kmax_extrapolate'])
    z = make_z_for_pk(more_config)[::-1]
    nz = len(z)

    h0 = block[cosmo, 'h0']
    # Use k modes for cosmopower
    k = more_config['lin_matter_power_cp'].modes
    nk = len(k)

    params_lin, params_nonlin, params_boost = get_cosmopower_inputs(block, z, nz, config, more_config)

    if more_config['reference_linear_spectra'] is None:
        P_lin = more_config['lin_matter_power_cp'].ten_to_predictions_np(params_lin)
        if config['NonLinear'] != 'NonLinear_none':
            P_nl = (P_lin * more_config['nonlin_matter_power_cp'].ten_to_predictions_np(params_boost)
                    if params_boost is not None else
                    more_config['nonlin_matter_power_cp'].ten_to_predictions_np(params_nonlin))
    else:
        if more_config['As_emulator'] is not None:
            block[names.cosmological_parameters, 'A_s'] = more_config['As_emulator'].predictions_np(params_nonlin)[0][0]

        P_lin = more_config['lin_matter_power_cp'].predictions_np(params_lin)
        if config['NonLinear'] != 'NonLinear_none':
            P_nl = more_config['nonlin_matter_power_cp'].predictions_np(params_nonlin)

        # Subtract the reference spectra
        for i in range(P_lin.shape[0]):
            P_lin[i] += more_config['reference_linear_spectra']
            if config['NonLinear'] != 'NonLinear_none':
                P_nl[i] += more_config['reference_nonlinear_spectra']

        P_lin = 10**P_lin
        if config['NonLinear'] != 'NonLinear_none':
            P_nl = 10**P_nl

    k = k / h0
    P_lin = P_lin * h0**3.0
    if config['NonLinear'] != 'NonLinear_none':
        P_nl = P_nl * h0**3.0

    primordial_PK = p.scalar_power(k * h0)
    transfer = np.sqrt(P_lin[0, :] / (primordial_PK * k * 2.0 * np.pi**2.0))
    # Assumes we have P_lin at z=0!
    # matter_power = primordial_PK * transfer**2 * k**4 / (k**3 / (2 * np.pi**2))
    block.put_double_array_1d('matter_power_transfer_func', 'k_h', k)
    block.put_double_array_1d('matter_power_transfer_func', 't_k', transfer)

    if more_config['use_specific_k_modes']:
        k_new = np.logspace(np.log10(more_config['kmin']), np.log10(kmax_power), num=more_config['nk'])
        P_lin_new = np.zeros(shape=(nz, len(k_new)))
        if config['NonLinear'] != 'NonLinear_none':
            P_nl_new = np.zeros(shape=(nz, len(k_new)))

        for i in range(nz):
            P_lin_spline = InterpolatedUnivariateSpline(k, P_lin[i], ext=0)
            P_lin_new[i] = P_lin_spline(k_new)
            if config['NonLinear'] != 'NonLinear_none':
                P_nl_spline = InterpolatedUnivariateSpline(k, P_nl[i], ext=0)
                P_nl_new[i] = P_nl_spline(k_new)

        P_lin = P_lin_new
        if config['NonLinear'] != 'NonLinear_none':
            P_nl = P_nl_new
        k = k_new

    # Save matter power as a grid
    block.put_grid('matter_power_lin', 'z', z, 'k_h', k, 'p_k', P_lin)
    if config['NonLinear'] != 'NonLinear_none':
        block.put_grid('matter_power_nl', 'z', z, 'k_h', k, 'p_k', P_nl)

    # Get growth rates and sigma_8
    sigma_sq_cs = CubicSpline(k, k**2 * window(k)**2 * P_lin[0] / (2 * np.pi**2))
    sigma_8 = np.sqrt(sigma_sq_cs.integrate(k.min(), k.max()))

    rs_DV, H, DA, F_AP = r.get_BAO(z, p).T
    D = compute_growth_factor(block, P_lin, k, z, more_config)

    # Save growth rates and sigma_8
    block[names.growth_parameters, 'z'] = z
    block[names.growth_parameters, 'a'] = 1/(1+z)
    #block[names.growth_parameters, 'sigma_8'] = sigma_8
    #block[names.growth_parameters, 'fsigma_8'] = fsigma_8
    block[names.growth_parameters, 'rs_DV'] = rs_DV
    block[names.growth_parameters, 'H'] = H
    block[names.growth_parameters, 'DA'] = DA
    block[names.growth_parameters, 'F_AP'] = F_AP
    block[names.growth_parameters, 'd_z'] = D
    #block[names.growth_parameters, 'f_z'] = f

    block[names.cosmological_parameters, 'sigma_8'] = sigma_8
    block[names.cosmological_parameters, 'S_8'] = sigma_8 * np.sqrt(p.omegam / 0.3)

    #omega_m = (p.ombh2+p.omch2)/(p.H0/100)**2 # use this because p.omegam has mass of neutrinos in it.
    #block[cosmo, 'omega_m'] = omega_m

def save_cls(r, block):

    params = get_cosmopower_inputs_cls(block)
    cmb_unit = (2.7255e6)**2 #muK
    tt_spectra = more_config['TT'].ten_to_predictions_np(params) * cmb_unit
    te_spectra = more_config['TE'].predictions_np(params) * cmb_unit
    ee_spectra = more_config['EE'].ten_to_predictions_np(params) * cmb_unit
    pp_spectra = more_config['PP'].ten_to_predictions_np(params)
    ell = more_config['TT'].modes
    # Planck likelihood requires (ell*(ell+1))/(2*pi) C_ell
    block[names.cmb_cl, 'TT'] = tt_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block[names.cmb_cl, 'TE'] = te_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block[names.cmb_cl, 'EE'] = ee_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block[names.cmb_cl, 'PP'] = pp_spectra[0]*(ell*(ell+1))/(2*np.pi)
    block[names.cmb_cl, 'ell'] = ell


def execute(block, config):
    config, more_config = config
    p = "<Error occurred during parameter setup>"
    try:
        p = extract_camb_params(block, config, more_config)
        r = camb.get_background(p)

    except camb.CAMBError:
        if more_config['n_printed_errors'] <= more_config['max_printed_errors']:
            print("CAMB error caught: for these parameters")
            print(p)
            print(traceback.format_exc())
            if more_config['n_printed_errors'] == more_config['max_printed_errors']:
                print("\nFurther errors will not be printed.")
            more_config['n_printed_errors'] += 1
        return 1

    with be_quiet_camb():
        save_derived_parameters(r, block)
        save_distances(r, block, more_config)
    
    if p.WantTransfer:
        with be_quiet_camb():
            save_matter_power(r, block, config, more_config)

    if p.WantCls:
        with be_quiet_camb():
            save_cls(r, block)
    
    return 0

# Transfer – camb.model.TransferParams

# nu_mass_eigenstates – (integer) Number of non-degenerate mass eigenstates
# share_delta_neff – (boolean) Share the non-integer part of num_nu_massless between the eigenstates
# nu_mass_degeneracies – (float64 array) Degeneracy of each distinct eigenstate
# nu_mass_fractions – (float64 array) Mass fraction in each distinct eigenstate
# nu_mass_numbers – (integer array) Number of physical neutrinos per distinct eigenstate
# scalar_initial_condition – (integer/string, one of: initial_adiabatic, initial_iso_CDM, initial_iso_baryon, initial_iso_neutrino, initial_iso_neutrino_vel, initial_vector)

# MassiveNuMethod – (integer/string, one of: Nu_int, Nu_trunc, Nu_approx, Nu_best)
# DoLateRadTruncation – (boolean) If true, use smooth approx to radition perturbations after decoupling on small scales, saving evolution of irrelevant osciallatory multipole equations

# Evolve_baryon_cs – (boolean) Evolve a separate equation for the baryon sound speed rather than using background approximation
# Evolve_delta_xe – (boolean) Evolve ionization fraction perturbations
# Evolve_delta_Ts – (boolean) Evolve the splin temperature perturbation (for 21cm)

# Log_lvalues – (boolean) Use log spacing for sampling in L
# use_cl_spline_template – (boolean) When interpolating use a fiducial spectrum shape to define ratio to spline


def test(**kwargs):
    from cosmosis.datablock import DataBlock
    options = DataBlock.from_yaml('test_setup.yml')
    for k,v in kwargs.items():
        options[opt, k] = v
        print('set', k)
    config = setup(options)
    block = DataBlock.from_yaml('test_execute.yml')
    return execute(block, config)
    

if __name__ == '__main__':
    test()
