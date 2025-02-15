import warnings
import camb
from cosmosis.datablock import names, option_section as opt
from cosmosis.datablock.cosmosis_py import errors
import numpy as np
from cosmosis.datablock import names as section_names
from scipy.interpolate import CubicSpline

cosmo = names.cosmological_parameters
lin_power_name = section_names.matter_power_lin

MODE_BG = "background"
MODE_THERM = "thermal"
MODE_CMB = "cmb"
MODE_TRANSFER = "transfer"
MODE_ALL = "all"
MODES = [MODE_BG, MODE_THERM, MODE_CMB, MODE_TRANSFER, MODE_ALL]


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

def get_choice(options, name, valid, default=None, prefix=''):
    choice = options.get_string(opt, name, default=default)
    if choice not in valid:
        raise ValueError("Parameter setting '{}' in camb must be one of: {}.  You tried: {}".format(name, valid, choice))
    return prefix + choice

def setup(options):
    M, m, v = camb.__version__.split(".")[:3]
    if not int(M) > 1 and not int(m) > 0 and not int(v) > 9:
        warnings.warn(f"CAMB version < 1.0.10 (found: {camb.__version__}). Massless neutrino handling not accounted for properly.")

    mode = options.get_string(opt, 'mode', default="all")
    if not mode in MODES:
        raise ValueError("Unknown mode {}.  Must be one of: {}".format(mode, MODES))

    config = {}
    config['WantCls'] = mode in [MODE_CMB, MODE_ALL]
    config['WantTransfer'] = mode in [MODE_TRANSFER, MODE_ALL]
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
    
    config['want_zstar'] = mode in [MODE_THERM, MODE_CMB, MODE_ALL]
    config['want_zdrag'] = config['want_zstar']

    # These are parameters that we do not pass directly to CAMBparams,
    # but use ourselves in some other way
    more_config = {}

    more_config["lmax_params"] = get_optional_params(options, opt, ["max_eta_k", "lens_potential_accuracy",
                                                                    "lens_margin", "k_eta_fac", "lens_k_eta_reference",
                                                                    #"min_l", "max_l_tensor", "Log_lvalues", , "max_eta_k_tensor"
                                                                     ])
    # lmax is required
    more_config["lmax_params"]["lmax"] = options.get_int(opt, "lmax", default=2600)                                                  
    
    more_config["initial_power_params"] = get_optional_params(options, opt, ["pivot_scalar", "pivot_tensor"])

    more_config["cosmology_params"] = get_optional_params(options, opt, ["neutrino_hierarchy" ,"theta_H0_range"])

    more_config['do_reionization'] = options.get_bool(opt, 'do_reionization', default=True)
    more_config['use_optical_depth'] = options.get_bool(opt, 'use_optical_depth', default=True)
    more_config["reionization_params"] = get_optional_params(options, opt, ["include_helium_fullreion", "tau_solve_accuracy_boost", 
                                                                            ("tau_timestep_boost", "timestep_boost"), 
                                                                            ("tau_max_redshift", "max_redshift")])
    
    more_config['use_tabulated_w'] = options.get_bool(opt, 'use_tabulated_w', default=False)
    more_config['use_ppf_w'] = options.get_bool(opt, 'use_ppf_w', default=False)
    
    more_config["nonlinear_params"] = get_optional_params(options, opt, ["halofit_version", "Min_kh_nonlinear"])

    more_config["accuracy_params"] = get_optional_params(options, opt, 
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
    more_config.update(get_optional_params(options, opt, ["zmid", "nz_mid"]))

    # Allow for finer redshift sampling at low redshifts
    if "zmid" in more_config:
        if not more_config["zmin"] < more_config["zmid"] or not more_config["zmid"] < more_config["zmax"]:
            raise ValueError("zmid needs to be larger than zmin and smaller than zmax!")

    # Allow usage of both background_* (for backwards compatability), as well *_background
    z_background = get_optional_params(options, opt, [("background_zmin", "zmin_background"),
                                                      ("zmin_background", "zmin_background"),
                                                      ("background_zmax", "zmax_background"),
                                                      ("zmax_background", "zmax_background"),
                                                      ("background_nz", "nz_background"),
                                                      ("nz_background", "nz_background"),])

    more_config['zmin_background'] = z_background.get('zmin_background', more_config['zmin'])
    more_config['zmax_background'] = z_background.get('zmax_background', more_config['zmax'])
    more_config['nz_background'] = z_background.get('nz_background', more_config['nz'])

    more_config["transfer_params"] = get_optional_params(options, opt, ["k_per_logint", "accurate_massive_neutrinos"])
    # Adjust CAMB defaults
    more_config["transfer_params"]["kmax"] = options.get_double(opt, "kmax", default=1.2)
    # more_config["transfer_params"]["high_precision"] = options.get_bool(opt, "high_precision", default=True)

    camb.set_feedback_level(level=options.get_int(opt, "feedback", default=0))
    return [config, more_config]

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
    if more_config["do_reionization"]:
        if more_config['use_optical_depth']:
            tau = block[cosmo, 'tau']
            reion = camb.reionization.TanhReionization(use_optical_depth=True, optical_depth=tau)
        else:
            sec = 'reionization'
            redshift = block[sec, 'redshift']
            delta_redshift = block[sec, 'delta_redshift']
            reion_params = get_optional_params(block, sec, ["fraction", "helium_redshift", "helium_delta_redshift", "helium_redshiftstart"])
            reion = camb.reionization.TanhReionization(
                use_optical_depth=False,
                redshift = redshift,
                delta_redshift = delta_redshift,
                include_helium_fullreion = include_helium_fullreion,
                **reion_params,
                **more_config["reionization_params"],
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
        a = block[name.de_equation_of_state, 'a']
        w = block[name.de_equation_of_state, 'w']
        dark_energy.set_w_a_table(a, w)
    else:
        w0 = block.get_double(cosmo, 'w', default=-1.0)
        wa = block.get_double(cosmo, 'wa', default=0.0)
        cs2 = block.get_double(cosmo, 'cs2_de', default=1.0)
        dark_energy.set_params(w=w0, wa=wa, cs2=cs2)

    return dark_energy

def extract_initial_power_params(block, config, more_config):
    optional_param_names = ["nrun", "nrunrun", "nt", "ntrun", "r"]
    optional_params = get_optional_params(block, cosmo, optional_param_names)

    init_power = camb.InitialPowerLaw()
    init_power.set_params(
        ns = block[cosmo, 'n_s'],
        As = block[cosmo, 'A_s'],
        **optional_params,
        **more_config["initial_power_params"]
    )
    return init_power

def extract_nonlinear_params(block, config, more_config):
    hmcode_params = get_optional_params(block, names.halo_model_parameters, 
                                        [("A", "HMCode_A_baryon"), 
                                         ("eta0", "HMCode_eta_baryon"),
                                         ("logT_AGN", "HMCode_logT_AGN")])
        
    return camb.nonlinear.Halofit(
        **more_config["nonlinear_params"],
        **hmcode_params
    )

def extract_accuracy_params(block, config, more_config):
    accuracy = camb.model.AccuracyParams(**more_config["accuracy_params"])
    return accuracy

def extract_transfer_params(block, config, more_config):
    PK_num_redshifts = more_config['nz']
    PK_redshifts = np.linspace(more_config['zmin'], more_config['zmax'], PK_num_redshifts)[::-1]
    transfer = camb.model.TransferParams(
        PK_num_redshifts=PK_num_redshifts,
        PK_redshifts=PK_redshifts,
        **more_config["transfer_params"]
    )
    return transfer


def extract_camb_params(block, config, more_config):
    init_power = extract_initial_power_params(block, config, more_config)
    recomb = extract_recombination_params(block, config, more_config)
    reion = extract_reionization_params(block, config, more_config)
    dark_energy = extract_dark_energy_params(block, config, more_config)
    nonlinear = extract_nonlinear_params(block, config, more_config)

    # Currently the camb.model.*Params classes default to 0 for attributes (https://github.com/cmbant/CAMB/issues/50),
    # so we're not using them.
    #accuracy = extract_accuracy_params(block, config, more_config)
    #transfer = extract_transfer_params(block, config, more_config)

    # Get optional parameters from datablock.
    cosmology_params = get_optional_params(block, cosmo, [("t_cmb" ,"TCMB"), 
                                                          ("yhe",   "YHe"),
                                                          ("omega_k", "omk"),
                                                          ("hubble", "H0"),
                                                          "mnu", 
                                                          ("n_eff", "nnu"),
                                                          "standard_neutrino_neff",
                                                          ("massive_nu", "num_massive_neutrinos"),
                                                          "num_massive_neutrinos",
                                                          ("a_lens", "Alens")])

    if block.has_value(cosmo, "massless_nu"):
        warnings.warn("Parameter massless_nu is being ignored. Set n_eff instead of the effective number of relativistic species in the early Universe.")
    if block.has_value(cosmo, "omega_nu") or block.has_value(cosmo, "omnuh2"):
        warnings.warn("Parameter omega_nu and omnuh2 are being ignored. Set mnu and num_massive_neutrinos instead.")

    # Set h if provided, otherwise look for theta_mc
    if not "H0" in cosmology_params:
        if block.has_value(cosmo, "h0"):
            cosmology_params["H0"] = block[cosmo, "h0"]*100
        else:
            cosmology_params["cosmomc_theta"] = block[cosmo, "cosmomc_theta"]/100
    
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
    p.set_cosmology(ombh2 = block[cosmo, 'ombh2'],
                    omch2 = block[cosmo, 'omch2'],
                    **more_config["cosmology_params"],
                    **cosmology_params)

    # Fix for CAMB version < 1.0.10
    if np.isclose(p.omnuh2, 0) and "nnu" in cosmology_params and not np.isclose(cosmology_params["nnu"], p.num_nu_massless): 
        p.num_nu_massless = cosmology_params["nnu"]


    # Setting reionization before setting the cosmology can give problems when
    # sampling in cosmomc_theta
    p.Reion = reion

    p.set_for_lmax(**more_config["lmax_params"])
    p.set_accuracy(**more_config["accuracy_params"])

    if "zmid" in more_config:
        z = np.concatenate((np.linspace(more_config['zmin'], 
                                        more_config['zmid'], 
                                        more_config['nz_mid'], 
                                        endpoint=False),
                            np.linspace(more_config['zmid'], 
                                        more_config['zmax'], 
                                        more_config['nz']-more_config['nz_mid'])))[::-1]
    else:
        z = np.linspace(more_config['zmin'], more_config['zmax'], more_config["nz"])[::-1]
    p.set_matter_power(redshifts=z, nonlinear=config["NonLinear"] in ["NonLinear_both", "NonLinear_pk"], **more_config["transfer_params"])

    return p

def window(k_mode):
    R = 8 # in units Mpc/h which is correct since k is in h/Mpc
    return 3*(np.sin(k_mode*R)-k_mode*R*np.cos(k_mode*R))/(k_mode*R)**3

def execute(block, config):
    config, more_config = config
    try:
        p = extract_camb_params(block, config, more_config)
    except camb.baseconfig.CAMBParamRangeError as e:
        print("CAMBParamRangeError:", e)
        return 1
    
    
    r = camb.get_background(p)
    
    # Write derived parameters to cosmological_parameters section
    derived = r.get_derived_params()
    for k, v in derived.items():
        block[names.cosmological_parameters, k] = v
    
    # Calculate Omega_Lambda. Doesn't include radiation!
    p.omegal = 1 - p.omegam - p.omk
    p.ommh2 = p.omegam * p.h**2


    for cosmosis_name, CAMB_name, scaling in [("h0"               , "h",               1),
                                              ("hubble"           , "h",             100),
                                              ("omnuh2"           , "omnuh2",          1),
                                              ("n_eff"            , "N_eff",           1),
                                              ("num_nu_massless"  , "num_nu_massless", 1),
                                              ("num_nu_massive"   , "num_nu_massive",  1),
                                              ("massive_nu"       , "num_nu_massive",  1),
                                              ("massless_nu"      , "num_nu_massless", 1),
                                              ("omega_b"          , "omegab",          1),
                                              ("omega_c"          , "omegac",          1),
                                              ("omega_nu"         , "omeganu",         1),
                                              ("omega_m"          , "omegam",          1),
                                              ("omega_lambda"     , "omegal",          1),
                                              ("ommh2"            , "ommh2",           1),]:
        CAMB_value = getattr(p, CAMB_name)*scaling
        if block.has_value(names.cosmological_parameters, cosmosis_name):
            input_value = block[names.cosmological_parameters, cosmosis_name]
            if not np.isclose(input_value, CAMB_value):
                warnings.warn(f"Parameter {cosmosis_name} inconsistent: input was {input_value} but value is now {CAMB_value}.")
                #raise
        else:
            block[names.cosmological_parameters, cosmosis_name] = CAMB_value

    if not block.has_value(names.cosmological_parameters, "cosmomc_theta"):
        block[names.cosmological_parameters, "cosmomc_theta"] = r.cosmomc_theta()*100

    z_background = np.linspace(more_config["zmin_background"], more_config["zmax_background"], more_config["nz_background"])

    # Get distances and write to datablock
    block[names.distances, "z"] = z_background
    block[names.distances, "a"] = 1/(z_background+1)
    block[names.distances, "D_A"] = r.angular_diameter_distance(z_background)
    block[names.distances, "D_C"] = r.comoving_radial_distance(z_background)
    block[names.distances, "D_M"] = r.comoving_radial_distance(z_background)
    d_L = r.luminosity_distance(z_background)
    block[names.distances, "D_L"] = d_L
    block[names.distances, "MU"] = np.insert(5*np.log10(d_L[1:])+25, 0, np.nan)

    block[names.distances, "H"] = r.h_of_z(z_background)

    rs_DV, H, DA, F_AP = r.get_BAO(z_background, p).T
    block[names.distances, "rs_DV"] = rs_DV
    block[names.distances, "F_AP"] = F_AP


    # Get growth rates and sigma_8
    
    P_k_lin = block[lin_power_name, 'P_k']
    k = block[lin_power_name, 'k_h']
    sigma_sq_cs = CubicSpline(k,k**2*window(k)**2*P_k_lin[0]/(2*np.pi**2))
    sigma8 = np.sqrt(sigma_sq_cs.integrate(k.min(),k.max()))

    
    block[names.cosmological_parameters, "sigma_8"] = sigma8
    block[names.cosmological_parameters, "S_8"] = sigma8*np.sqrt(p.omegam/0.3)

    omega_m = (p.ombh2+p.omch2)/(p.H0/100)**2 # use this because p.omegam has mass of neutrinos in it. 
    block[cosmo, 'omega_m'] = omega_m
    
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
        print("set", k)
    config = setup(options)
    block = DataBlock.from_yaml('test_execute.yml')
    return execute(block, config)
    

if __name__ == '__main__':
    test()
