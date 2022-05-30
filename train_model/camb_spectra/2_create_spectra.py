#!/usr/bin/env/python
import numpy as np
import camb
import sys
from tqdm import tqdm
import os
import multiprocessing
import os.path


os.system('export OMP_NUM_THREADS=1')

print('Using CAMB %s'%(camb.__version__))

num_k = 420

krange1 = np.logspace(np.log10(1e-5), np.log10(1e-4), num=20, endpoint=False)
krange2 = np.logspace(np.log10(1e-4), np.log10(1e-3), num=40, endpoint=False)
krange3 = np.logspace(np.log10(1e-3), np.log10(1e-2), num=60, endpoint=False)
krange4 = np.logspace(np.log10(1e-2), np.log10(1e-1), num=80, endpoint=False)
krange5 = np.logspace(np.log10(1e-1), np.log10(1), num=100, endpoint=False)
krange6 = np.logspace(np.log10(1), np.log10(10), num=120, endpoint=False)
krange7 = np.logspace(np.log10(10), np.log10(20), num=40, endpoint=False)

krange_new = np.concatenate((krange1,krange2))
krange_new = np.concatenate((krange_new, krange3))
krange_new = np.concatenate((krange_new, krange4))
krange_new = np.concatenate((krange_new, krange5))
krange_new = np.concatenate((krange_new, krange6))
k = np.concatenate((krange_new, krange7))

num_k = len(k)  # 560 k-modes
np.savetxt('outputs/k_modes.txt', k)

redshifts = np.linspace(0.0, 6.0, 100).tolist()
redshifts.sort(reverse=True)

def spectra_generation(params):
    # if(os.path.isfile('powerspectra/'+version+'_spectra_'+str(int(params[0]))+'.npz')):
    #     print('file exist already:',params)
    #     return 0

    print(params)
    cp = camb.set_params(ombh2 = params[1],
                         omch2 = params[2],
                         H0 = 100.*params[3],
                         ns = params[4],
                         As = 1e-10*np.exp(params[5]), 
                         omk=0.0, 
                         lmax=5000, 
                         WantTransfer=True,  
                         kmax=100.0,
                         mnu=0.06, standard_neutrino_neff=3.046,
                         halofit_version='mead2020_feedback',
                         tau=0.079,
                         TCMB=2.726, YHe=0.25,
                         HMCode_logT_AGN=params[7],
                         redshifts=redshifts,
                         verbose=False)

    try:
        results = camb.get_results(cp)
        PKcambnl = results.get_matter_power_interpolator(nonlinear=True,
                                                       hubble_units=False,
                                                       k_hunit=False)
        PKcambl = results.get_matter_power_interpolator(nonlinear=False,
                                                       hubble_units=False,
                                                       k_hunit=False)


        Pnonlin = PKcambnl.P(z=params[6], kh=k)
        Plin = PKcambl.P(z=params[6], kh=k)
        boost = Pnonlin/Plin

        np.savez('powerspectra/'+version+'_spectra_'+str(int(params[0])),Plin=Plin,boost=boost)

        return Plin,boost
       
    except:
        print('something wrong with CAMB')

    return


params_linear = np.load('outputs/train_mead2020_feedback_parameter_linear.npz')
params_boost = np.load('outputs/train_mead2020_feedback_parameter_boost.npz')
version='train'

for i in range(9):
    spectra_i = np.arange(i*20000,(i+1)*20000).astype(np.int64)
    arg = zip(np.array([spectra_i,params_linear['obh2'][spectra_i],params_linear['omch2'][spectra_i],params_linear['h'][spectra_i],params_linear['n_s'][spectra_i],params_linear['ln10^{10}A_s'][spectra_i],params_linear['z'][spectra_i],params_boost['log_T_AGN'][spectra_i]]).T)
    pool = multiprocessing.Pool(processes=200)
    pool.starmap(spectra_generation, arg)
    pool.close()


P_lin = []
nl_boost = []
for i in tqdm(range(len(params_linear['omch2']))):
    spectra=np.load('powerspectra/train_spectra_'+str(i)+'.npz')
    P_lin.append(spectra['Plin'])
    nl_boost.append(spectra['boost'])

np.savez('outputs/linear_matter_mead2020_feedback', modes = k, features = P_lin)
np.savez('outputs/non_linear_boost_mead2020_feedback', modes = k, features = nl_boost)

np.savez('outputs/log10_linear_matter_mead2020_feedback', modes = k, features = np.log10(P_lin))
np.savez('outputs/log10_non_linear_boost_mead2020_feedback', modes = k, features = np.log10(nl_boost))




params_linear = np.load('outputs/test_mead2020_feedback_parameter_linear.npz')
params_boost = np.load('outputs/test_mead2020_feedback_parameter_boost.npz')
version='test'

spectra_i = np.arange(0,20000).astype(np.int64)
arg = zip(np.array([spectra_i,params_linear['obh2'][spectra_i],params_linear['omch2'][spectra_i],params_linear['h'][spectra_i],params_linear['n_s'][spectra_i],params_linear['ln10^{10}A_s'][spectra_i],params_linear['z'][spectra_i],params_boost['log_T_AGN'][spectra_i]]).T)
pool = multiprocessing.Pool(processes=200)
pool.starmap(spectra_generation, arg)
pool.close()



P_lin = []
nl_boost = []
for i in tqdm(range(len(params_linear['omch2']))):
    spectra=np.load('powerspectra/test_spectra_'+str(i)+'.npz')
    P_lin.append(spectra['Plin'])
    nl_boost.append(spectra['boost'])

np.savez('outputs/linear_matter_test_mead2020_feedback', modes = k, features = P_lin)
np.savez('outputs/non_linear_boost_test_mead2020_feedback', modes = k, features = nl_boost)



