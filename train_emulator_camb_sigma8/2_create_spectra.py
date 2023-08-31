#!/usr/bin/env/python
import os
os.system('export OMP_NUM_THREADS=1')#need to run this line in terminal 

import numpy as np
import camb
import sys
from tqdm import tqdm
import os
import multiprocessing
import os.path
from pathlib import Path

print('Using CAMB %s'%(camb.__version__))


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



def spectra_generation_mead2020_feedback(params):

    if(Path('powerspectra/'+version+'_mead2020_feedback_spectra_'+str(int(params[0]))+'.npz').exists()):
        return 0

    sigma8_input = params[5]
    fid_As = 2e-9

    cp = camb.CAMBparams(WantTransfer=True, 
                    Want_CMB=False, Want_CMB_lensing=False, DoLensing=False, 
                    NonLinear="NonLinear_none",
                    WantTensors=False, WantVectors=False, WantCls=False, 
                    WantDerivedParameters=False,
                    want_zdrag=False, want_zstar=False)
    cp.set_accuracy(DoLateRadTruncation=True)
    cp.Transfer.high_precision = False
    cp.Transfer.accurate_massive_neutrino_transfers = False
    cp.Transfer.kmax = 100
    cp.Transfer.k_per_logint = 5
    cp.Transfer.PK_redshifts = np.array([0.0])

    cp.set_cosmology(H0=100.*params[3], ombh2=params[1], omch2=params[2], omk=0.0, mnu=0.06)
    cp.set_dark_energy(w=-1, wa=0)
    cp.set_initial_power(camb.initialpower.InitialPowerLaw(As=fid_As, ns=params[4]))

    cp.Reion = camb.reionization.TanhReionization()
    cp.Reion.Reionization = False

    r = camb.get_results(cp)
    fid_sigma8 = r.get_sigma8()[-1]
    As = fid_As*(sigma8_input/fid_sigma8)**2
    
    
    if(np.log(As*1e10)>5.0):
        print('As to great',params,As,np.log(As*10**10))
        return 0
    
    if(params[6]==0.0):
        redshifts = [0.0]
    else:
        redshifts = [0.0,params[6]*0.7,params[6]*0.8,params[6]*0.9,params[6],params[6]*1.1,params[6]*1.2,params[6]*1.3]
        redshifts.sort(reverse=True)

    cp = camb.set_params(ombh2 = params[1],
                         omch2 = params[2],
                         H0 = 100.*params[3],
                         ns = params[4],
                         As = As, 
                         omk=0.0,
                         lmax=5000,
                         WantTransfer=True,
                         kmax=100.0,
                         num_massive_neutrinos=1,
                         mnu=0.06,
                         nnu=2.0328,
                         halofit_version='mead2020_feedback',
                         tau=0.079,
                         TCMB=2.726, YHe=0.25,
                         HMCode_logT_AGN=params[7],
                         redshifts=redshifts,
                         verbose=False)
    cp.Reion = camb.reionization.TanhReionization()
    cp.Reion.Reionization = False

    cp.set_cosmology(H0=100.*params[3], ombh2=params[1], omch2=params[2], omk=0.0, mnu=0.06)

    results = camb.get_results(cp)
    PKcambnl = results.get_matter_power_interpolator(nonlinear=True,
                                                    hubble_units=False,
                                                    k_hunit=False)
    PKcambl = results.get_matter_power_interpolator(nonlinear=False,
                                                    hubble_units=False,
                                                    k_hunit=False)


    Pnonlin = PKcambnl.P(z=params[6], kh=k)
    Plin = PKcambl.P(z=params[6], kh=k)

    np.savez('powerspectra/'+version+'_mead2020_feedback_spectra_'+str(int(params[0])),Plin=Plin,Pnonlin=Pnonlin)

    print('done params: ',params,As)
    
    return 0


####   mead2020_feedback ####
params_all = np.load('outputs/train_mead2020_feedback_parameter.npz')
version='train'

for i in np.arange(0,45):
    spectra_i = np.arange(i*10000,(i+1)*10000).astype(np.int64)
    arg = zip(np.array([spectra_i,params_all['obh2'][spectra_i],params_all['omch2'][spectra_i],params_all['h'][spectra_i],params_all['n_s'][spectra_i],params_all['sigma8'][spectra_i],params_all['z'][spectra_i],params_all['log_T_AGN'][spectra_i]]).T)
    pool = multiprocessing.Pool(processes=100)
    pool.starmap(spectra_generation_mead2020_feedback, arg)
    pool.close()


P_lin = []
P_nonlin = []
available_indices = []
missing_indices = []
for i in tqdm(range(len(params_all['omch2']))):
    if(Path('powerspectra/train_mead2020_feedback_spectra_'+str(i)+'.npz').exists()):
        spectra=np.load('powerspectra/train_mead2020_feedback_spectra_'+str(i)+'.npz')
        P_lin.append(spectra['Plin'])
        P_nonlin.append(spectra['Pnonlin'])
        available_indices.append(i)
    else:
        missing_indices.append(i)



np.save('outputs/missing_indices_train',missing_indices)
np.savez('outputs/linear_matter_mead2020_feedback', modes = k, features = P_lin, available_indices = available_indices)
np.savez('outputs/non_linear_matter_mead2020_feedback', modes = k, features = P_nonlin, available_indices = available_indices)

## create reference power spectrum as the center of the parameter space
spectra_generation_mead2020_feedback(params=[-99,np.mean(params_all['obh2']),np.mean(params_all['omch2']),np.mean(params_all['h']),np.mean(params_all['n_s']),np.mean(params_all['sigma8']),np.mean(params_all['z']),np.mean(params_all['log_T_AGN'])])
spectra=np.load('powerspectra/train_mead2020_feedback_spectra_'+str(-99)+'.npz')
np.savez('outputs/center_linear_matter_mead2020_feedback', modes = k, features = spectra['Plin'])
np.savez('outputs/center_non_linear_matter_mead2020_feedback', modes = k, features = spectra['Pnonlin'])


params_all = np.load('outputs/test_mead2020_feedback_parameter.npz')
version='test'
for i in range(5):
    spectra_i = np.arange(i*10000,(i+1)*10000).astype(np.int64)
    arg = zip(np.array([spectra_i,params_all['obh2'][spectra_i],params_all['omch2'][spectra_i],params_all['h'][spectra_i],params_all['n_s'][spectra_i],params_all['sigma8'][spectra_i],params_all['z'][spectra_i],params_all['log_T_AGN'][spectra_i]]).T)
    pool = multiprocessing.Pool(processes=100)
    pool.starmap(spectra_generation_mead2020_feedback, arg)
    pool.close()


P_lin = []
P_nonlin = []
available_indices = []
missing_indices = []
for i in tqdm(range(len(params_all['omch2']))):
    if(Path('powerspectra/test_mead2020_feedback_spectra_'+str(i)+'.npz').exists()):
        spectra=np.load('powerspectra/test_mead2020_feedback_spectra_'+str(i)+'.npz')
        P_lin.append(spectra['Plin'])
        P_nonlin.append(spectra['Pnonlin'])
        available_indices.append(i)
    else:
        missing_indices.append(i)
    
np.savez('outputs/linear_matter_test_mead2020_feedback', modes = k, features = P_lin, available_indices = available_indices)
np.savez('outputs/non_linear_matter_test_mead2020_feedback', modes = k, features = P_nonlin, available_indices = available_indices)
np.save('outputs/missing_indices_test',missing_indices)




