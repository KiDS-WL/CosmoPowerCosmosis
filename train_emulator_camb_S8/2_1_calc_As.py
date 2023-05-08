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

redshifts = np.linspace(0.0, 6.0, 100).tolist()
redshifts.sort(reverse=True)

As_min = 1e-10*np.exp(1.61)
As_max = 1e-10*np.exp(3.91)

def spectra_generation_mead2020_feedback(params):

     
    Omega_m=(params[1]+params[2])/params[3]**2
    sigma8_input = params[5]/(Omega_m/0.3)**0.5
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

    return As


####   mead2020_feedback ####
params_all = np.load('outputs/train_mead2020_feedback_parameter.npz')
version='train'

As_list = []
for i in tqdm(range(450)):
    spectra_i = np.arange(i*1000,(i+1)*1000).astype(np.int64)
    arg = zip(np.array([spectra_i,params_all['obh2'][spectra_i],params_all['omch2'][spectra_i],params_all['h'][spectra_i],params_all['n_s'][spectra_i],params_all['S8'][spectra_i],params_all['z'][spectra_i],params_all['log_T_AGN'][spectra_i]]).T)
    pool = multiprocessing.Pool(processes=100)
    results = pool.starmap(spectra_generation_mead2020_feedback, arg)
    pool.close()

    As_list.append(results)
np.save('outputs/As_values_train',np.array(As_list))

params_all = np.load('outputs/test_mead2020_feedback_parameter.npz')
version='test'

As_list = []
for i in tqdm(range(50)):
    spectra_i = np.arange(i*1000,(i+1)*1000).astype(np.int64)
    arg = zip(np.array([spectra_i,params_all['obh2'][spectra_i],params_all['omch2'][spectra_i],params_all['h'][spectra_i],params_all['n_s'][spectra_i],params_all['S8'][spectra_i],params_all['z'][spectra_i],params_all['log_T_AGN'][spectra_i]]).T)
    pool = multiprocessing.Pool(processes=100)
    results = pool.starmap(spectra_generation_mead2020_feedback, arg)
    pool.close()

    As_list.append(results)

np.save('outputs/As_values_test',np.array(As_list))





