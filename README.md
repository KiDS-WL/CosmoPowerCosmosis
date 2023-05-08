# Cosmopower-in-Cosmosis

We developed cosmosis modules called comsopower_interface found in cosmosis_modules that uses a pretrained COSMOPOWER emulator found in the folder trained_models and  to predict the linear and non-linear power spectrum.
Since the COSMOPOWER is only used to predict the power spectra, we additional need the camb_distance module, which is identical to the py_camb module without calculating the power spectra.

In order to to be able to use these modules, you need to install:
pip install cosmopower

For a detailed description of COSMOPOWER, see https://arxiv.org/pdf/2106.03846.pdf, where the pre-trained models can also be found here https://github.com/alessiospuriomancini/cosmopower.

In order to run cosmopower inside cosmosis we modified the KiDS-1000 cosebins pipeline.ini file, which is also provided here.

We also added new emulators, which are different from those provided in COSMOPOWER.
These are created in the following:
We used 450 000 spectra calculated with mead2020 feedback. 
50 000 for testing.
For the emulation, we used two methods. 
log10 P(k) - log10 P_reference(k)
For the this method the reference power spectrum which is also provided needs to be loaded in. The new emulator can be found in train_emulator_camb_S8/outputs with its corresponding interface in cosmosis_modules. 

If you want to train the emulator again using the power spectra generated from cosmosis we added in the folder train_model 4 python modules. Remember that you need to create the two subfolders outputs and plots. If you want to train the emulator on different camb spectra like mead2020_feedback, we added 4+1 python modules in train_model/camb_spectra. The structure of the modules are the following:

1_create_params:  creating the train and test parameters, for which the power spectra will be calculated. If you change them you also need to modify 2_create_spectra.

2_create_spectra: calculates the power spectra using the cosmosis modules found in cosmosis_modules_4_training, where you need to modify powerspectra.ini if you wanna use different power spectra estimators. Don’t forget to change the module name also in 2_create_spectra.py

2_1_calc_As.py: If you train your emulator on S8 it can happend that a combination of Omega_m and S8 yields As values for which the power spectrum cannout computed. To check if this happens during the MCMC we added an As emulator.

3_train_emulator: As the name says, it trains the emulator.

4_test_emulator: It tests the emulator and creates two plots found in plots, where one shows the accuracy of the linear and non-linear emulator.
