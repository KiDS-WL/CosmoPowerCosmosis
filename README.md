# Cosmopower-in-Cosmosis

We developed cosmosis modules called comsopower_interface.py that uses a pretrained COSMOPOWER emulator found in the folder trained_models to predict the linear and non-linear power spectrum.
Since the COSMOPOWER is only used to predict the power spectra we additional need the camb_distance module, which is identical to the py_camb module without calcualting the power spectra. 

In order to to be able to use these modules you need to install:
pip install cosmopower

For a detailed description of COSMOPOWER see https://arxiv.org/pdf/2106.03846.pdf, where the pretrained models can also be found here https://github.com/alessiospuriomancini/cosmopower. 

In order to run cosmopower inside cosmosis we modified the KiDS-1000 cosebins pipeline.ini file, which is also provided here.

If you want to train the emulator again using the power spectra genreated from cosmosis we added in the folder train_model 4 python modules. Remember that you need to create the two subfolders putputs and plots. If you want to train the emulator on different camb spectra like mead2020_feedback than we added 4 python modules in train_model/camb_spectra. The structure of the modules are the following:

1_create_params:  creating the train and test paramters, for which the power spectra will be calculated. If you change them you also need to modify 2_create_spectra. 

2_create_spectra: calacualtes the power spectra using the cosmosis modules found in cosmosis_modules_4_training, where you need to modify powerspectra.ini if you wanna use different power spectra estimators. Don't forget to change the module name also in 2_create_spectra.py

3_train_emulator: As the name says it trains the emulator.

4_test_emulator: It tests the emulator and creates two plots found in plots, where one is showing the accuary of the linear emulator and the other the accuracy of the boost factor.




