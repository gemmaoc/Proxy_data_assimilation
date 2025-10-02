
# Proxy_data_assimilation

Repo containing code for conducting proxy data assimilation using the ensemble Kalman filter. Adapted from code created by Greg Hakim's group at the Unviersity of Washington. Additional documentation on these methods from Greg's group can be found here: https://atmos.washington.edu/~hakim/LMR/docs/index.html. 

This repo contains the code needed to conduct these steps. The prior and proxy data should be contained in separate directories.
Recommended structure in parent directory:
  - subdirectory "data" containing "model", "instrumental", and "proxies" subdirectories.
  - subdirectory "ye_precalc_files"
  - subdirectory "data_assimilation_code" containing the code in this repo.
  - subdirectory "output"

Step 1: Create proxy database
-----------------------------
  1a. Create a new directory under the "proxies" subdirectory, named after the desired database name. Add NCDC-templated .txt files containing the proxy data.
  
  1b. Process proxy data by running "LMR_proxy_preprocess.py". Select the name of the proxy databaset in the user parameters. This generate a .pckl file of the new proxy db, in the "proxies" directory.
  
  1c. Build proxy system models for each proxy record. run "LMR_PSMbuild.py" and edit parameters as needed. Instrumental temperature and precip data for conducting the calibration needs to be provided.

Step 2: Generate prior
----------------------
  2a. Collect prior data, e.g. monthly output from a climate simulation or an ensemble of climate simulations. Want at least 1000 years total to draw from. 
  
  2b. Format prior data to input into PDA code. Prior needs to contain monthly gridded (any resolution) output of relevant variables: temperature and precip at a minimum.
      Generate one file for each variable, with dimenions time, lat, lon. If there are levels for certain variables, use the lowest level. If concatenating multiple ensemble
      members into one long prior, ensure the times are unique. They are just placeholders. I typically use nco tools to manipulate the files via the terminal.  
      
  2c. Add prior info to framework. Add the prior to and meta data to LMR_prior.py, datasets.yml, and config.yml.
  
  2d. Build prior (generate ye values, or the prior estimates of the proxy data). In the directory "misc/", run "build_ye_file.py" by typing into the terminal: "nohup python build_ye_file.py". This create .npz files containing the ye values, which will be saved in the directory "ye_precalc_files".

Step 3: Perform data assimilation
---------------------------------
  3a. Edit "config.yml" to reflect desired configuration (e.g., reconstruction years, prior, proxy database, PSM, grid, etc.
  
  3b. Run in the terminal with the command: "nohup python LMR_wrapper.py". 
      If you have multiple config files saved, run "nohup python LMR_wrapper.py config_x.yml".
      The printouts from running the assimilation will be saved to nohup.out.
      Reconstruction output will be saved under the directory "output", named after the reconstruction name specified in the config file. 
      
  3c. Format output. Reconstructions are output as .npz files. Run the file "LMR_convertNPZtoNETCDF.py" to convert to netCDF files. 
