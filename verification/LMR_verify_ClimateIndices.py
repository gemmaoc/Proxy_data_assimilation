"""

Module: LMR_verify_ClimateIndices.py

Purpose: Plots and generate verification statistics for various
         climate indices derived from one or more LMR reconstructions.
         Verification is performed against indices calculated from 
         various instrumental-era products. 
       
Notes:   The input files for LMR climate indices are netcdf files
         generated using the make_posterior_climate_indices.py
         module, located in the "diagnostics" sub-directory.
         This must be run before using the current module.
       

Originator: R. Tardif, U. of Washington, September 2019

Revisions: 

"""

import os, glob, sys
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from collections import OrderedDict

import matplotlib.pyplot as plt

# LMR specific imports
sys.path.append('../')
from LMR_utils import coefficient_efficiency

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
exps = OrderedDict([
#      label      
    ('LMR v2.1',  ('/home/disk/kalman3/rtardif/LMR/output',                             # 1: input directory
                   'productionFinal2_gisgpcc_ccms4_LMRdbv1.1.0_ClimateIndices')),       # 1: name of experiment
    ('LMR v2.0',  ('/home/disk/enkf3/rtardif/LMR/output/productionFinal_V1_Summer2018', # 2: input directory
                   'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0_ClimateIndices')),        # 2: name of experiment
    ])
fname = 'posterior_climate_indices_MCruns_ensemble_full.nc'

indices = ['nino34', 'pdo', 'amo', 'ao', 'nao', 'soi', 'sam']

#year_range = (0,2000)
year_range = (1800,2000)

do_verif = True
obsdir = '/home/disk/kalman3/rtardif/LMR/data/verification/ClimateIndices'
#                          data file name                                 label
obsfiles = {
    'nino34' : [
                ('ESRL_PSD_nino34.anom.monthly.data.1870-2018.txt',   'ESRL/PSD-HadISST'),
                ('KNMI_nino34_ERSSTv5.monthly.data.1854-2018.txt',    'KNMI-ERSSTv5'),
               ],
    'pdo'    : [
                ('ESRL_PSD_pdo.monthly.data.1900-2018.txt',           'UW Mantua'),
                ('NCEI_pdo.monthly.data.1854-2018.txt',               'NCEI-ERSSTv4'),
                ('KNMI_pdo_HadSST3.monthly.data.1884-2018.txt',       'KNMI-HadSST3'),
                ('KNMI_pdo_ERSSTv5.monthly.data.1880-2018.txt',       'KNMI-ERSSTv5'),
               ],
    'amo'    : [
                ('ESRL_PSD_amo.unsmoothed.monthly.data.1856-2018.txt','ESRL/PSD-KaplanV2'),
                ('KNMI_amo_HadSST4.monthly.data.1853-2018.txt',       'KNMI-HadSST4'),
                ('KNMI_amo_ERSSTv5.monthly.data.1854-2018.txt',       'KNMI-ERSSTv5'),
               ],
    'ao'     : [
                ('CSU_Thompson_ao_slp_monthly.data.1899-2018.txt',    'CSU-Thompson'),
                ('BNUIAP_ao_HadSLP2.annual.data.1850-2012.txt',       'BNU/IAP-HadSLP2'),
                ('BNUIAP_ao_NCAR.annual.data.1899-2012.txt',          'BNU/IAP-NCAR'),
               ],
    'nao'    : [
                ('CRU_nao_stations.annual.data.1825-2018.txt',        'CRU (annual)'),
                ('UW_JISAO_nao.annual.data.1864-2000.txt',            'UW/JISAO (annual)'),
               ],
    'soi'    : [('CRU_soi_stations.annual.data.1866-2018.txt',        'CRU (annual)')],
    'sam'    : [('BAS_sam.annual.data.1957-2018.txt',                 'BAS-Marshall (annual)')],
    }

# directory where the figures will be generated
figdir = './'
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

figysize = 4

index_names = {
    'nino34' : 'Nino3.4' ,
    'pdo'    : 'Pacific Decadal Oscillation',
    'amo'    : 'Atlantic Multidecadal Oscillation',
    'ao'     : 'Arctic Oscillation',
    'nao'    : 'North Atlantic Oscillation',
    'soi'    : 'Southern Oscillation',
    'sam'    : 'Southern Annular Mode',
    }

expnames = list(exps.keys())
nspec = len(expnames)
expnames_str = '_'.join([exp.replace(' ', '-') for exp in expnames])


# --- access the data ---
filehandles = []
for expname in expnames:
    datadir = exps[expname][0]
    nexp    = exps[expname][1]
    file = os.path.join(datadir, nexp, fname)
    filehandles.append(Dataset(file,'r'))

ix = 0
for fh in filehandles:
    time = fh['time']
    dates = num2date(time[:], units=time.units, calendar=time.calendar)
    yrs = np.array([date.year for date in dates])

    if ix == 0:
        nyrs, = yrs.shape
        years = np.zeros(shape=[nspec,nyrs])*np.nan
        idx_data = np.zeros(shape=[nspec,nyrs])*np.nan

    years[ix,:] = yrs

    # larger figsize if more frames
    figysize +=1 
    
    ix +=1

# loop over indices to plot
for index in indices:

    fig, axes = plt.subplots(nrows=nspec, ncols=1, figsize=[12,figysize], sharex=True)
    try:
        iter(axes)
    except:
        axes = [axes]
    
    ix = 0
    for axi in axes:
        fh = filehandles[ix]
        
        idx_data = fh[index]
        nyears,nMC,nens = idx_data.shape

        # "grand" ensemble and its statistics
        idx_gens    = np.reshape(idx_data, (nyears,nMC*nens))
        idx_ensmean = np.mean(idx_gens, axis=1)
        idx_low     = np.percentile(idx_gens,5, axis=1)
        idx_high    = np.percentile(idx_gens,95, axis=1)
        nrows=1

        axi.plot(years[ix,:], idx_ensmean, color = 'black', linestyle = '-', linewidth = 1)
        axi.fill_between(years[ix,:], idx_low, idx_high, color='black', alpha=0.25, linewidth=0.)
        axi.plot(year_range, [0,0], color = 'dimgray', linestyle = '--', linewidth = 0.8)

        
        if do_verif:
            # overlay instrumental-derived indices, observation-based dataset
            verif_colors = ['red', 'blue', 'cyan', 'lime']

            nbverif = len(obsfiles[index])
            for iv in range(nbverif):
                                
                PSDfile  = os.path.join(obsdir,obsfiles[index][iv][0])
                obslabel = obsfiles[index][iv][1]

                # upload the obs. data
                obs_year_lst = []
                obs_data_lst = []
                with open(PSDfile,'r') as fobs:
                    for line in fobs:
                        if not line.startswith('#'):
                            columns = line.split(' ')
                            columns = [item for item in columns if item != '' and item != '\n']

                            timetag = columns[0]
                            data_monthly = np.array([float(item) for item in columns[1:]])

                            data_monthly[data_monthly == -99.99] = np.nan
                            data_monthly[data_monthly == -999.9] = np.nan
                            # check on valid data
                            indsok = np.isnan(data_monthly)
                            nbtot = len(indsok)
                            nbmissing = np.sum(indsok)
                            if float(nbmissing)/float(nbtot) > 0.2:
                                data = np.nan
                            else:
                                data = np.nanmean(data_monthly)

                            obs_year_lst.append(timetag)
                            obs_data_lst.append(data)
                fobs.close()

                obs_years_all = np.array(obs_year_lst).astype('int')
                obs_data_annual = np.array(obs_data_lst).astype('float')
                obs_years = np.array(list(set(obs_years_all)))
                year_mask = (obs_years_all >= 1951) & (obs_years_all <= 1980)
                obs_data_annual = obs_data_annual - obs_data_annual [year_mask].mean()

                # verification statistics
                yrmask_lmr  = np.in1d(years[ix,:],obs_years)
                yrmask_obs  = np.in1d(obs_years, years[ix,:])
                df = pd.DataFrame({'LMR':idx_ensmean[yrmask_lmr], 'obs': obs_data_annual[yrmask_obs]})
                r_obs = df['LMR'].corr(df['obs'])            
                ce_obs = coefficient_efficiency(obs_data_annual[yrmask_obs],idx_ensmean[yrmask_lmr],valid=0.8)
                label = obslabel + ' (r=%5.2f, CE=%6.2f)' %(r_obs, ce_obs)
                axi.plot(obs_years, obs_data_annual, color = verif_colors[iv], linestyle = '-',linewidth=1.5,alpha=0.5,label=label)
                axi.legend(loc=2,ncol=3, fontsize=10,handlelength=1.,frameon=False)
            

            
        axi.set_ylabel(index.upper()+' index', fontweight='bold')
        axi.set_xlim(year_range)
        axi.minorticks_on()
        axi.xaxis.grid(which='both', linestyle=':', linewidth=0.85, color='lightgray', alpha=0.75)
        axi.xaxis.set_major_locator(plt.MaxNLocator(10))
        #axi.xaxis.set_minor_locator(plt.MaxNLocator(5))
        xaxis_label = ix

        # title(s)
        if ix == 0:            
            axi.set_title(index_names[index]+'\n'+expnames[ix], fontweight='bold',fontsize=12)            
        else:
            axi.set_title(expnames[ix], fontweight='bold',fontsize=12)
            
        ix +=1


    # adjust y-axis ranges to common values
    axi_min = []
    axi_max = []
    for axi in axes:
        rng = axi.get_ylim()
        axi_min.append(rng[0])
        axi_max.append(rng[1])
    min_all = np.min(axi_min)
    max_all = np.max(axi_max)
    for axi in axes:
        axi.set_ylim([min_all,max_all])

    # label x-axis
    axes[xaxis_label].set_xlabel('Year CE', fontweight='bold')
    
    # save the figure
    period_str = str(year_range[0])+'-'+str(year_range[1])
    outfname = '%s_%s_%s.png' %(index,expnames_str,period_str)
    plt.savefig(figdir+'/'+outfname, bbox_inches='tight')
    plt.close()
    

