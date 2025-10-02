"""
Module: LMR_psms.py

Purpose: Module containing methods for various Proxy System Models (PSMs)
         Adapted from LMR_proxy and LMR_calibrate using OOP by Andre Perkins

Originator: Andre Perkins, U. of Washington

Revisions: 
          - Use of new more efficient "get_distance" function to calculate the
            distance between proxy sites and analysis grid points.
            [R. Tardif, U. of Washington, February 2016]
          - Added the "LinearPSM_TorP" class, allowing the use of
            temperature-calibrated *OR* precipitation-calibrated linear PSMs.
            For each proxy record to be assimilated, the selection of the PSM
            (i.e. T vs P) is perfomed on the basis of the smallest MSE of
            regression residuals.
            [R. Tardif, U. of Washington, April 2016]
          - Added the "h_interpPSM" psm class for use of isotope-enabled GCM 
            data as prior: Ye values are taken as the prior isotope field either 
            at the nearest grid pt. or as the weighted-average of values at grid 
            points surrounding the assimilated isotope proxy site.
            [ R. Tardif, U. of Washington, June 2016 ]
          - Added the "BilinearPSM" class for PSMs based on bivariate linear 
            regressions w/ temperature AND precipitation/PSDI as independent 
            variables.
            [ R. Tardif, Univ. of Washington, June 2016 ]
          - Added the capability of calibrating/using PSMs calibrated on the basis 
            of a proxy record seasonality metadata.
            [ R. Tardif, Univ. of Washington, July 2016 ]
          - Added the capability of objectively calibrating/using PSMs calibrated 
            on the basis objectively-derived seasonality. 
            [ R. Tardif, Univ. of Washington, December 2016 ]
          - Added the "BayesRegUK37PSM" class, the forward model used in 
            the assimilation of alkenone uk37 proxy data. Code based on 
            spline coefficients provided by J. Tierney (U of Arizona).
            [ R. Tardif, Univ. of Washington, January 2017 ]
"""
import logging
import numpy as np
import logging
import os.path

import LMR_calibrate
from LMR_utils import (haversine, get_distance, smooth2D,
                       get_data_closest_gridpt, class_docs_fixer)

import pandas as pd
from scipy.stats import linregress
import statsmodels.formula.api as sm

from abc import ABCMeta, abstractmethod
from load_data import load_cpickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import bayspline
except ImportError as e1:
    print('Warning:', e1)
try:
    import bayspar
except ImportError as e2:
    print('Warning:', e2)
try:
    import deltaoxfox as dfox
except ImportError as e3:
    print('Warning:', e3)
try:
    import baymag
except ImportError as e4:
    print('Warning:', e4)


# Logging output utility, configuration controlled by driver
_log = logging.getLogger(__name__)


class BasePSM(metaclass=ABCMeta):
    """
    Proxy system model.

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
    psm_kwargs: dict (unpacked)
        Specfic arguments for the target PSM
    """

    def __init__(self, config, proxy_obj, **psm_kwargs):
        pass

    @abstractmethod
    def psm(self, prior_obj):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        prior_obj BasePriorObject
            Prior to be mapped to observation space (Ye).

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """
        pass

    @abstractmethod
    def error(self):
        """
        Error model for given PSM.
        """
        pass

    @classmethod
    def get_kwargs(cls, config):
        """
        Returns keyword arguments required for instantiation of given PSM.

        Parameters
        ----------
        config: LMR_config
            Configuration module used for current LMR run.

        Returns
        -------
        kwargs: dict
            Keyword arguments for given PSM
        """
        try:
            psm_data = cls._load_psm_data(config)
            return {'psm_data': psm_data}
        except IOError as e:
            print(e)
            return {}


@class_docs_fixer
class LinearPSM(BasePSM):
    """
    PSM based on linear regression.

    Attributes
    ----------
    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy sitex
    corr: float
        Correlation of proxy record against calibration data
    slope: float
        Linear regression slope of proxy/calibration fit
    intercept: float
        Linear regression y-intercept of proxy/calibration fit
    R: float
        Mean-squared error of proxy/calibration fit

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
    psm_data: dict
        Pre-calibrated PSM dictionary containing current associated
        proxy site's calibration information

    Raises
    ------
    ValueError
        If PSM is below critical correlation threshold.
    """

    def __init__(self, config, proxy_obj, psm_data=None, calib_obj=None):

        self.psm_key = 'linear'
        
        proxy = proxy_obj.type
        site = proxy_obj.id
        r_crit = config.psm.linear.psm_r_crit
        self.lat  = proxy_obj.lat
        self.lon  = proxy_obj.lon
        self.elev = proxy_obj.elev

        self.datatag_calib = config.psm.linear.datatag_calib
        self.avgPeriod = config.psm.linear.avgPeriod

        
        # Very crude assignment of sensitivity
        # TODO: more inclusive & flexible way of doing this
        if self.datatag_calib in ['GPCC', 'DaiPDSI']:
            self.sensitivity = 'moisture'
        else:
            self.sensitivity = 'temperature'
            
        try:
            # Try using pre-calibrated psm_data
            if psm_data is None:
                psm_data = self._load_psm_data(config)
                #pre_calib_file = config.psm.linear.pre_calib_datafile
                #return load_cpickle(pre_calib_file)

            psm_site_data = psm_data[(proxy, site)]
            self.corr = psm_site_data['PSMcorrel']
            self.slope = psm_site_data['PSMslope']
            self.intercept = psm_site_data['PSMintercept']
            self.R = psm_site_data['PSMmse']

            # check if seasonality defined in the psm data
            # if it is, return as an attribute
            if 'Seasonality' in list(psm_site_data.keys()):
                self.seasonality = psm_site_data['Seasonality']
            
        except KeyError as e:
            raise ValueError('Could not find proxy in pre-calibration file... '
                             'Skipping: {}'.format(proxy_obj.id))

        
        except IOError as e:
            # No precalibration found, have to do it for this proxy
            #print(('No pre-calibration file found for'
            #       ' {} ({}) ... calibrating ...'.format(proxy_obj.id,
            #                                           proxy_obj.type)))
            _log.info(('No pre-calibration file found for'
                   ' {} ({}) ... calibrating ...'.format(proxy_obj.id,
                                                         proxy_obj.type)))
            
            # check if calibration object already exists
            if calib_obj is None:
                print ('calibration object does not exist ...creating it...')
                # TODO: Fix call and Calib Module                
                datag_calib = config.psm.linear.datatag_calib
                calib_obj= LMR_calibrate.calibration_assignment(datag_calib)
                calib_obj.datadir_calib = config.psm.linear.datadir_calib
                calib_obj.read_calibration()

            self.calibrate(calib_obj, proxy_obj)
    
        ###Gemma's Code
        except OSError as e:
            # No precalibration found, have to do it for this proxy
            print(('No pre-calibration file found for'
                   ' {} ({}) ... calibrating ...'.format(proxy_obj.id,
                                                         proxy_obj.type)))
            _log.info(('No pre-calibration file found for'
                   ' {} ({}) ... calibrating ...'.format(proxy_obj.id,
                                                         proxy_obj.type)))

            # check if calibration object already exists
            if calib_obj is None:
                #print 'calibration object does not exist ...creating it...'
                # TODO: Fix call and Calib Module                
                datag_calib = config.psm.linear.datatag_calib
                calib_obj= LMR_calibrate.calibration_assignment(datag_calib)
                calib_obj.datadir_calib = config.psm.linear.datadir_calib
                calib_obj.read_calibration()

            self.calibrate(calib_obj, proxy_obj)
                
         ###End Gemma's code
         
        # Raise exception if critical correlation value not met
        if abs(self.corr) < r_crit:
            raise ValueError(('Proxy model correlation ({:.2f}) does not meet '
                              'critical threshold ({:.2f}).'
                              ).format(self.corr, r_crit))


        
    # TODO: Ideally prior state info and coordinates should all be in single obj
    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # Associate state variable with PSM calibration dataset
        # TODO: possible calibration sources hard coded for now... should define associations at config level
        if self.datatag_calib in ['GISTEMP','GISTEMP_NicBrom', 'MLOST', 'HadCRUT', 'BerkeleyEarth']:
            # temperature
            state_var = 'tas_sfc_Amon'
        elif self.datatag_calib in ['GPCC','DaiPDSI']:
            # moisture 
            if self.datatag_calib == 'GPCC':
                state_var = 'pr_sfc_Amon'
            elif self.datatag_calib == 'DaiPDSI':
                state_var = 'scpdsi_sfc_Amon'
            else:
                raise KeyError('Unrecognized moisture calibration source.'
                               ' State variable not identified for Ye calculation.')
        else:
            raise KeyError('Unrecognized calibration-state variable association.'
                           ' State variable not identified for Ye calculation.')

        if state_var not in list(X_state_info.keys()):
            raise KeyError('Needed variable not in state vector for Ye'
                           ' calculation.')

        # TODO: end index should already be +1, more pythonic
        tas_startidx, tas_endidx = X_state_info[state_var]['pos']
        ind_lon = X_state_info[state_var]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var]['spacecoords'].index('lat')
        X_lons = X_coords[tas_startidx:(tas_endidx+1), ind_lon]
        X_lats = X_coords[tas_startidx:(tas_endidx+1), ind_lat]
        tas_data = Xb[tas_startidx:(tas_endidx+1), :]

        gridpoint_data = self.get_close_grid_point_data(tas_data.T,
                                                        X_lons,
                                                        X_lats)
        Ye = self.basic_psm(gridpoint_data)

        return Ye

    def basic_psm(self, data):
        """
        A PSM that doesn't need to do the state unpacking steps...

        Parameters
        ----------
        data: ndarray
            Data to be used in the psm calculation of estimated observations
            (Ye)

        Returns
        -------
        Ye: ndarray
            Estimated observations from the proxy system model
        """

        return self.slope * data + self.intercept

    def get_close_grid_point_data(self, data, lon, lat):
        """
        Extracts data along the sampling dimension that is closest to the
        grid point (lat/lon) of the current PSM object.

        Parameters
        ----------
        data: ndarray
            Gridded data matching dimensions of (sample, lat, lon) or
            (sample, lat*lon).
        lon: ndarray
            Longitudes pertaining to the input data.  Can have shape as a single
            vector (lat), a grid (lat, lon), or a flattened grid (lat*lon).
        lat: ndarray
            Latitudes pertaining to the input data. Can have shape as a single
            vector (lon), a grid (lat, lon), or a flattened grid (lat*lon).

        Returns
        -------
        tmp_dat: ndarray
            Grid point data closes to the lat/lon of the current PSM object
        """

        lonshp = lon.shape
        latshp = lat.shape

        # If not equal we got a single vector?
        if lonshp != latshp:
            lon, lat = np.meshgrid(lon, lat)

        if len(lon.shape) > 1:
            lon = lon.ravel()
            lat = lat.ravel()

        # Calculate distance
        dist = haversine(self.lon, self.lat, lon, lat)

        if len(dist) in data.shape:
            loc_idx = dist.argmin()
            tmp_dat = data[..., loc_idx]
        else:
            # TODO: This is not general lat/lon being swapped, OK for now...
            min_dist_lat_idx, \
            min_dist_lon_idx = np.unravel_index(dist.argmin(), data.shape[-2:])
            tmp_dat = data[..., min_dist_lat_idx, min_dist_lon_idx]
            
        return tmp_dat

    # Define the error model for this proxy
    @staticmethod
    def error():
        return 0.1

    # TODO: Simplify a lot of the actions in the calibration
    def calibrate(self, C, proxy, diag_output=False, diag_output_figs=False):
        """
        Calibrate given proxy record against observation data and set relevant
        PSM attributes.

        Parameters
        ----------
        C: calibration_master like
            Calibration object containing data, time, lat, lon info
        proxy: BaseProxyObject like
            Proxy object to fit to the calibration data
        diag_output, diag_output_figs: bool, optional
            Diagnostic output flags for calibration method
        """

        calib_spatial_avg = False
        Npts = 9  # nb of neighboring pts used in smoothing

        #print 'Calibrating: ', '{:25}'.format(proxy.id), '{:35}'.format(proxy.type)
        
        # --------------------------------------------
        # Use linear model (regression) as default PSM
        # --------------------------------------------
        
        nbmaxnan = 0
        
        # Look for indices of calibration grid point closest in space (in 2D)
        # to proxy site
        dist = get_distance(proxy.lon, proxy.lat, C.lon, C.lat)
        # indices of nearest grid pt.
        jind, kind = np.unravel_index(dist.argmin(), dist.shape)

        if calib_spatial_avg:
            C2Dsmooth = np.zeros(
                [C.time.shape[0], C.lat.shape[0], C.lon.shape[0]])
            for m in range(C.time.shape[0]):
                C2Dsmooth[m, :, :] = smooth2D(C.temp_anomaly[m, :, :], n=Npts)
            calvals = C2Dsmooth[:, jind, kind]
        else:
            calvals = C.temp_anomaly[:, jind, kind]


        # -------------------------------------------------------
        # Calculate averages of calibration data over appropriate
        # intervals (annual or according to proxy seasonality)
        # -------------------------------------------------------
        if self.avgPeriod == 'annual':
            # Simply use annual averages
            avgMonths = [1,2,3,4,5,6,7,8,9,10,11,12]
        elif 'season' in self.avgPeriod:
            # Consider the seasonality of the proxy record
            avgMonths =  proxy.seasonality
            #print('avgMonths:',avgMonths)#test GKO. remove!!
        else:
            exit('ERROR: Unrecognized value for avgPeriod! Exiting!')
        
        nbmonths = len(avgMonths)
        cyears = list(set([C.time[k].year for k in range(len(C.time))])) # 'set' is used to get unique values in list
        nbcyears = len(cyears)
        reg_x = np.zeros(shape=[nbcyears])
        reg_x[:] = np.nan # initialize with nan's

        for i in range(nbcyears):
            #print('i=',i,'in nbcyears=',nbcyears) #GKO test
            # monthly data from current year
            indsyr = [j for j,v in enumerate(C.time) if v.year == cyears[i] and v.month in avgMonths]
            # check if data from previous year is to be included
            indsyrm1 = []
            if any(m < 0 for m in avgMonths):
                year_before = [abs(m) for m in avgMonths if m < 0]
                indsyrm1 = [j for j,v in enumerate(C.time) if v.year == cyears[i]-1. and v.month in year_before]
            # check if data from following year is to be included
            indsyrp1 = []
            if any(m > 12 for m in avgMonths):
                year_follow = [m-12 for m in avgMonths if m > 12]
                indsyrp1 = [j for j,v in enumerate(C.time) if v.year == cyears[i]+1. and v.month in year_follow]

            inds = indsyrm1 + indsyr + indsyrp1
            #print('inds',inds,'indsyrm1',indsyrm1,'indsyr',indsyr,'indsyrp1',indsyrp1) #GKO test
            if len(inds) == nbmonths: # all months are in the data
                tmp = np.nanmean(calvals[inds],axis=0)
                #print('calvals[inds]:',calvals[inds])
                nancount = np.isnan(calvals[inds]).sum(axis=0)
                if nancount > nbmaxnan: 
                    tmp = np.nan
                    #print('nan count:',nancount,'nbmaxnan',nbmaxnan) #GKO Test
            else:
                tmp = np.nan
                #print('nan for this i') #GKO test
            reg_x[i] = tmp
            
            
        # making sure calibration anomalies are unbiased (avg = 0) (NEW: RT 12/22/16)
        #reg_x = reg_x - np.nanmean(reg_x)

        
        # -------------------------
        # Perform linear regression
        # -------------------------
        # Use pandas DataFrame to store proxy & calibration data side-by-side
        header = ['variable', 'y']
        # Fill-in proxy data
        df = pd.DataFrame({'time':proxy.time, 'y': proxy.values})
        df.columns = header
        # Add calibration data
        frame = pd.DataFrame({'variable':cyears, 'Calibration':reg_x})
        #print('reg_x for this proxy:',reg_x) #test GKO. remove!
        df = df.merge(frame, how='outer', on='variable')

        col0 = df.columns[0]
        df.set_index(col0, drop=True, inplace=True)
        df.index.name = 'time'
        df.sort_index(inplace=True)

        # ensure all df entries are floats: if not, sm.ols gives garbage
        df = df.astype(np.float)
        #print(df) #GKO test. remove. 
        
        # Perform the regression
        nobs = None
        try:
            #print('trying linear regression with sm.ols') #GKO test. remove. 
            regress = sm.ols(formula="y ~ Calibration", data=df).fit()
            #print(regress) #test GKO. remove. 
            # number of data points used in the regression
            nobs = int(regress.nobs)
        except ValueError: # no overlapping data
            nobs = 0
            #print('except value error passed') #GKO test. remove. 
        except Exception as exception:
            print('Exception raised', exception)
#<<<<<<< HEAD
#        if nobs is None:
#            raise ValueError()
#        elif nobs is not None and nobs < 10:  # skip rest if insufficient overlapping data
#=======

        if nobs is not None and nobs < 25:  # skip rest if insufficient overlapping data
#>>>>>>> ea212b8664b415cb895130adf2c57704487461f0
            print('Insufficient obs./calib. overlap'
                  ' to calibrate psm: nobs=%d' %nobs)
            raise ValueError()

        
        # START NEW (GH) 21 June 2015... RT edit June 2016
        # detrend both the proxy and the calibration data
        #
        # RT: This code is old and not fully compatible with more recent modifications
        # (use of pandas and sm.OLS for calculation of regression...)
        # The following few lines of code ensures better management and compatibility

        detrend_proxy = False
        detrend_calib = False
        standardize_proxy = False
        
        # This block is to ensure compatibility with GH's code below
        y_ok =  df['y'][ df['y'].notnull()]
        calib_ok =  df['Calibration'][ df['Calibration'].notnull()]
        time_common = np.intersect1d(y_ok.index.values, calib_ok.index.values)
        reg_ya = df['y'][time_common].values
        reg_xa = df['Calibration'][time_common].values

        
        # if any of the flags above are activated, run the following code. Otherwise, just ignore it all. 
        if detrend_proxy or detrend_calib or standardize_proxy:

            # save copies of the original data for residual estimates later
            reg_xa_all = np.copy(reg_xa)
            reg_ya_all = np.copy(reg_ya)

            if detrend_proxy:
                # proxy detrend: (1) linear regression, (2) fit, (3) detrend
                xvar = list(range(len(reg_ya)))
                proxy_slope, proxy_intercept, r_value, p_value, std_err = \
                            linregress(xvar, reg_ya)
                proxy_fit = proxy_slope*np.squeeze(xvar) + proxy_intercept
                reg_ya = reg_ya - proxy_fit # detrend for proxy

            if detrend_calib:
                # calibration detrend: (1) linear regression, (2) fit, (3) detrend
                xvar = list(range(len(reg_xa)))
                calib_slope, calib_intercept, r_value, p_value, std_err = \
                            linregress(xvar, reg_xa)
                calib_fit = calib_slope*np.squeeze(xvar) + calib_intercept
                reg_xa = reg_xa - calib_fit

            if standardize_proxy:
                print('Calib stats (x)              [min, max, mean, std]:', np.nanmin(
                    reg_xa), np.nanmax(reg_xa), np.nanmean(reg_xa), np.nanstd(reg_xa))
                print('Proxy stats (y:original)     [min, max, mean, std]:', np.nanmin(
                    reg_ya), np.nanmax(reg_ya), np.nanmean(reg_ya), np.nanstd(reg_ya))

                # standardize proxy values over period of overlap with calibration data
                reg_ya = (reg_ya - np.nanmean(reg_ya))/np.nanstd(reg_ya)
                print('Proxy stats (y:standardized) [min, max, mean, std]:', np.nanmin(
                    reg_ya), np.nanmax(reg_ya), np.nanmean(reg_ya), np.nanstd(reg_ya))
                # GH: note that std_err pertains to the slope, not the residuals!!!


            # Build new pandas DataFrame that contains the modified data:
            # Use pandas DataFrame to store proxy & calibration data side-by-side
            header = ['variable', 'y']
            # Fill-in proxy data
            dfmod = pd.DataFrame({'time':common_time, 'y': reg_ya})
            dfmod.columns = header
            # Add calibration data
            frame = pd.DataFrame({'variable':common_time, 'Calibration':reg_xa})
            dfmod = dfmod.merge(frame, how='outer', on='variable')

            col0 = dfmod.columns[0]
            dfmod.set_index(col0, drop=True, inplace=True)
            dfmod.index.name = 'time'
            dfmod.sort_index(inplace=True)

            # Perform the regression using updated arrays
            regress = sm.ols(formula="y ~ Calibration", data=dfmod).fit()

            
        # END NEW (GH) 21 June 2015 ... RT edit June 2016

        
        # Assign PSM calibration attributes
        # Extract the needed regression parameters
        print(regress.params)
        self.intercept         = regress.params[0]
        self.slope             = regress.params[1]
        self.NbPts             = nobs
        self.corr              = np.sqrt(regress.rsquared)
        if self.slope < 0: self.corr = -self.corr
        
        # Stats on fit residuals
        MSE = np.mean((regress.resid) ** 2)
        self.R = MSE
        SSE = np.sum((regress.resid) ** 2)
        self.SSE = SSE

        # Model information
        self.AIC = regress.aic
        self.BIC = regress.bic
        self.R2    = regress.rsquared
        self.R2adj = regress.rsquared_adj
#<<<<<<< HEAD
#
#=======
#        
#>>>>>>> ea212b8664b415cb895130adf2c57704487461f0
        # Extra diagnostics
        self.calib_time = time_common
        self.calib_refer_values = reg_xa
        self.calib_proxy_values = reg_ya
        fit = self.slope * reg_xa + self.intercept
        self.calib_proxy_fit = fit


        if diag_output:
            # Diagnostic output
            print("***PSM stats:")
            print(regress.summary())

            if diag_output_figs:
                # Figure (scatter plot w/ summary statistics)
                line = self.slope * reg_xa + self.intercept
                plt.plot(reg_xa, line, 'r-', reg_xa, reg_ya, 'o',
                           markersize=7, markerfacecolor='#5CB8E6',
                           markeredgecolor='black', markeredgewidth=1)
                # GH: I don't know how this ran in the first place; must exploit
                # some global namespace
                plt.title('%s: %s' % (proxy.type, proxy.id))
                plt.xlabel('Calibration data')
                plt.ylabel('Proxy data')
                xmin, xmax, ymin, ymax = plt.axis()
                # Annotate with summary stats
                ypos = ymax - 0.05 * (ymax - ymin)
                xpos = xmin + 0.025 * (xmax - xmin)
                plt.text(xpos, ypos, 'Nobs = %s' % str(nobs), fontsize=12,
                           fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                plt.text(xpos, ypos,
                           'Slope = %s' % "{:.4f}".format(self.slope),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                plt.text(xpos, ypos,
                           'Intcpt = %s' % "{:.4f}".format(self.intercept),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                plt.text(xpos, ypos, 'Corr = %s' % "{:.4f}".format(self.corr),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                plt.text(xpos, ypos, 'Res.MSE = %s' % "{:.4f}".format(MSE),
                           fontsize=12, fontweight='bold')

                plt.savefig('proxy_%s_%s_LinearModel_calib.png' % (
                    proxy.type.replace(" ", "_"), proxy.id),
                              bbox_inches='tight')
                plt.close()

    @staticmethod
    def _load_psm_data(config):
        """Helper method for loading from dataframe"""
        pre_calib_file = config.psm.linear.pre_calib_datafile
        
        return load_cpickle(pre_calib_file)


class LinearPSM_TorP(LinearPSM):
    """
    PSM based on linear regression w.r.t. temperature or precipitation
    
    **Important note: This class assumes that all linear PSMs have been 
                      pre-calibrated w.r.t. to temperature and precipitation, 
                      using the "LinearPSM" class defined above.
                      No attempts at calibration are performed here! 
                      The following code only assigns PSM linear regression 
                      parameters to individual proxies from fits to temperature
                      OR precipitation/moisture according to a measure of
                      "goodness-of-fit" criteria (here, MSE of the resisduals).

    Attributes
    ----------
    sensitivity: string
        Indicates the sensitivity (temperature vs moisture) of the proxy record
    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    corr: float
        Correlation of proxy record against calibration data
    slope: float
        Linear regression slope of proxy/calibration fit
    intercept: float
        Linear regression y-intercept of proxy/calibration fit
    R: float
        Mean-squared error of proxy/calibration fit

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
    psm_data: dict
        Pre-calibrated PSM dictionary containing current associated
        proxy site's calibration information

    Raises
    ------
    ValueError
        If PSM is below critical correlation threshold.
    """

    def __init__(self, config, proxy_obj, psm_data_T=None, psm_data_P=None):

        self.psm_key = 'linear_TorP'
        
        proxy = proxy_obj.type
        site = proxy_obj.id
        r_crit = config.psm.linear_TorP.psm_r_crit
        self.lat  = proxy_obj.lat
        self.lon  = proxy_obj.lon
        self.elev = proxy_obj.elev

        self.datatag_calib_T = config.psm.linear_TorP.datatag_calib_T
        self.datatag_calib_P = config.psm.linear_TorP.datatag_calib_P

        self.avgPeriod = config.psm.linear_TorP.avgPeriod

        
        # Try using pre-calibrated psm_data for **temperature**
        try:
            if psm_data_T is None:
                psm_data_T = self._load_psm_data(config,'temperature')
            psm_site_data_T = psm_data_T[(proxy, site)]
        except (KeyError, IOError) as e:
            # No precalibration found, have to do it for this proxy
            # print 'PSM (temperature) not calibrated for:' + str((proxy, site))
            #_log.error(e)
            _log.info('PSM (temperature) not calibrated for:' +
                        str((proxy, site)))

        # Try using pre-calibrated psm_data for **precipitation/moisture**
        try:
            if psm_data_P is None:
                psm_data_P = self._load_psm_data(config,'moisture')
            psm_site_data_P = psm_data_P[(proxy, site)]

        except (KeyError, IOError) as e:
            # No precalibration found, have to do it for this proxy
            #_log.error(e)
            _log.info('PSM (moisture) not calibrated for:' +
                        str((proxy, site)))

        # Check if PSM is calibrated w.r.t. temperature and/or w.r.t.
        # precipitation/moisture. Assign proxy "sensitivity" based on
        # PSM calibration.

        # First check if required PSM calibration data is available      
        psm_ok = True
        if not psm_data_T:
            print ('In "LinearPSM_TorP" class: Cannot find PSM calibration data for temperature.')
            psm_ok = False
        if not psm_data_P:
            print ('In "LinearPSM_TorP" class: Cannot find PSM calibration data for moisture.')
            psm_ok = False
        if not psm_ok:
            raise SystemExit

        # Now check about temperature vs. moisture ...
        if (proxy, site) in list(psm_data_T.keys()) and (proxy, site) in list(psm_data_P.keys()):
            # calibrated for temperature AND for precipitation, check relative goodness of fit 
            # and assign "sensitivity" & corresponding PSM parameters according to best fit.
            if psm_site_data_T['PSMmse'] < psm_site_data_P['PSMmse']:
                # smaller residual errors w.r.t. temperature
                self.sensitivity = 'temperature'
                self.corr = psm_site_data_T['PSMcorrel']
                self.slope = psm_site_data_T['PSMslope']
                self.intercept = psm_site_data_T['PSMintercept']
                self.R = psm_site_data_T['PSMmse']
                self.seasonality = psm_site_data_T['Seasonality']
            else:
                # smaller residual errors w.r.t. precipitation/moisture
                self.sensitivity = 'moisture'
                self.corr = psm_site_data_P['PSMcorrel']
                self.slope = psm_site_data_P['PSMslope']
                self.intercept = psm_site_data_P['PSMintercept']
                self.R = psm_site_data_P['PSMmse']
                self.seasonality = psm_site_data_P['Seasonality']
                
        elif (proxy, site) in list(psm_data_T.keys()) and (proxy, site) not in list(psm_data_P.keys()):
            # calibrated for temperature and NOT for precipitation, assign as "temperature" sensitive
            self.sensitivity = 'temperature'
            self.corr = psm_site_data_T['PSMcorrel']
            self.slope = psm_site_data_T['PSMslope']
            self.intercept = psm_site_data_T['PSMintercept']
            self.R = psm_site_data_T['PSMmse']
            self.seasonality = psm_site_data_T['Seasonality']
        elif (proxy, site) not in list(psm_data_T.keys()) and (proxy, site) in list(psm_data_P.keys()):
            # calibrated for precipitation/moisture and NOT for temperature, assign as "moisture" sensitive
            self.sensitivity = 'moisture'
            self.corr = psm_site_data_P['PSMcorrel']
            self.slope = psm_site_data_P['PSMslope']
            self.intercept = psm_site_data_P['PSMintercept']
            self.R = psm_site_data_P['PSMmse']
            self.seasonality = psm_site_data_P['Seasonality']
        else:
            raise ValueError('Could not find proxy in pre-calibration file... '
                             'Skipping: {}'.format(proxy_obj.id))

        
        # Raise exception if critical correlation value not met
        if abs(self.corr) < r_crit:
            raise ValueError(('Proxy model correlation ({:.2f}) does not meet '
                              'critical threshold ({:.2f}).'
                              ).format(self.corr, r_crit))

    # TODO: Ideally prior state info and coordinates should all be in single obj
    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """
        
        # ----------------------
        # Calculate the Ye's ...
        # ----------------------
        if self.sensitivity:
            # "sensitivity" defined for this proxy
            if self.sensitivity == 'temperature':
                state_var = 'tas_sfc_Amon'
            elif self.sensitivity == 'moisture':
                # To consider different moisture variables
                if self.datatag_calib_P == 'GPCC':
                    state_var = 'pr_sfc_Amon'
                elif self.datatag_calib_P == 'DaiPDSI':
                    state_var = 'scpdsi_sfc_Amon'
                else:
                    raise KeyError('Unrecognized calibration-state variable association.'
                                   ' State variable not identified for Ye calculation.')
            else:
                raise KeyError('Unkown PSM sensitivity. PSM not identified for Ye'
                               ' calculation.')
        else:
            # "sensitivity" not defined for this proxy: simply revert to the default (temperature)
            state_var = 'tas_sfc_Amon'

        if state_var not in list(X_state_info.keys()):
            raise KeyError('Needed variable not in state vector for Ye'
                           ' calculation.')
        
        # TODO: end index should already be +1, more pythonic
        calibvar_startidx, calibvar_endidx = X_state_info[state_var]['pos']
        ind_lon = X_state_info[state_var]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var]['spacecoords'].index('lat')
        X_lons = X_coords[calibvar_startidx:(calibvar_endidx+1), ind_lon]
        X_lats = X_coords[calibvar_startidx:(calibvar_endidx+1), ind_lat]
        
        # Extract data from closest grid point to location of proxy site
        # [self.lat,self.lon]
        calibvar_data = Xb[calibvar_startidx:(calibvar_endidx+1), :]
        # get_close_grid_point_data is inherited from the LinearPSM class
        # on which this one is based.
        gridpoint_data = self.get_close_grid_point_data(calibvar_data.T,
                                                        X_lons,
                                                        X_lats)

        Ye = self.slope * gridpoint_data + self.intercept
        
        return Ye

    # Define the error model for this proxy
    @staticmethod
    def error():
        return 0.1

    # TODO: Simplify a lot of the actions in the calibration
    def calibrate(self, C, proxy, diag_output=False, diag_output_figs=False):
        """
        Calibrate given proxy record against observation data and set relevant
        PSM attributes.

        Parameters
        ----------
        C: calibration_master like
            Calibration object containing data, time, lat, lon info
        proxy: BaseProxyObject like
            Proxy object to fit to the calibration data
        diag_output, diag_output_figs: bool, optional
            Diagnostic output flags for calibration method
        """

        print('Calibration not performed in this psm class!')
        pass

    @classmethod
    def get_kwargs(cls, config):
        try:
            psm_data_T = cls._load_psm_data(config, calib_var = 'temperature')
            psm_data_P = cls._load_psm_data(config, calib_var = 'moisture')

            return {'psm_data_T': psm_data_T, 'psm_data_P': psm_data_P}
        except IOError as e:
            print(e)
            return {}

    @staticmethod
    def _load_psm_data(config,calib_var):
        """Helper method for loading from dataframe"""
        if calib_var == 'temperature':
            pre_calib_file = config.psm.linear_TorP.pre_calib_datafile_T
        elif calib_var == 'moisture':
            pre_calib_file = config.psm.linear_TorP.pre_calib_datafile_P
        else:
            raise ValueError
        
        return load_cpickle(pre_calib_file)


class BilinearPSM(BasePSM):
    """
    PSM based on bivariate linear regression w.r.t. temperature AND precipitation/moisture
    

    Attributes
    ----------
    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    corr: float
        Correlation of proxy record against calibration data
    slope: float
        Linear regression slope of proxy/calibration fit
    intercept: float
        Linear regression y-intercept of proxy/calibration fit
    R: float
        Mean-squared error of proxy/calibration fit

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
    psm_data: dict
        Pre-calibrated PSM dictionary containing current associated
        proxy site's calibration information

    Raises
    ------
    ValueError
        If PSM is below critical correlation threshold.
    """

    def __init__(self, config, proxy_obj, psm_data=None, calib_obj_T=None, calib_obj_P=None):

        self.psm_key = 'bilinear'

        proxy = proxy_obj.type
        site = proxy_obj.id
        r_crit = config.psm.bilinear.psm_r_crit
        self.lat  = proxy_obj.lat
        self.lon  = proxy_obj.lon
        self.elev = proxy_obj.elev

        self.avgPeriod = config.psm.bilinear.avgPeriod
        
        # Assign sensitivity as temperature_moisture
        self.sensitivity = 'temperature_moisture'

        # Try using pre-calibrated psm_data
        try:
            if psm_data is None:
                psm_data = self._load_psm_data(config)
            psm_site_data = psm_data[(proxy, site)]
            
            self.corr = psm_site_data['PSMcorrel']
            self.slope_temperature = psm_site_data['PSMslope_temperature']
            self.slope_moisture = psm_site_data['PSMslope_moisture']
            self.intercept = psm_site_data['PSMintercept']
            self.R = psm_site_data['PSMmse']

            self.calib_temperature = psm_site_data['calib_temperature']
            self.calib_moisture = psm_site_data['calib_moisture']

            # check if seasonality defined in the psm data
            # if it is, return as an attribute of psm object
            if 'Seasonality' in list(psm_site_data.keys()):
                self.seasonality = psm_site_data['Seasonality']
            
        except KeyError as e:
            raise ValueError('Could not find proxy in pre-calibration file... '
                             'Skipping: {}'.format(proxy_obj.id))
        except IOError as e:
            # No precalibration found, have to do it for this proxy
            print(('No pre-calibration found for'
                   ' {} ({}) ... calibrating ...'.format(proxy_obj.id,
                                                         proxy_obj.type)))

            
            # check if calibration object already exists 
            # Two calibration sources to consider           
            if calib_obj_T is None and  calib_obj_P is None:

                datag_calib_T = config.psm.bilinear.datatag_calib_T
                datag_calib_P = config.psm.bilinear.datatag_calib_P

                self.calib_temperature = datag_calib_T
                self.calib_moisture = datag_calib_P
            
                # Two calibration objects, temperature and precipitation/moisture
                calib_obj_T = LMR_calibrate.calibration_assignment(datag_calib_T)
                calib_obj_T.datadir_calib = config.psm.bilinear.datadir_calib
                calib_obj_T.read_calibration()
                calib_obj_P = LMR_calibrate.calibration_assignment(datag_calib_P)
                calib_obj_P.datadir_calib = config.psm.bilinear.datadir_calib
                calib_obj_P.read_calibration()

            self.calibrate(calib_obj_T, calib_obj_P, proxy_obj)
            
        # Raise exception if critical correlation value not met
        if abs(self.corr) < r_crit:
            raise ValueError(('Proxy model correlation ({:.2f}) does not meet '
                              'critical threshold ({:.2f}).'
                              ).format(self.corr, r_crit))


    # TODO: Ideally prior state info and coordinates should all be in single obj
    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # Defining state variables to consider in the calculation of Ye's
        state_var_T = 'tas_sfc_Amon'
        if self.calib_moisture == 'GPCC':
            state_var_P = 'pr_sfc_Amon'
        elif self.calib_moisture == 'DaiPDSI':
            state_var_P = 'scpdsi_sfc_Amon'
        else:
            raise KeyError('Unrecognized calibration-state variable association. State variable not identified for Ye'
                           ' calculation.')

        if state_var_T not in list(X_state_info.keys()) or state_var_P not in list(X_state_info.keys()):
            raise KeyError('One or more needed variable not in state vector for Ye'
                           ' calculation.')

        # Do not assume that temperature and precipitation/moisture variables are on same grid (for generality)
        # So need to make distinct distance calculations for both state variables
        # Temperature ---
        # TODO: end index should already be +1, more pythonic
        calibvar_startidx, calibvar_endidx = X_state_info[state_var_T]['pos']
        ind_lon = X_state_info[state_var_T]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var_T]['spacecoords'].index('lat')
        # Find row index of X for which [X_lat,X_lon] corresponds to closest
        # grid point to location of proxy site [self.lat,self.lon]
        # Calclulate distances from proxy site.
        stateDim = calibvar_endidx - calibvar_startidx + 1
        ensDim = Xb.shape[1]
        dist = np.empty(stateDim)
        dist = haversine(self.lon, self.lat,
                          X_coords[calibvar_startidx:(calibvar_endidx+1), ind_lon],
                          X_coords[calibvar_startidx:(calibvar_endidx+1), ind_lat])

        # row index of nearest grid pt. in prior (minimum distance)
        kind_T = np.unravel_index(dist.argmin(), dist.shape)[0] + calibvar_startidx

        # Precipitation/Moisture ---
        # TODO: end index should already be +1, more pythonic
        calibvar_startidx, calibvar_endidx = X_state_info[state_var_P]['pos']
        ind_lon = X_state_info[state_var_P]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var_P]['spacecoords'].index('lat')
        # Find row index of X for which [X_lat,X_lon] corresponds to closest
        # grid point to location of proxy site [self.lat,self.lon]
        # Calclulate distances from proxy site.
        stateDim = calibvar_endidx - calibvar_startidx + 1
        ensDim = Xb.shape[1]
        dist = np.empty(stateDim)
        dist = haversine(self.lon, self.lat,
                          X_coords[calibvar_startidx:(calibvar_endidx+1), ind_lon],
                          X_coords[calibvar_startidx:(calibvar_endidx+1), ind_lat])

        # row index of nearest grid pt. in prior (minimum distance)
        kind_P = np.unravel_index(dist.argmin(), dist.shape)[0] + calibvar_startidx

        # Now calculate the Ye's
        Ye = self.slope_temperature * np.squeeze(Xb[kind_T, :]) + self.slope_moisture * np.squeeze(Xb[kind_P, :]) + self.intercept

        return Ye

    # Define the error model for this proxy
    @staticmethod
    def error():
        return 0.1

    # TODO: Simplify a lot of the actions in the calibration
    def calibrate(self, C_T, C_P, proxy, diag_output=False, diag_output_figs=False):
        """
        Calibrate given proxy record against observation data and set relevant
        PSM attributes.

        Parameters
        ----------
        C_T: calibration_master like
            Calibration object containing temperature data, time, lat, lon info
        C_P: calibration_master like
            Calibration object containing precipitation/moisture data, time, lat, lon info
        proxy: BaseProxyObject like
            Proxy object to fit to the calibration data
        diag_output, diag_output_figs: bool, optional
            Diagnostic output flags for calibration method
        """
        
        calib_spatial_avg = False
        Npts = 9  # nb of neighboring pts used in smoothing

        #print 'Calibrating: ', '{:25}'.format(proxy.id), '{:35}'.format(proxy.type)
        
        # ----------------------------------------------
        # Use bilinear model (regression) as default PSM
        # ----------------------------------------------

        nbmaxnan = 0
        
        # Look for indices of calibration grid point closest in space (in 2D)
        # to proxy site
        # For temperature calibration dataset
        dist_T = get_distance(proxy.lon, proxy.lat, C_T.lon, C_T.lat)
        # indices of nearest grid pt.
        jind_T, kind_T = np.unravel_index(dist_T.argmin(), dist_T.shape)
        # For precipitation/moisture calibration dataset
        dist_P = get_distance(proxy.lon, proxy.lat, C_P.lon, C_P.lat)
        # indices of nearest grid pt.
        jind_P, kind_P = np.unravel_index(dist_P.argmin(), dist_P.shape)

        # Apply spatial smoother to calibration gridded data, if option is activated
        if calib_spatial_avg:
            # temperature
            C2Dsmooth = np.zeros([C_T.time.shape[0], C_T.lat.shape[0], C_T.lon.shape[0]])
            for m in range(C_T.time.shape[0]):
                C2Dsmooth[m, :, :] = smooth2D(C_T.temp_anomaly[m, :, :], n=Npts)            
            calvals_T = C2Dsmooth[:, jind_T, kind_T]

            # precipitation/moisture
            C2Dsmooth = np.zeros([C_P.time.shape[0], C_P.lat.shape[0], C_P.lon.shape[0]])
            for m in range(C_P.time.shape[0]):
                C2Dsmooth[m, :, :] = smooth2D(C_P.temp_anomaly[m, :, :], n=Npts)            
            calvals_P = C2Dsmooth[:, jind_P, kind_P]
            
        else:
            calvals_T = C_T.temp_anomaly[:, jind_T, kind_T]
            calvals_P = C_P.temp_anomaly[:, jind_P, kind_P]


        # -------------------------------------------------------
        # Calculate averages of calibration data over appropriate
        # intervals (annual or according to proxy seasonality
        # -------------------------------------------------------
        if self.avgPeriod == 'annual':
            # Simply use annual averages
            avgMonths_T = [1,2,3,4,5,6,7,8,9,10,11,12]
            avgMonths_P = [1,2,3,4,5,6,7,8,9,10,11,12]
        elif 'season' in self.avgPeriod:

            # Distinction btw temperature & moisture seasonalities?
            if hasattr(proxy,'seasonality_T') and  hasattr(proxy,'seasonality_P'):
                avgMonths_T =  proxy.seasonality_T
                avgMonths_P =  proxy.seasonality_P
            else:
                # Revert to the seasonality of the proxy record from the
                # original metadata
                avgMonths_T =  proxy.seasonality
                avgMonths_P =  proxy.seasonality

        else:
            raise SystemExit('ERROR: Unrecognized value for avgPeriod! Exiting!')

        nbmonths_T = len(avgMonths_T)
        nbmonths_P = len(avgMonths_P)
            
        # Temperature data
        cyears_T = list(set([C_T.time[k].year for k in range(len(C_T.time))])) # 'set' is used to get unique values in list
        nbcyears_T = len(cyears_T)
        reg_x_T = np.zeros(shape=[nbcyears_T])
        reg_x_T[:] = np.nan # initialize with nan's

        for i in range(nbcyears_T):
            # monthly data from current year
            indsyr = [j for j,v in enumerate(C_T.time) if v.year == cyears_T[i] and v.month in avgMonths_T]
            # check if data from previous year is to be included
            indsyrm1 = []
            if any(m < 0 for m in avgMonths_T):
                year_before = [abs(m) for m in avgMonths_T if m < 0]
                indsyrm1 = [j for j,v in enumerate(C_T.time) if v.year == cyears_T[i]-1. and v.month in year_before]
            # check if data from following year is to be included
            indsyrp1 = []
            if any(m > 12 for m in avgMonths_T):
                year_follow = [m-12 for m in avgMonths_T if m > 12]
                indsyrp1 = [j for j,v in enumerate(C_T.time) if v.year == cyears_T[i]+1. and v.month in year_follow]

            inds = indsyrm1 + indsyr + indsyrp1
            if len(inds) == nbmonths_T: # all months are in the data
                tmp = np.nanmean(calvals_T[inds],axis=0)
                nancount = np.isnan(calvals_T[inds]).sum(axis=0)
                if nancount > nbmaxnan: tmp = np.nan
            else:
                tmp = np.nan
            reg_x_T[i] = tmp

        
        # Moisture data
        cyears_P = list(set([C_P.time[k].year for k in range(len(C_P.time))])) # 'set' is used to get unique values in list
        nbcyears_P = len(cyears_P)
        reg_x_P = np.zeros(shape=[nbcyears_P])
        reg_x_P[:] = np.nan # initialize with nan's

        for i in range(nbcyears_P):
            # monthly data from current year
            indsyr = [j for j,v in enumerate(C_P.time) if v.year == cyears_P[i] and v.month in avgMonths_P]
            # check if data from previous year is to be included
            indsyrm1 = []
            if any(m < 0 for m in avgMonths_P):
                year_before = [abs(m) for m in avgMonths_P if m < 0]
                indsyrm1 = [j for j,v in enumerate(C_P.time) if v.year == cyears_P[i]-1. and v.month in year_before]
            # check if data from following year is to be included
            indsyrp1 = []
            if any(m > 12 for m in avgMonths_P):
                year_follow = [m-12 for m in avgMonths_P if m > 12]
                indsyrp1 = [j for j,v in enumerate(C_P.time) if v.year == cyears_P[i]+1. and v.month in year_follow]

            inds = indsyrm1 + indsyr + indsyrp1
            if len(inds) == nbmonths_P: # all months are in the data
                tmp = np.nanmean(calvals_P[inds],axis=0)
                nancount = np.isnan(calvals_P[inds]).sum(axis=0)
                if nancount > nbmaxnan: tmp = np.nan
            else:
                tmp = np.nan
            reg_x_P[i] = tmp


        # making sure calibration anomalies are unbiased (avg =0) (NEW: RT 12/22/16)
        #reg_x_T = reg_x_T - np.nanmean(reg_x_T)
        #reg_x_P = reg_x_P - np.nanmean(reg_x_P)

            
        # ---------------------------
        # Perform bilinear regression
        # ---------------------------

        # Use panda DataFrame to store proxy & calibration data side-by-side
        header = ['variable', 'y']
        # Fill-in proxy data
        df = pd.DataFrame({'time':proxy.time, 'y': proxy.values})
        df.columns = header
        # Add temperature calibration data
        frameT = pd.DataFrame({'variable':cyears_T, 'Temperature':reg_x_T})
        df = df.merge(frameT, how='outer', on='variable')
        # Add precipitation/moisture calibration data
        frameP = pd.DataFrame({'variable':cyears_P, 'Moisture':reg_x_P})
        df = df.merge(frameP, how='outer', on='variable')
        col0 = df.columns[0]
        df.set_index(col0, drop=True, inplace=True)
        df.index.name = 'Time'
        df.sort_index(inplace=True)

        # ensure all df entries are floats: if not, sm.ols gives garbage
        df = df.astype(np.float)
        
        # Perform the regression
        nobs = None
        try:
            regress = sm.ols(formula="y ~ Temperature + Moisture", data=df).fit()
            # number of data points used in the regression
            nobs = int(regress.nobs)
        except ValueError: # no overlapping data
            nobs = 0
        except Exception as exception:
            print('Exception raised', exception)
            
        if nobs is not None and nobs < 25:  # skip if insufficient overlapping data
            print('Insufficient obs./calib. overlap'
                  ' to calibrate psm: nobs=%d' %nobs)
            raise ValueError()


        # extract the needed regression parameters
        self.intercept         = regress.params[0]
        self.slope_temperature = regress.params[1]
        self.slope_moisture    = regress.params[2]

        self.NbPts = nobs
        self.corr  = np.sqrt(regress.rsquared)

        # Stats on fit residuals
        MSE = np.mean((regress.resid) ** 2)
        self.R = MSE
        SSE = np.sum((regress.resid) ** 2)
        self.SSE = SSE

        # Model information
        self.AIC = regress.aic
        self.BIC = regress.bic
        self.R2    = regress.rsquared
        self.R2adj = regress.rsquared_adj
#<<<<<<< HEAD
#
#=======
#        
#>>>>>>> ea212b8664b415cb895130adf2c57704487461f0
        # Extra diagnostics
        # ... add here ...
        
        y_ok =  df['y'][ df['y'].notnull()]
        calib_T_ok =  df['Temperature'][ df['Temperature'].notnull()]
        calib_P_ok =  df['Moisture'][ df['Moisture'].notnull()]
        calib_ok =  np.intersect1d(calib_T_ok.index.values,calib_P_ok.index.values)
        time_common = np.intersect1d(y_ok.index.values, calib_ok)
        
        reg_ya = df['y'][time_common].values
        reg_xa_T = df['Temperature'][time_common].values
        reg_xa_P = df['Moisture'][time_common].values

        self.calib_time = time_common
        self.calib_proxy_values = reg_ya
        self.calib_temperature_refer_values = reg_xa_T
        self.calib_moisture_refer_values = reg_xa_P        
        fit = self.slope_temperature * reg_xa_T + self.slope_moisture * reg_xa_P + self.intercept
        self.calib_proxy_fit = fit


        diag_output = False
        diag_output_figs = False
        
        if diag_output:
            # Diagnostic output
            print("***PSM stats:")
            print(regress.summary())
            print(' ')
            print('Pairwise correlations:')
            print('----------------------')
            print(df.corr())
            print(' ')
            print(' ')
            
            if diag_output_figs:
                # Figure (scatter plot w/ summary statistics)

                y = df['y']
                X = df[['Temperature','Moisture']]
                
                xx1, xx2 = np.meshgrid(np.linspace(X.Temperature.min(),X.Temperature.max(),200),
                                       np.linspace(X.Moisture.min(),X.Moisture.max(),200))
                Model = self.intercept + self.slope_temperature*xx1 + self.slope_moisture*xx2
                
                fig = plt.figure(figsize=(10,7))
                ax = Axes3D(fig,azim=-130,elev=15)

                # plot hyperplane
                surf = ax.plot_surface(xx1,xx2,Model,cmap=plt.cm.RdBu_r,alpha=0.6,linewidth=0)

                # plot data points                
                resid = y - (self.intercept + self.slope_temperature*X['Temperature'] + self.slope_temperature*X['Moisture'])
                ax.scatter(X[resid>=0.].Temperature,X[resid>=0.].Moisture,y[resid>=0.], color='black', alpha=1.0,facecolor='white')
                ax.scatter(X[resid<0.].Temperature,X[resid<0.].Moisture,y[resid<0.], color='black', alpha=1.0)
                
                # axis labels
                plt.suptitle('%s: %s' % (proxy.type, proxy.id),fontsize=16,fontweight='bold')

                ax.set_xlabel('Calibration temperature')
                ax.set_ylabel('Calibration moisture')
                ax.set_zlabel('Proxy data')

                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                zmin, zmax = ax.get_zlim()

                ax.text2D(0.05,0.92, 'Nobs=%s  Correlation=%s  MSE=%s' %(self.NbPts, "{:.4f}".format(self.corr),"{:.4f}".format(self.R)), transform=ax.transAxes, fontsize=14)
                ax.text2D(0.05,0.89, 'proxy = %s + (%s x Temperature) + (%s x Moisture)' %("{:.4f}".format(self.intercept),
                                                                                                 "{:.4f}".format(self.slope_temperature),
                                                                                                 "{:.4f}".format(self.slope_moisture)), transform=ax.transAxes, fontsize=14)
                
                plt.savefig('proxy_%s_%s_BilinearModel_calib.png' % (
                    proxy.type.replace(" ", "_"), proxy.id),
                              bbox_inches='tight')
                plt.close()

    @staticmethod
    def _load_psm_data(config):
        """Helper method for loading from dataframe"""

        pre_calib_file = config.psm.bilinear.pre_calib_datafile
        
        return load_cpickle(pre_calib_file)


class h_interpPSM(BasePSM):
    """
    Horizontal interpolator as a PSM.
    Interpolator is a distance-weighted average of surrounding gridpoint values, 
    with weights determined by exponentially-decaying function calculated using
    a user-defined influence radius.

    Attributes
    ----------
    RadiusInfluence: float
        Distance-scale used the calculation of exponentially-decaying weights in interpolator (in km)
    sensitivity: string
        Indicates the sensitivity (temperature vs moisture) of the proxy record
        Here set to "None" as this information is irrelevant, but given for 
        compatibility with other PSM classes
    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    R: float
        Obs. error variance associated to proxy site

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to

    Raises
    ------
    ValueError
        ...
    """
    def __init__(self, config, proxy_obj, R_data=None):
        print('at h_interp PSM object in LMR_psms!')
        # JBadgeley: More flexible use of h_interp: Now name of state variable associated
        # to each proxy type specified along h_interp in config.
        # This assigns psm_key as "h_interp,<<name of state variable>>"
        proxy_db = config.proxies.use_from[0]
        self.psm_key = getattr(config.proxies, proxy_db).proxy_psm_type[proxy_obj.type]
        
        self.proxy = proxy_obj.type
        self.site = proxy_obj.id
        print(self.proxy, self.site)
        self.lat  = proxy_obj.lat
        self.lon  = proxy_obj.lon
        try:
            self.elev = proxy_obj.elev
        except:
            self.elev = None        
        self.sensitivity = None
        self.RadiusInfluence = config.psm.h_interp.radius_influence
        self.psm_required_variables = config.psm.h_interp.psm_required_variables
        
        # Try finding file containing obs. error variance info
        # and assign the R value to proxy psm object
        try:
            print('try statement in h_interp PSM initiated')
            if R_data is None:
                print('R_data is None')
                R_data = self._load_psm_data(config)
                print('here')
            self.R = R_data[(self.proxy, self.site)]
            print(self.proxy,self.site,self.R)  
        except (KeyError, IOError) as e:
            print('exception passed')
            # No obs. error variance data found
            _log.info('Cannot find obs. error variance data for:' + str((self.proxy, self.site)))
        print('made it to here')

    # TODO: Ideally prior state info and coordinates should all be in single obj
    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # define state variable that should be interpolated to proxy site
        state_var = self.psm_key.split(',')[1]

        if state_var not in list(X_state_info.keys()):
            raise KeyError('Needed variable %s not in state vector for Ye'
                           ' calculation.' %state_var)

        # TODO: end index should already be +1, more pythonic
        statevar_startidx, statevar_endidx = X_state_info[state_var]['pos']
        ind_lon = X_state_info[state_var]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var]['spacecoords'].index('lat')

        # Find row index of X for which [X_lat,X_lon] corresponds to closest
        # grid point to location of proxy site [self.lat,self.lon]

        # Calclulate distances from proxy site.
        stateDim = statevar_endidx - statevar_startidx + 1
        ensDim = Xb.shape[1]
        dist = np.empty(stateDim)
        dist = haversine(self.lon, self.lat,
                          X_coords[statevar_startidx:(statevar_endidx+1), ind_lon],
                          X_coords[statevar_startidx:(statevar_endidx+1), ind_lat])

        # Check if RadiusInfluence is defined (not None). Use weighted-averaging as interpolator if it is.
        # Otherwise, seek value at nearest gridpoint to proxy location
        if self.RadiusInfluence:
            # Calculate weighted-average
            #  exponential decay
            L =  self.RadiusInfluence
            weights = np.exp(-np.square(dist)/np.square(L))
            # make the weights sum to one
            weights /=weights.sum(axis=0)

            Ye = np.dot(weights.T,Xb[statevar_startidx:(statevar_endidx+1), :])
        else:
            # Pick value at nearest grid point
            # row index of nearest grid pt. in prior (minimum distance)
            kind = np.unravel_index(dist.argmin(), dist.shape)[0] + statevar_startidx

            Ye = np.squeeze(Xb[kind, :])


        return Ye

    # Define a default error model for this proxy
    @staticmethod
    def error():
        return 0.1

    @classmethod
    def get_kwargs(cls, config):
        try:
            # obs. error variabce data
            R_data = cls._load_psm_data(config)
            return {'R_data': R_data}
        except IOError as e:
            print(e)
            raise SystemExit
    
    @staticmethod
    def _load_psm_data(config):
        """Helper method for loading from file"""

        R_data_file = config.psm.h_interp.datafile_obsError
        
        if R_data_file:
            # a file is specified, check if it exists
            if not os.path.isfile(R_data_file):
                raise SystemExit('in h_interp PSM: Cannot find file containing obs. error variance data: %s'
                                 %R_data_file)
            else:
                # this returns an array of tuples:
                # (proxy type of type string, proxy site name of type string, R value of type float)
                # transformed into a list
                
                #changed by G. OConnor on 7.10.19 for special characters needed to read new accumulation records in R (i.e. Hercules Neve)
                Rdata_list = np.genfromtxt(R_data_file,delimiter=',',dtype=None, encoding='utf-8').tolist()
                #Rdata_list = np.genfromtxt(R_data_file,delimiter=',',dtype=None, encoding='ascii').tolist()
                
                # transform into a dictionary with (proxy type, proxy site) tuples as keys and R as the paired values
                Rdata_dict = dict([((item[0],item[1]),item[2]) for item in Rdata_list])
        else:
            # a file has not been specified: raise an error
            raise SystemExit('in h_interp PSM: A file containing obs. error variance data '
                             'has not been specified. Exiting.')

        return Rdata_dict


class BayesRegPSM(BasePSM):
    """
    Abstract Class for Bayesian regression-based PSMs
    """
    def __init__(self, config, proxy_obj):
        self.proxy = proxy_obj.type
        self.site = proxy_obj.id
        self.lat  = proxy_obj.lat
        self.lon  = proxy_obj.lon
        self.elev = proxy_obj.elev

    @abstractmethod
    def _estimate_r(self, *args, **kwargs):
        """Fit Bayes reg to range of input values and return max variance

        Parameters
        ----------
        *args:
            Passed to numpy.arange()
        **kwargs:
            Passed to numpy.arange()

        Returns
        -------
        Scalar giving observation error (R) PSM.
        """
        pass

    @abstractmethod
    def _mcmc_predict(self, **kwargs):
        """Get ensemble of MCMC predictions from PSM

        Parameters
        ----------
        **kwargs:

        Returns
        -------
        array (nx, ne) giving an ensemble of PSM realizations. The first
        dim, nx, is equal to length of climate variables input. The second
        dim, ne, is for each ensemble realizations.
        """
        pass

    def _get_gridpoint_data(self, state_var, Xb, X_state_info, X_coords):
        """Extract a variable from a Kalman state vector

        Parameters
        ----------
        state_var: str
            Target variable to extract from Kalman state arrays
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)
        """
        var_startidx, var_endidx = X_state_info[state_var]['pos']
        ind_lon = X_state_info[state_var]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var]['spacecoords'].index('lat')
        X_lons = X_coords[var_startidx:(var_endidx + 1), ind_lon]
        X_lats = X_coords[var_startidx:(var_endidx + 1), ind_lat]
        var_data = Xb[var_startidx:(var_endidx + 1), :]
        # get prior data at proxy site
        gridpoint_data = get_data_closest_gridpt(var_data, X_lons, X_lats,
                                                 self.lon, self.lat, getvalid=True)
        return gridpoint_data


@class_docs_fixer
class BayesRegUK37PSM(BayesRegPSM):
    """
    ... 

    Attributes
    ----------

    ...

    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    R: float
        Obs. error variance associated to proxy site

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to.

    Raises
    ------
    ValueError
        ...
    """

    def __init__(self, config, proxy_obj):
        super().__init__(config, proxy_obj)
        self.psm_key = 'bayesreg_uk37'
        self.sensitivity = 'sst'
        self.psm_required_variables = config.psm.bayesreg_uk37.psm_required_variables
        self.R = self._estimate_r(31)

    def _estimate_r(self, *args, **kwargs):
        xrange = np.arange(*args, **kwargs)
        ensemble = self._mcmc_predict(xrange)
        out = np.max(np.var(ensemble, axis=1))
        
        return out
        
    def _mcmc_predict(self, x):
        """Get MCMC ensemble model prediction for input variable x"""
        y = bayspline.predict_uk(sst=x)
        return y.ensemble

    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # Defining state variables to consider in the calculation of Ye's
        # 
        for state_var in list(self.psm_required_variables.keys()):
            try:
                varok = X_state_info[state_var]                
            except KeyError as ekey:
                print('In BayesRegUK37PSM: Needed variable (%s) not in state vector for Ye calculation.'
                      %ekey)
                raise SystemExit()
        
        var_startidx, var_endidx = X_state_info[state_var]['pos']
        ind_lon = X_state_info[state_var]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var]['spacecoords'].index('lat')
        X_lons = X_coords[var_startidx:(var_endidx + 1), ind_lon]
        X_lats = X_coords[var_startidx:(var_endidx + 1), ind_lat]
        var_data = Xb[var_startidx:(var_endidx + 1), :]
        # get prior data at proxy site
        gridpoint_data = get_data_closest_gridpt(var_data, X_lons, X_lats,
                                                 self.lon, self.lat,
                                                 getvalid=True)

        # check if gridpoint_data is in K: need deg. C for forward model
        # RT - 11/18: More robust use of "units" metadata now included in state vector info.
        """
        if np.isfinite(gridpoint_data).all() and np.any(gridpoint_data):
            if X_state_info[state_var]['units']:
                if X_state_info[state_var]['units'] == 'K' or X_state_info[state_var]['units'] == 'Kelvin':
                    gridpoint_data = gridpoint_data - 273.15
        else:
            gridpoint_data[:] = np.nan
        """
        mask = np.isfinite(gridpoint_data)
        if X_state_info[state_var]['units']:
            if X_state_info[state_var]['units'] == 'K' or X_state_info[state_var]['units'] == 'Kelvin':
                gridpoint_data[mask] = gridpoint_data[mask] - 273.15
            elif 'Celsius' in X_state_info[state_var]['units'] or X_state_info[state_var]['units'] == 'C' or \
                 X_state_info[state_var]['units'] == 'degree C' or X_state_info[state_var]['units'] == 'deg. C' or \
                 X_state_info[state_var]['units'] == 'degree C.' or X_state_info[state_var]['units'] == 'deg. C.' or \
                 X_state_info[state_var]['units'] == 'degC' :
                pass
            else:
                raise SystemExit('ERROR in BayesRegUK37PSM: unrecognized definition of units for prior temperature.')
        else:
            raise SystemExit('ERROR in BayesRegUK37PSM: units for prior temperature undefined.')
                
            
        Ye_ens = self._mcmc_predict(x=gridpoint_data)
        # take the mean of the ensemble of estimates
        Ye = np.mean(Ye_ens, axis=1)
        return Ye

    # Define a default error model for this proxy
    @staticmethod
    def error():
        return 0.1


@class_docs_fixer
class BayesRegTEX86PSM(BayesRegPSM):
    """
    ...

    Attributes
    ----------

    ...

    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    R: float
        Obs. error variance associated to proxy site

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to.

    Raises
    ------
    ValueError
        ...
    """

    def __init__(self, config, proxy_obj):
        super().__init__(config, proxy_obj)
        self.psm_key = 'bayesreg_tex86'
        self.sensitivity = None
        self.psm_required_variables = config.psm.bayesreg_tex86.psm_required_variables
        self.temptype = str(config.psm.bayesreg_tex86.temptype)
        self.R = self._estimate_r(31)

    def _estimate_r(self, *args, **kwargs):
        """Fit Bayes regression to range of input values and return max variance
        """
        xrange = np.arange(*args, **kwargs)
        ensemble = self._mcmc_predict(xrange)
        out = np.max(np.var(ensemble, axis=1))
        
        return out

        
    def _mcmc_predict(self, x):
        """Get MCMC ensemble model prediction for input variable x"""

        # Longitude is often over 180, raising error in bayspar.
        lon = float(self.lon)
        if lon > 180:
            lon -= 360

        y = bayspar.predict_tex(seatemp=x, lat=self.lat, lon=lon,
                                temptype=self.temptype)
        return y.ensemble

    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # Defining state variables to consider in the calculation of Ye's
        #
        for state_var in list(self.psm_required_variables.keys()):
            try:
                varok = X_state_info[state_var]                
            except KeyError as ekey:
                print('In BayesRegTEX86PSM: Needed variable (%s) not in state vector for Ye calculation.'
                      %ekey)
                raise SystemExit()
        
        var_startidx, var_endidx = X_state_info[state_var]['pos']
        ind_lon = X_state_info[state_var]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var]['spacecoords'].index('lat')
        X_lons = X_coords[var_startidx:(var_endidx + 1), ind_lon]
        X_lats = X_coords[var_startidx:(var_endidx + 1), ind_lat]
        var_data = Xb[var_startidx:(var_endidx + 1), :]
        # get prior data at proxy site
        gridpoint_data = get_data_closest_gridpt(var_data, X_lons, X_lats,
                                                 self.lon, self.lat,
                                                 getvalid=True)

        # check if gridpoint_data is in K: need deg. C for forward model
        # RT - 11/18: More robust use of "units" metadata now included in state vector info.
        mask = np.isfinite(gridpoint_data)
        if X_state_info[state_var]['units']:
            if X_state_info[state_var]['units'] == 'K' or X_state_info[state_var]['units'] == 'Kelvin':
                gridpoint_data[mask] = gridpoint_data[mask] - 273.15
            elif 'Celsius' in X_state_info[state_var]['units'] or X_state_info[state_var]['units'] == 'C' or \
                 X_state_info[state_var]['units'] == 'degree C' or X_state_info[state_var]['units'] == 'deg. C' or \
                 X_state_info[state_var]['units'] == 'degree C.' or X_state_info[state_var]['units'] == 'deg. C.' or \
                 X_state_info[state_var]['units'] == 'degC' :
                pass
            else:
                raise SystemExit('ERROR in BayesRegTEX86PSM: unrecognized definition of units for prior temperature.')
        else:
            raise SystemExit('ERROR in BayesRegUK37PSM: units for prior temperature undefined.')

        ye_ens = self._mcmc_predict(x=gridpoint_data)
        ye = np.mean(ye_ens, axis=1)
        return ye

    # Define a default error model for this proxy
    @staticmethod
    def error():
        return 0.1


@class_docs_fixer
class BayesRegD18oPSM(BayesRegPSM):
    """
    ...

    Attributes
    ----------

    ...

    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    R: float
        Obs. error variance associated to proxy site

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to.

    Raises
    ------
    ValueError
        ...
    """

    def __init__(self, config, proxy_obj, spp=None):
        super().__init__(config, proxy_obj)
        self.psm_key = 'bayesreg_d18o'
        self.sensitivity = None
        self.psm_required_variables = config.psm.bayesreg_d18o.psm_required_variables
        self.psm_seatemp_variable = config.psm.bayesreg_d18o.psm_seatemp_variable
        self.psm_d18osw_variable = config.psm.bayesreg_d18o.psm_d18osw_variable
        self.psm_d18osw_from_salinity = config.psm.bayesreg_d18o.psm_d18osw_from_salinity
        self.seasonal_seatemp = config.psm.bayesreg_d18o.seasonal_seatemp
        self.spp = spp
        self.R = self._estimate_r()

    def _estimate_r(self, *args, **kwargs):
        """Find R, error variance of this PSM. Ignores input args.
        """
        if self.psm_d18osw_from_salinity:
            # We can use a single temperature because it shouldn't matter,
            # but we can't just compute from model parameters because we need
            # to consider uncertainty from SOS -> d18Osw model in addition to
            # the proxy forward model itself.
            ensemble = self._mcmc_predict(sst=np.array([15.0]), sos=np.array([35.0]))
        else:
            # using d18Osw as input, so no SOS -> d18Osw model to consider
            ensemble = self._mcmc_predict(sst=np.array([15.0]), d18osw=np.array([0.0]))
        out = np.max(np.var(ensemble, axis=1))

        return out

    def _mcmc_predict(self, sst, sos=None, d18osw=None):
        """Get MCMC ensemble prediction for input variables
        """
        assert (sos is not None) ^ (d18osw is not None), 'requires `d18osw` or `sos` as input'

        predict_kwargs = {'seatemp': sst,
                          'spp': self.spp,
                          'salinity': sos,
                          'd18osw': d18osw,
                          'latlon': None,
                          'seasonal_seatemp': self.seasonal_seatemp}

        if d18osw is None:
            # Without d18Osw, we need latlon and SOS.
            # Longitude is often over 180, raising error in deltaoxfox.
            lon = float(self.lon)
            if lon > 180:
                lon -= 360
            predict_kwargs['latlon'] = (self.lat, lon)

        y = dfox.predict_d18oc(**predict_kwargs)

        return y.ensemble

    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """
        sst_var_name = list(self.psm_seatemp_variable.keys())[0]
        d18osw_var_name = list(self.psm_d18osw_variable.keys())[0]

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # Defining state variables to consider in the calculation of Ye's
        #
        for state_var in list(self.psm_required_variables.keys()):
            try:
                varok = X_state_info[state_var]                
            except KeyError as ekey:
                print('In BayesRegD18oPSM: Needed variable (%s) not in state vector for Ye calculation.'
                      %ekey)
                raise SystemExit()
                
        xb_sst = self._get_gridpoint_data(sst_var_name, Xb, X_state_info, X_coords)
        xb_d18osw = self._get_gridpoint_data(d18osw_var_name, Xb, X_state_info, X_coords)
        
        # check if gridpoint_data is in K: need deg. C for forward model
        # RT - 11/18: More robust use of "units" metadata now included in state vector info.
        mask = np.isfinite(xb_sst)
        if X_state_info[sst_var_name]['units']:
            if X_state_info[sst_var_name]['units'] == 'K' or X_state_info[sst_var_name]['units'] == 'Kelvin':
                xb_sst[mask] = xb_sst[mask] - 273.15
            elif 'Celsius' in X_state_info[sst_var_name]['units'] or X_state_info[sst_var_name]['units'] == 'C' or \
                 X_state_info[sst_var_name]['units'] == 'degree C' or X_state_info[sst_var_name]['units'] == 'deg. C' or \
                 X_state_info[sst_var_name]['units'] == 'degree C.' or X_state_info[sst_var_name]['units'] == 'deg. C.' or \
                 X_state_info[sst_var_name]['units'] == 'degC' :
                pass
            else:
                raise SystemExit('ERROR in BayesRegD18oPSM: unrecognized definition of units for prior temperature.')
        else:
            raise SystemExit('ERROR in BayesRegD18oPSM: units for prior temperature undefined.')
        
        # Build args to pass to self._mcmc_predict() to handle whether SOS or
        # d18Osw is available in state vector.
        predict_kwargs = {'sst' : xb_sst}
        if self.psm_d18osw_from_salinity:
            predict_kwargs['sos'] = xb_d18osw
        else:
            predict_kwargs['d18osw'] = xb_d18osw

        ye_ens = self._mcmc_predict(**predict_kwargs)
        ye = np.mean(ye_ens, axis=1)
        return ye

    # Define a default error model for this proxy
    @staticmethod
    def error():
        return 0.1


@class_docs_fixer
class BayesRegD18oPooledPSM(BayesRegD18oPSM):
    def __init__(self, config, proxy_obj):
        super().__init__(config, proxy_obj, spp=None)
        self.psm_key = 'bayesreg_d18o_pooled'


@class_docs_fixer
class BayesRegD18oRuberwhitePSM(BayesRegD18oPSM):
    def __init__(self, config, proxy_obj):
        super().__init__(config, proxy_obj, 'G. ruber white')
        self.psm_key = 'bayesreg_d18o_ruberwhite'


@class_docs_fixer
class BayesRegD18oSacculiferPSM(BayesRegD18oPSM):
    def __init__(self, config, proxy_obj):
        super().__init__(config, proxy_obj, 'G. sacculifer')
        self.psm_key = 'bayesreg_d18o_sacculifer'


@class_docs_fixer
class BayesRegD18oBulloidesPSM(BayesRegD18oPSM):
    def __init__(self, config, proxy_obj):
        super().__init__(config, proxy_obj, 'G. bulloides')
        self.psm_key = 'bayesreg_d18o_bulloides'


@class_docs_fixer
class BayesRegD18oPachydermaPSM(BayesRegD18oPSM):
    def __init__(self, config, proxy_obj):
        super().__init__(config, proxy_obj, 'N. pachyderma')
        self.psm_key = 'bayesreg_d18o_pachyderma'


@class_docs_fixer
class BayesRegD18oIncomptaPSM(BayesRegD18oPSM):
    def __init__(self, config, proxy_obj):
        super().__init__(config, proxy_obj, 'N. incompta')
        self.psm_key = 'bayesreg_d18o_incompta'


class BayesRegMgcaPSM(BayesRegPSM):
    """
    ...

    Attributes
    ----------

    ...

    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    R: float
        Obs. error variance associated to proxy site

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to.

    Raises
    ------
    ValueError
        ...
    """

    def __init__(self, config, proxy_obj, cleaning, spp=None):
        super().__init__(config, proxy_obj)
        self.psm_key = 'bayesreg_mgca'
        self.sensitivity = None
        self.psm_seatemp_variable = config.psm.bayesreg_mgca.psm_seatemp_variable
        self.psm_salinity_variable = config.psm.bayesreg_mgca.psm_salinity_variable        
        self.psm_omega_variable = config.psm.bayesreg_mgca.psm_omega_variable
        self.psm_ph_variable = config.psm.bayesreg_mgca.psm_ph_variable        
        self.psm_required_variables = config.psm.bayesreg_mgca.psm_required_variables
        self.spp = spp

        self.seasonal_seatemp = config.psm.bayesreg_mgca.seasonal_seatemp
        self.seawater_age = config.psm.bayesreg_mgca.seawater_age
        self.cleaning = int(cleaning)

        # observation error variance
        #self.R = self._estimate_r(15, 16)  # BM: This is temporary. Should consider wider range. RT: revise this!
        self.R = self._estimate_r()
        
    def _estimate_r(self, *args, **kwargs):
        """Fit Bayes regression to range of input values and return max variance
        """
        #xrange = np.arange(*args, **kwargs)
        # This assumes that changes in salinity will not influence ensemble variance. (RT: but it does!)
        #ensemble = self._mcmc_predict(sst=xrange)
        # Revision: consider wider range of environmental conditions
        # Location specific modern values of pH and omega
        # if longitude expressed in [0,360] domain, need in [-180,180]
        lon = float(self.lon)
        if lon > 180:
            lon -= 360
        depth = np.abs(self.elev)  # `baymag` wants positive depths.
        latlon=(self.lat, lon)
        ph = baymag.omgph.fetch_ph(latlon=latlon)
        omega = baymag.omgph.fetch_omega(latlon,depth)

        # Generate random distributions of SST and SSS values within appropriate ranges
        # to get wide diversity in sst/sss pairs of values
        # NOTE: ranges could be adjusted for the various foram species according to
        # their typical habitats (see Table 1 from Tierney et al. 2019)
        sst_min = 1.
        sst_max = 30.
        sss_min = 10.
        sss_max = 40.
        # 1000 random values within ranges
        sst = sst_min + (np.random.rand(1000) *(sst_max - sst_min))
        sss = sss_min + (np.random.rand(1000) *(sss_max - sss_min))

        ensemble = self._mcmc_predict(sst,sss)

        # for MgCa proxies, DA is performed in log space (ln) as PSM has exponential form
        return np.median(np.var(np.log(ensemble), axis=1))

    def _mcmc_predict(self, sst, sss, ph=None, omega=None):
        """Get MCMC ensemble prediction from input variables
        """
        # Longitude is often over 180.
        lon = float(self.lon)
        if lon > 180:
            lon -= 360
        depth = np.abs(self.elev)  # `baymag` wants positive depths.
        latlon=(self.lat, lon)

        # if ph and omega not available from prior, use modern record
        if ph is None:
            ph = baymag.omgph.fetch_ph(latlon=latlon)
        if omega is None:
            omega = baymag.omgph.fetch_omega(latlon,depth)
        
        # calculate proxy estimates        
#old        y = baymag.predict_mgca(seatemp=sst, spp=self.spp, latlon=(self.lat, lon),
#                                omega=omega, depth=depth, cleaning=self.cleaning,
#                                sw_age=self.seawater_age,
#                                seasonal_seatemp=self.seasonal_seatemp)

        ymod = baymag.predict_mgca(sst, self.cleaning, sss, ph, omega, self.spp)

        # Correction for seawater age (for deep time applications)
        # Do calculation if value for seawater age is provided
        if self.seawater_age:
            y = baymag.sw_correction(ymod, self.seawater_age)
        else:
            y = ymod

        return y.ensemble

    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # Defining state variables to consider in the calculation of Ye's
        #
        for state_var in list(self.psm_required_variables.keys()):
            if state_var not in list(X_state_info.keys()):
                raise KeyError('Needed variable not in state vector for Ye calculation.')

        # temperature
        seatemp_var_name = list(self.psm_seatemp_variable.keys())[0]
        xb_sst = self._get_gridpoint_data(seatemp_var_name, Xb, X_state_info, X_coords)
        
        # check if sst gridpoint_data is in K: need deg. C for forward model
        mask = np.isfinite(xb_sst)
        if X_state_info[seatemp_var_name]['units']:
            if X_state_info[seatemp_var_name]['units'] == 'K' or X_state_info[seatemp_var_name]['units'] == 'Kelvin':
                xb_sst[mask] = xb_sst[mask] - 273.15
            elif 'Celsius' in X_state_info[seatemp_var_name]['units'] or X_state_info[seatemp_var_name]['units'] == 'C' or \
                 X_state_info[seatemp_var_name]['units'] == 'degree C' or X_state_info[seatemp_var_name]['units'] == 'deg. C' or \
                 X_state_info[seatemp_var_name]['units'] == 'degree C.' or X_state_info[seatemp_var_name]['units'] == 'deg. C.' or \
                 X_state_info[seatemp_var_name]['units'] == 'degC' :
                pass
            else:
                raise SystemExit('ERROR in BayesRegMgcaPSM: unrecognized definition of units for prior temperature.')
        else:
            raise SystemExit('ERROR in BayesRegMgcaPSM: units for prior temperature undefined.')

        # salinity
        salinity_var_name = list(self.psm_salinity_variable.keys())[0]
        xb_sss = self._get_gridpoint_data(salinity_var_name, Xb, X_state_info, X_coords)
        
        
        # Grab ocean omega and pH if we have 'em, otherwise `None` and `baymagpy` should
        # use values from the modern record.
        xb_omega = None
        if self.psm_omega_variable:
            omega_var_name = list(self.psm_omega_variable.keys())[0]
            xb_omega = self._get_gridpoint_data(omega_var_name, Xb, X_state_info, X_coords)

        xb_ph = None
        if self.psm_ph_variable:
            ph_var_name = list(self.psm_ph_variable.keys())[0]
            xb_ph = self._get_gridpoint_data(ph_var_name, Xb, X_state_info, X_coords)

        
        # calculate proxy estimates: this ensemble is from Bayesian PSM
        ye_ens = self._mcmc_predict(sst=xb_sst, sss=xb_sss, ph=xb_ph, omega=xb_omega)

        # ensemble-mean
        # for MgCa proxies, DA is performed in log space (ln) as PSM has exponential form
        ye = np.mean(np.log(ye_ens), axis=1)

        return ye


    # Define a default error model for this proxy
    @staticmethod
    def error():
        return 0.1

# Define specific classes 'foram type':'sample cleaning' combinations for Mg/Ca proxies.
# Might do better as a class factory in the future.

@class_docs_fixer
class BayesRegMgcaPooledRedPSM(BayesRegMgcaPSM):
    def __init__(self, config, proxy_obj):
        # Mg/Ca for model with pooled foram species with fully reductive sample cleaning
        super().__init__(config, proxy_obj, cleaning=1, spp='all')
        self.psm_key = 'bayesreg_mgca_pooled_red'


@class_docs_fixer
class BayesRegMgcaPooledBcpPSM(BayesRegMgcaPSM):
    def __init__(self, config, proxy_obj):
        # Mg/Ca for model with pooled foram species with Barker sample cleaning protocol
        super().__init__(config, proxy_obj, cleaning=0, spp='all')
        self.psm_key = 'bayesreg_mgca_pooled_bcp'


@class_docs_fixer
class BayesRegMgcaRuberwhiteRedPSM(BayesRegMgcaPSM):
    def __init__(self, config, proxy_obj):
        # Mg/Ca for ruber white with fully reductive sample cleaning
        super().__init__(config, proxy_obj, spp='ruber', cleaning=1)
        self.psm_key = 'bayesreg_mgca_ruberwhite_red'


@class_docs_fixer
class BayesRegMgcaRuberwhiteBcpPSM(BayesRegMgcaPSM):
    def __init__(self, config, proxy_obj):
        # Mg/Ca for ruber white with Barker sample cleaning protocol
        super().__init__(config, proxy_obj, spp='ruber', cleaning=0)
        self.psm_key = 'bayesreg_mgca_ruberwhite_bcp'


@class_docs_fixer
class BayesRegMgcaSacculiferRedPSM(BayesRegMgcaPSM):
    def __init__(self, config, proxy_obj):
        # Mg/Ca for sacculifer with fully reductive sample cleaning
        super().__init__(config, proxy_obj, spp='sacculifer', cleaning=1)
        self.psm_key = 'bayesreg_mgca_sacculifer_red'


@class_docs_fixer
class BayesRegMgcaSacculiferBcpPSM(BayesRegMgcaPSM):
    def __init__(self, config, proxy_obj):
        # Mg/Ca for sacculifer with Barker sample cleaning protocol
        super().__init__(config, proxy_obj, spp='sacculifer', cleaning=0)
        self.psm_key = 'bayesreg_mgca_sacculifer_bcp'


@class_docs_fixer
class BayesRegMgcaBulloidesRedPSM(BayesRegMgcaPSM):
    def __init__(self, config, proxy_obj):
        # Mg/Ca for bulloides with fully reductive sample cleaning
        super().__init__(config, proxy_obj, spp='bulloides', cleaning=1)
        self.psm_key = 'bayesreg_mgca_bulloides_red'


@class_docs_fixer
class BayesRegMgcaBulloidesBcpPSM(BayesRegMgcaPSM):
    def __init__(self, config, proxy_obj):
        # Mg/Ca for bulloides with Barker sample cleaning protocol
        super().__init__(config, proxy_obj, spp='bulloides', cleaning=0)
        self.psm_key = 'bayesreg_mgca_bulloides_bcp'


@class_docs_fixer
class BayesRegMgcaPachydermaRedPSM(BayesRegMgcaPSM):
    def __init__(self, config, proxy_obj):
        # Mg/Ca for pachyderma sinistral with fully reductive sample cleaning
        super().__init__(config, proxy_obj, spp='pachy', cleaning=1)
        self.psm_key = 'bayesreg_mgca_pachyderma_red'


@class_docs_fixer
class BayesRegMgcaPachydermaBcpPSM(BayesRegMgcaPSM):
    def __init__(self, config, proxy_obj):
        # Mg/Ca for pachyderma sinistral with Barker sample cleaning protocol
        super().__init__(config, proxy_obj, spp='pachy', cleaning=0)
        self.psm_key = 'bayesreg_mgca_pachyderma_bcp'


class linear_multiyearPSM(BasePSM):
    """
    Linear PSM (univariate) with slope, intercept, and R given in a text file.
    Can be used for multiyear reconstructions.
    
    The information below may not be fully fleshed-out.
        
    Attributes
    ----------
    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    elev: float
        Elevation/depth of proxy site
    slope: float
        Given by text file
    intercept: float
        Given by text file
    R: float
        Given by text file
    
    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
        
    Raises
    ------
    ...
    """

    def __init__(self, config, proxy_obj, PSMlm_data=None):
    
        self.psm_key = 'linear_multiyear'

        self.proxy = proxy_obj.type
        self.site = proxy_obj.id
        self.lat = proxy_obj.lat
        self.lon = proxy_obj.lon
        try:
            self.elev = proxy_obj.elev
        except:
            self.elev = None
        self.sensitivity = None
        self.psm_required_variables = config.psm.linear_multiyear.psm_required_variables
        self.datafile_PSM = config.psm.linear_multiyear.datafile_PSM
        self.linmult_var = config.psm.linear_multiyear.linmult_var

        # Try finding file containing PSM information and
        # assign the values to the proxy psm object
        try:
            if PSMlm_data is None:
                PSMlm_data = self._load_psm_data(config)
            self.R = PSMlm_data[(self.proxy, self.site)][0]
            self.slope = PSMlm_data[(self.proxy, self.site)][1]
            self.intercept = PSMlm_data[(self.proxy, self.site)][2]
        except (KeyError, IOError) as e:
            # No PSM info file found
            #print(e)
            #print('Cannot find PSM infor data for:' + str((self.proxy, self.site)))
            #TODO: fix this exception
            # If the above exception doesn't work, then this can be a temporary work-around:
            self.R = 999999
            self.slope = 0
            self.intercept = 0

    # TODO: Ideally prior state info and coordinates should all be in single obj
    def psm(self, Xb, X_state_info, X_coords):
        """
        Maps a given state to observations for the given proxy
        
        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)
            
        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # define state variable that should be interpolated to proxy site
        state_var = self.linmult_var
    
        if state_var not in X_state_info.keys():
            raise KeyError('Needed variable not in state vector for Ye'
                       ' calculation.')
        
        # Find row index of X for which [X_lat,X_lon] corresponds to closest
        # grid point to location of proxy site [self.lat,self.lon]

        # Calclulate distances from proxy site.
            
        # TODO: end index should already be +1, more pythonic
        var_startidx, var_endidx = X_state_info[state_var]['pos']
        ind_lon = X_state_info[state_var]['spacecoords'].index('lon')
        ind_lat = X_state_info[state_var]['spacecoords'].index('lat')
        
        stateDim = var_endidx - var_startidx + 1
        ensDim = Xb.shape[1]
        dist = np.empty(stateDim)
        dist = haversine(self.lon, self.lat,
                         X_coords[var_startidx:(var_endidx+1), ind_lon],
                         X_coords[var_startidx:(var_endidx+1), ind_lat])
                        
        # Pick value at nearest grid point
        # row index of nearest grid pt. in prior (minimum distance)
        Kind_var = np.unravel_index(dist.argmin(), dist.shape)[0] + var_startidx
        
        Ye = self.slope * np.squeeze(Xb[Kind_var, :]) + self.intercept
                                               
        return Ye

    # Define a default error model for this proxy
    @staticmethod
    def error():
        return 0.1
    
    @staticmethod
    def get_kwargs(config):
        try:
            # PSM info
            PSMlm_data = linear_multiyearPSM._load_psm_data(config)
            return {'PSMlm_data': PSMlm_data}
        except IOError as e:
            print(e)
            print('In "linear_multiyearPSM" class: Cannot find PSM calibration file: %s. Exiting!' % PSMlm_data_file)
            raise SystemExit
    
    @staticmethod
    def _load_psm_data(config):
        """Helper method for loading from dataframe"""
        
        PSMlm_data_file = config.psm.linear_multiyear.datafile_PSM
        
        if PSMlm_data_file:
            # check if file exists
            if not os.path.isfile(PSMlm_data_file):
                print('In "linear_multiyearPSM" class: Cannot find PSM calibration file: %s. Exiting!' % PSMlm_data_file)
                raise SystemExit(1)
            
            else:
                # This returns a python dictionary with entries as: {(proxy type, proxy name): Rvalue}
                PSMlmdata_dict={}
                with open(PSMlm_data_file,'r') as f:
                    linenb = 0
                    for line in f:
                        try:
                            ptype,pname,Rval,pslope,pintercept = line.split(',') # expects commas as field separator
                            pintercept = pintercept.strip('\n') # remove end-or-line chars
                            # populate dictionary entry, make sure values are floats
                            PSMlmdata_dict[(ptype,pname)] = [float(Rval), float(pslope), float(pintercept)]
                        except:
                            print('WARNING: In "linear_multiyearPSM", load_psm_data: Error reading/parsing data on line', linenb, ':', line)
                        
                        linenb +=1

        else:
            PSMlmdata_dict = {}
        
        return PSMlmdata_dict


# Mapping dict to PSM object type, this is where proxy_type/psm relations
# should be specified (I think.) - AP
_psm_classes = {'linear': LinearPSM, 'linear_TorP': LinearPSM_TorP,
                'bilinear': BilinearPSM,'h_interp': h_interpPSM,
                'bayesreg_uk37': BayesRegUK37PSM, 'bayesreg_tex86': BayesRegTEX86PSM,
                'bayesreg_d18o_pooled': BayesRegD18oPooledPSM,
                'bayesreg_d18o_pachyderma': BayesRegD18oPachydermaPSM,
                'bayesreg_d18o_incompta': BayesRegD18oPachydermaPSM,
                'bayesreg_d18o_bulloides': BayesRegD18oBulloidesPSM,
                'bayesreg_d18o_sacculifer': BayesRegD18oSacculiferPSM,
                'bayesreg_d18o_ruberwhite': BayesRegD18oRuberwhitePSM,
                'bayesreg_mgca_ruberwhite_red': BayesRegMgcaRuberwhiteRedPSM,
                'bayesreg_mgca_ruberwhite_bcp': BayesRegMgcaRuberwhiteBcpPSM,
                'bayesreg_mgca_sacculifer_red': BayesRegMgcaSacculiferRedPSM,
                'bayesreg_mgca_sacculifer_bcp': BayesRegMgcaSacculiferBcpPSM,
                'bayesreg_mgca_bulloides_red': BayesRegMgcaBulloidesRedPSM,
                'bayesreg_mgca_bulloides_bcp': BayesRegMgcaBulloidesBcpPSM,
                'bayesreg_mgca_pachyderma_red': BayesRegMgcaPachydermaRedPSM,
                'bayesreg_mgca_pachyderma_bcp': BayesRegMgcaPachydermaBcpPSM,
                'bayesreg_mgca_pooled_red' : BayesRegMgcaPooledRedPSM,
                'bayesreg_mgca_pooled_bcp' : BayesRegMgcaPooledBcpPSM,
                'linear_multiyear': linear_multiyearPSM,
                }


def get_psm_class(psm_type):
    """
    Retrieve psm class type to be instantiated.

    Parameters
    ----------
    psm_type: str
        Dict key to retrieve correct PSM class type.

    Returns
    -------
    BasePSM like:
        Class type to be instantiated and attached to a proxy.
    """
    return _psm_classes[psm_type]

