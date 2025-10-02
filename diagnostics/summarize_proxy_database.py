"""
Module: summarize_proxy_database.py

 This script reads in the preprocessed proxy datasets stored in LMR-ready
 pandas DataFrames in pickle files and a specified PSM file and
 does two things:
  1) Produces a list of all proxy types and units.
  2) Produces individual figures of all preprocessed proxy records, 
     along with metadata.

 Notes: 
 - You'll need to make the preprocessed files (using LMR_proxy_preprocess.py)
   and a PSM file (generated with LMR_PSMbuild.py) before using this.  
 - Also, change the "data_directory" and "output_directory" to point to the
   appropriate places on your machine.
 - Only works on the LMRv2 release (merged PAGES2k phase2 + LMR-specific 
   NCDC-templated proxy records.

 author: Michael P. Erb
 date  : 4/17/2017

 Revisions:  
 - Adapted the original code to handle all records (PAGES2kv2 and NCDC) 
   considered in the LMR project.
   [R. Tardif, Univ. of Washington, May 2017]
 - Added the generation of geographical maps showing the location 
   of proxy records per proxy types considered by the data assimilation.
   [R. Tardif, Univ. of Washington, May 2017]
 - Extension to proxies considered in the "Deep Times" version of LMR
   [R. Tardif, Univ. of Washington, Sept. 2019]
 - Use of new external .yml file containing proxy definitions
   (mapping of proxy archive/measurements to proxy types used in LMR)
   [R. Tardif, Univ. of Washington, Sept. 2019]
 - Improved map plot(s) showing locations of proxy records
   and new plot showing temporal distribution of available proxy data
   [R. Tardif, Univ. of Washington, Oct. 2019]

"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, exists
from mpl_toolkits.basemap import Basemap


# ------------------------------------------------------------ #
# -------------- Begin: user-defined parameters -------------- #

# directory where data from reconstruction experiments are located 
#data_directory = "/home/disk/kalman3/rtardif/LMR/"
data_directory = "/home/disk/kalman3/rtardif/LMRpy3/"

# which proxy database? PAGES2kv2, LMRdb or NCDCdadt
#proxy_db = 'PAGES2kv1'
proxy_db = 'LMRdb'
#proxy_db = 'NCDCdadt'

# Version of the database to query (for proxy_db = LMRdb only)
dbversion = 'v1.1.0' # LMR Common Era
#dbversion = 'v0.3.0' # Deep Times

# Filter on proxy temporal resolution (range is inclusive)
temporal_resolution_range = (1,1); resolution_tag = 'annual'; year_range = (0,2000)             # LMR Common Era
#temporal_resolution_range = (1,5000); resolution_tag = 'centennial'; year_range = (-22000,2000) # Deep Times

timeaxis = 'year CE'
#timeaxis = 'kyr BP' # for Deep Times proxies

# Whether the y-axis (nb of proxy records) on the temporal plots 
# should be in log-scale or not.
logaxis_temporal_plots = False

# Directory where the output (figures) will be created.
# The directory needs to exist prior to running this module.
# ----------------------------------------------------------
#output_directory = "/home/disk/kalman3/rtardif/LMR/data/proxies/PAGES2kv1/Figs"
output_directory = "/home/disk/kalman3/rtardif/LMRpy3/data/proxies/LMRdb/Figs/summary_v1.1.0"
#output_directory = "/home/disk/kalman3/rtardif/LMRpy3/data/proxies/DADT/Figs/summary_v0.1.1"


# Swith to indicate whether you want the figure to the produced on-screen (False)
# or save in .png files (True)
save_instead_of_plot = True

# --------------  End: user-defined parameters  -------------- #
# ------------------------------------------------------------ #

if not exists(output_directory):
    print('Output directory does not exist. Creating it now.')
    os.makedirs(output_directory)

### ----------------------------------------------------------------------------
### Proxy type definitions
### ----------------------------------------------------------------------------

# file containing proxy definitions
proxy_def_file = '../proxy_definitions.yml'
fprx = open(proxy_def_file,'r')
yml_dict = yaml.load(fprx)
proxy_def = yml_dict[proxy_db]
fprx.close()

### ----------------------------------------------------------------------------
### LOAD DATA
### ----------------------------------------------------------------------------

# Load the proxy data and metadata as dataframes.

if proxy_db == 'PAGES2kv1':
    metadata = np.load(data_directory+'data/proxies/Pages2kv1_Metadata.df.pckl',allow_pickle=True)
    proxies = np.load(data_directory+'data/proxies/Pages2kv1_Proxies.df.pckl',allow_pickle=True)
    # Load an LMR PSM file.
    try:
        psms = np.load(data_directory+'PSM/PSMs_PAGES2kv1_GISTEMP.pckl',allow_pickle=True)
    except:
        psms = []

    proxy_symbols_color = {
        'Tree ring_Width'       : ('^','#66CC00'), 
        'Tree ring_Density'     : ('v','#FFCC00'), 
        'Coral_d18O'            : ('o','#FF8080'),
        'Coral_Luminescence'    : ('o','#FFB980'),
        'Ice core_d18O'         : ('d','#66FFFF'), 
        'Ice core_d2H'          : ('d','#B8E6E6'), 
        'Ice core_Accumulation' : ('d','#5CB8E6'), 
        'Lake sediment_All'     : ('s','#FF00FF'), 
        'Marine sediment_All'   : ('<','#00b7b6'), 
        'Speleothem_All'        : ('p','#996600'), 
    }
    ncols = 3
        
elif proxy_db == 'LMRdb':    
    metadata = np.load(data_directory+'data/proxies/LMRdb_'+dbversion+'_Metadata.df.pckl',allow_pickle=True)
    proxies = np.load(data_directory+'data/proxies/LMRdb_'+dbversion+'_Proxies.df.pckl',allow_pickle=True)
    # Load an LMR PSM file.
    try:
        psms = np.load(data_directory+'PSM/PSMs_LMRdb_'+dbversion+'_annual_GISTEMP.pckl',allow_pickle=True)
    except:
        psms = []

    proxy_symbols_color = {
        'Bivalve_d18O'                     : ('h','#FFFF00'),
        'Corals and Sclerosponges_Rates'   : ('o','#FFE6E6'),
        'Corals and Sclerosponges_SrCa'    : ('o','#E60000'),
        'Corals and Sclerosponges_d18O'    : ('o','#FF8080'),
        'Ice Cores_Accumulation'           : ('d','#5CB8E6'),
        'Ice Cores_MeltFeature'            : ('d','#0000FF'),
        'Ice Cores_d18O'                   : ('d','#66FFFF'),
        'Ice Cores_dD'                     : ('d','#B8E6E6'),
        'Lake Cores_Misc'                  : ('s','#FFB3FF'),
        'Lake Cores_Varve'                 : ('s','#FF00FF'),
        'Marine Cores_d18O'                : ('<','#00b7b6'),
        'Speleothems_d18O'                 : ('p','#D9B3FF'),
        'Tree Rings_WidthBreit'            : ('^','#B3FF66'),
        'Tree Rings_WidthPages2'           : ('^','#66CC00'),
        'Tree Rings_WoodDensity'           : ('v','#FFCC00'),
        'Tree Rings_Isotopes'              : ('*','#CCFFCC'),    
    }
    ncols = 3
        
elif proxy_db == 'NCDCdadt':
    metadata = pd.read_hdf(data_directory+'data/proxies/DADT_'+dbversion+'_Proxies.h5',key='meta')
    proxies = pd.read_hdf(data_directory+'data/proxies/DADT_'+dbversion+'_Proxies.h5',key='proxy')
    psms = []

    proxy_symbols_color = {
        'Marine sediments_uk37'               :('o','#4D9900'),
        'Marine sediments_tex86'              :('s','#22FD00'),
        'Marine sediments_d18o_ruberwhite'    :('^','#E60000'),
        'Marine sediments_d18o_sacculifer'    :('v','#FF9900'),
        'Marine sediments_d18o_bulloides'     :('>','#5CB8E6'),
        'Marine sediments_d18o_pachyderma'    :('<','#0000FF'),
        'Marine sediments_mgca_ruberwhite_bcp':('H','#E6CCFF'),
        'Marine sediments_mgca_ruberwhite_red':('H','#BF80FF'),        
        'Marine sediments_mgca_bulloides_bcp' :('h','#FF80FF'),
        'Marine sediments_mgca_bulloides_red' :('h','#B300B3'),
        'Marine sediments_mgca_sacculifer_bcp':('D','#00e6e6'),
        'Marine sediments_mgca_sacculifer_red':('D','#00b3b3'),
        'Marine sediments_mgca_pachyderma_bcp':('X','#FFFF00'),
        'Marine sediments_mgca_pachyderma_red':('X','#FFCC99'),
#        'Marine sediments_mgca_pooled_bcp'    :('H','#D9B3FF'), # for deep geological times
#        'Marine sediments_mgca_pooled_red'    :('h','#FF00FF'), # for deep geological times
#        'lake sediment_pollenTemp'            :('p','#FFB980'),
    }
    ncols = 3
    
# include dbversion in the name
proxy_db = '_'.join([proxy_db,dbversion])

### CALCULATIONS ---

# Count all of the different proxy types and print a list.
archive_counts = {}
archive_types = np.unique(metadata['Archive type'])
for ptype in archive_types:
    archive_counts[ptype] = np.unique(metadata['Proxy measurement'][metadata['Archive type'] == ptype],return_counts=True)

print("=================")
print(" Archive counts:")
print("=================")
for ptype in archive_types:
    for units in range(0,len(archive_counts[ptype][0])):
        print('%25s - %32s : %4d' % (ptype, archive_counts[ptype][0][units], archive_counts[ptype][1][units]))



### ----------------------------------------------------------------------------
### Maps of proxy locations, per type. 
### ----------------------------------------------------------------------------

fig = plt.figure(figsize=(11,9))
#ax  = fig.add_axes([0.1,0.1,0.8,0.8])
m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.  # GLOBAL

water = '#D3ECF8'
continents = '#F2F2F2'
        
m.drawmapboundary(fill_color=water)
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.fillcontinents(color=continents,lake_color=water)
m.drawparallels(np.arange(-80.,81.,latres),linewidth=0.5)
m.drawmeridians(np.arange(-180.,181.,lonres),linewidth=0.5)


sumnbproxies = 0
print(' ')
print('Proxy types for data assimilation -------------------')
proxy_types =  list(proxy_def.keys())
ptype_legend = []
l = []
for ptype in sorted(proxy_types):
    latslons = []

    if year_range:
        latslons.append([(metadata['Lat (N)'][i],metadata['Lon (E)'][i]) for i in range(0,len(metadata['Proxy ID'])) \
                         if metadata['Archive type'][i] == ptype.split('_')[0] and metadata['Proxy measurement'][i] in proxy_def[ptype] \
                         and metadata['Resolution (yr)'][i] >= temporal_resolution_range[0] \
                         and metadata['Resolution (yr)'][i] <= temporal_resolution_range[1] \
                         and metadata['Oldest (C.E.)'][i] <= year_range[1] \
                         and metadata['Youngest (C.E.)'][i] >= year_range[0]])
    else:
        latslons.append([(metadata['Lat (N)'][i],metadata['Lon (E)'][i]) for i in range(0,len(metadata['Proxy ID'])) \
                         if metadata['Archive type'][i] == ptype.split('_')[0] and metadata['Proxy measurement'][i] in proxy_def[ptype] \
                         and metadata['Resolution (yr)'][i] >= temporal_resolution_range[0] \
                         and metadata['Resolution (yr)'][i] <= temporal_resolution_range[1]])

    nbproxies = len(latslons[0])

    sumnbproxies = sumnbproxies + nbproxies

    if nbproxies > 0:    
        plotlist = latslons[0]

        nbunique = len(list(set(plotlist)))
        nbsamelocations = nbproxies - nbunique        
        print('%35s : %4d (same lats/lons: %3d)' %(ptype,nbproxies,nbsamelocations))

        lats = np.asarray([item[0] for item in plotlist])
        lons = np.asarray([item[1] for item in plotlist])
        
        marker = proxy_symbols_color[ptype][0]
        color_dots = proxy_symbols_color[ptype][1]       

        x, y = m(lons,lats)
        l.append(m.scatter(x,y,35,marker=marker,color=color_dots,edgecolor='black',linewidth='0.5',zorder=4))

        ptype_legend.append('%s (%d)' %(ptype,nbproxies))
        

print('%35s : %4d' %('Total',sumnbproxies))

plt.title('%s: %d records' % (proxy_db, sumnbproxies), fontweight='bold',fontsize=14)
plt.legend(l,ptype_legend,
           scatterpoints=1,
           loc='lower center', bbox_to_anchor=(0.5, -0.25),
           ncol=ncols,
           fontsize=8)

#plt.show()
if year_range:
    period_str = 'to'.join(map(str,year_range))
    fname = 'map_proxies_%s_%s.png' %(proxy_db,period_str)
else:
    period_str = ''
    fname = 'map_proxies_%s.png' %(proxy_db)
plt.savefig('%s' %(os.path.join(output_directory,fname)),bbox_inches='tight')
plt.close()


### ----------------------------------------------------------------------------
### Temporal distribution of proxies (total and per type)
### ----------------------------------------------------------------------------

# define reference time axis
if year_range is None:
    year_range = (int(np.min(metadata['Oldest (C.E.)'])),
                  int(np.max(metadata['Youngest (C.E.)'])))

if resolution_tag == 'annual':
    time_ref = np.arange(year_range[0], year_range[1]+1,1)
elif resolution_tag == 'centennial':
    time_ref = np.arange(year_range[0], year_range[1]+1,200)
else:
    print('...')

nbtimes = len(time_ref)
time_ref_resolution = time_ref[1]-time_ref[0]
counts_dict = {}
proxy_types =  sorted(list(proxy_def.keys()))
nbtypes = len(proxy_types)
nb_proxies_type  = np.zeros(shape=[nbtypes,nbtimes],dtype=int)
p = 0
for ptype in proxy_types:

    print('Determining temporal distribution of proxy data for:', ptype, '...')

    sites_years = []
    for i in range(0,len(metadata['Proxy ID'])):
        if metadata['Archive type'][i] == ptype.split('_')[0] \
           and metadata['Proxy measurement'][i] in proxy_def[ptype] \
           and metadata['Resolution (yr)'][i] >= temporal_resolution_range[0] \
           and metadata['Resolution (yr)'][i] <= temporal_resolution_range[1]:
            plot_series = proxies[metadata['Proxy ID'][i]].to_dense()
            years_valid = plot_series[plot_series.notnull()].index.values
            sites_years.append(years_valid)

    for t in range(nbtimes):
        ind = [j for j, s in enumerate(sites_years) if any((s > time_ref[t]-time_ref_resolution/2.) & (s <= time_ref[t]+time_ref_resolution/2.))]
        nb_proxies_type[p,t] = len(ind)

    counts_dict[ptype] = nb_proxies_type[p,:]
    p += 1


nb_proxies_total = np.sum(nb_proxies_type, axis=0)

p_ordered_reverse = sorted(counts_dict, key=lambda k: np.max(counts_dict[k]), reverse=True)
p_ordered = sorted(counts_dict, key=lambda k: np.max(counts_dict[k]))

counts_cumul = np.zeros(shape=[nbtimes],dtype=int)
counts_dict_cumul = {}
for p in p_ordered:
    counts_cumul = counts_cumul + counts_dict[p]
    counts_dict_cumul[p] = counts_cumul

    
# --- the plot ---

p_plot_order = sorted(counts_dict_cumul, key=lambda k: np.max(counts_dict_cumul[k]), reverse=True)

if timeaxis == 'kyr BP':
    plot_time = -1.0*((-1.0*time_ref[:] + 1950.)/1000.)
    plot_year_range = -1.0*((-1.0*np.array(year_range) + 1950.)/1000.)
    tick_inter = 2000.
else:
    plot_time = time_ref
    plot_year_range = year_range
    tick_inter = 200.


fig, ax = plt.subplots()

if logaxis_temporal_plots:
    p1 = ax.semilogy(plot_time,counts_cumul,color='#000000',linewidth=2,label='Total')
    minval = 1
else:
    p1 = ax.plot(plot_time,counts_cumul,color='#000000',linewidth=2,label='Total')
    minval = 0
    
plt.xlim((plot_time[0],plot_time[-1]))

for p in p_plot_order:
    if any(c > 0 for c in counts_dict_cumul[p]):
        #plt.fill_between(plot_time,minval,counts_dict[p],color=proxy_symbols_color[p][1],linewidth=2,label=p)
        plt.fill_between(plot_time,minval,counts_dict_cumul[p],color=proxy_symbols_color[p][1],linewidth=2,label=p)

plt.xlabel('Time (%s)'%timeaxis,fontsize=14, fontweight='bold')
plt.ylabel('Cumulative number of proxies',fontsize=14,fontweight='bold')
plt.legend(loc='upper left',bbox_to_anchor=(1., 1.),ncol=1,numpoints=1,fontsize=8,frameon=False)
plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.85])

# re-define time axis if needed
if timeaxis == 'kyr BP': 
    xlabels = ax.get_xticks()
    indnonzero = np.where(xlabels != 0.0)        
    xlabels[indnonzero] = -1.0*xlabels[indnonzero]
    ax.set_xticklabels(xlabels)
        
ax.minorticks_on()

#plt.show()
period_str = 'to'.join(map(str,year_range))
plt.savefig('%s/temporal_distribution_proxies_%s_%s.png' %(output_directory,proxy_db,period_str),bbox_inches='tight')
plt.close()


### ----------------------------------------------------------------------------
### FIGURES of individual records (time series)
### ----------------------------------------------------------------------------

# Save a list of all records with PSMs.
records_with_psms = []
for i in range(0,len(psms)):
    records_with_psms.append(list(psms.keys())[i][1])


plt.style.use('ggplot')

#for i in range(0,3):  # To make sample figures, use this line instead of the next line.
for i in range(0,len(metadata['Proxy ID'])):
    print("Proxy: ",i+1,"/",len(metadata['Proxy ID']), metadata['Proxy ID'][i])
    if metadata['Proxy ID'][i] in records_with_psms: has_psm = "YES"
    else: has_psm = "NO"
    
    # Make a plot of each proxy.
    plt.figure(figsize=(10,8))
    ax = plt.axes([.1,.6,.8,.3])
    if pd.api.types.is_sparse(proxies[metadata['Proxy ID'][i]]):
        plot_series = proxies[metadata['Proxy ID'][i]].to_dense()
    else:
        plot_series = proxies[metadata['Proxy ID'][i]]
    plt.plot(plot_series,'-b',linewidth=2,alpha=.3)
    plt.plot(plot_series,'.',color='b')
    
    plt.suptitle("LMR proxy time series",fontweight='bold',fontsize=12)
    plt.title(metadata['Proxy ID'][i],fontsize=11)
    plt.xlabel("Year CE")
    plt.ylabel(metadata['Proxy measurement'][i])

    fntsize = 9
    offsetscale = 0.07
    
    # Print metadata on each figure.
    for offset, key in enumerate(metadata):
        if key != 'Proxy ID':
            plt.text(0,-.3-offsetscale*offset,key+":",transform=ax.transAxes,fontsize=fntsize)
            if key == 'Study name' or key == 'Investigators':
                if len(metadata[key][i]) > 100:
                    metadata_entry = metadata[key][i][0:100]+' ...'
                else:
                    metadata_entry = metadata[key][i]
            else:
                metadata_entry = metadata[key][i]

            if isinstance(metadata_entry,str):
                try:
                    metadata_entry = metadata_entry.encode(encoding='utf-8', errors='ignore')
                    trc = 1
                except (UnicodeEncodeError, UnicodeDecodeError):
                    try:
                        metadata_entry = metadata_entry.decode('utf-8','ignore')                        
                        trc = 2
                    except UnicodeDecodeError:
                        metadata_entry = metadata_entry.decode('iso-8859-1')
                        trc = 3

                if isinstance(metadata_entry, bytes):
                    metadata_entry = metadata_entry.decode()
                metadata_entry.encode('ascii', 'ignore')
                
            plt.text(.23,-.3-offsetscale*offset,metadata_entry,transform=ax.transAxes,fontsize=fntsize)
            
    plt.text(0,-.4-offsetscale*offset,"Proxy is in given PSM file:",transform=ax.transAxes,fontsize=fntsize)
    plt.text(.23,-.4-offsetscale*offset,has_psm,transform=ax.transAxes,fontsize=fntsize)
    
    if save_instead_of_plot:
        plt.savefig(output_directory+'/ts_'+metadata['Archive type'][i].replace(" ","_")+"_"+metadata['Proxy ID'][i].replace("/","_")+".png")
    else:
        plt.show()
    plt.close()
