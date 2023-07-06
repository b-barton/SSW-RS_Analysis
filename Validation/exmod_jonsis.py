#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PyFVCOM.read import MFileReader
from PyFVCOM.plot import Time
from PyFVCOM.grid import unstructured_grid_depths
import PyFVCOM.plot as fvplot
import glob
import netCDF4 as nc
import datetime as dt
import properscoring as ps
from mpl_toolkits.basemap import Basemap

# NOTE mount JASMIN first

# Model folder

mname = '/scratch/Scottish_waters_FVCOM/SSW_RS/'
mfiles = sorted(glob.glob(mname + 'SSW_Reanalysis_v1.1*/SSWRS*hr*RE.nc'))

in_dir = '/scratch/benbar/Processed_Data/'
match_file = in_dir + 'Indices/index_jonsis.npz'
obs_file = in_dir + 'Indices/jonsis_match.npz'
mod_file = in_dir + 'Indices/jonsis_model.npz'

mjas = '/scratch/benbar/JASMIN/Model_Output/Hourly/'


with nc.Dataset(mfiles[0], 'r') as nc_fid:
  lat_modg = nc_fid.variables['lat'][:]
  lon_modg = nc_fid.variables['lon'][:]
  siglay_mod = nc_fid.variables['siglay'][:]
  h_mod = nc_fid.variables['h'][:]
  zeta_mod = nc_fid.variables['zeta'][:]

lon_modg[lon_modg > 180] = lon_modg[lon_modg > 180] - 360


data = np.load(match_file, allow_pickle=True)
m_date = data['m_date']
m_hour = data['m_hour']
m_node = data['m_node']
m_sig = data['m_sig']
o_date = data['o_date']
o_loc = data['o_loc']
o_index = data['o_index']
dep_obs = data['dep_obs']
 
data.close()


temp_mod = np.ma.zeros(dep_obs.shape) -9999
sal_mod = np.ma.zeros(dep_obs.shape) -9999

for i in range(len(m_date)):
  date_str = m_date[i].strftime('%Y%m%d')
  fn = sorted(glob.glob(mjas + '*/SSWRS*' 
      + date_str + '*RE.nc'))

  if len(fn):
    with nc.Dataset(fn[-1], 'r') as nc_fid:
      temp_mod[i] = nc_fid.variables['temp'][m_hour[i], m_sig[i], m_node[i]]
      sal_mod[i] = nc_fid.variables['salinity'][m_hour[i], m_sig[i], m_node[i]]

temp_mod[dep_obs == -9999] = -9999
sal_mod[dep_obs == -9999] = -9999


np.savez(mod_file, temp_mod=temp_mod, sal_mod=sal_mod)



