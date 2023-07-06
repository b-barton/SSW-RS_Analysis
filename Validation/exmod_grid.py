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
in_dir = '/scratch/benbar/Processed_Data_V3.02'
#mfiles = sorted(glob.glob(mname + 'SSW_Reanalysis_v1.1*/SSWRS*hr*RE.nc'))
match_file = in_dir + '/Indices/index_full.npz'
mod_file = in_dir + '/Indices/grid_model_full.npz'
obs_file = in_dir + '/Indices/prof_match_full.npz'
mjas = '/scratch/benbar/JASMIN/Model_Output_V3.02/Hourly/'

#fn = sorted(glob.glob(mjas + '*/SSWRS*hr*RE.nc'))
fn = []
#for yr in range(1993, 1995):
#  fn.extend(sorted(glob.glob(mjas + '*/SSWRS*V1_*hr*' + str(yr) + '*RE.nc')))
#fn = fn[:-10]
#print(fn[-1], len(fn))
for yr in range(1993, 2020):
  fn.extend(sorted(glob.glob(mjas + str(yr) + '*/SSWRS*V3.02*hr*RE.nc')))


with nc.Dataset(fn[0], 'r') as nc_fid:
  lat_modg = nc_fid.variables['lat'][:]
  lon_modg = nc_fid.variables['lon'][:]
  siglay_mod = nc_fid.variables['siglay'][:]
  h_mod = nc_fid.variables['h'][:]
  zeta_mod = nc_fid.variables['zeta'][:]

lon_modg[lon_modg > 180] = lon_modg[lon_modg > 180] - 360

depth_mod = -h_mod * siglay_mod
print(np.min(depth_mod), np.max(depth_mod))

data = np.load(match_file, allow_pickle=True)
m_date = data['m_date']
m_hour = data['m_hour']
m_node = data['m_node']
m_sig = data['m_sig']

o_date = data['o_date']
o_prof = data['o_prof'] 
dep_obs = data['dep_obs']
data.close()

dep_obs = np.ma.masked_where(dep_obs >= 9999, dep_obs)
m_node = m_node.astype(int)

data = np.load(obs_file, allow_pickle=True)
prof_obs = data['prof_obs'] 
data.close()

temp_mgrid = np.ma.zeros((depth_mod.shape[0], len(o_prof))) -9999
sal_mgrid = np.ma.zeros((depth_mod.shape[0], len(o_prof))) -9999
h_mgrid = np.ma.zeros((len(o_prof))) -9999
#dep_mgrid = np.ma.zeros((depth_mod.shape[0], len(o_prof))) -9999


load_saved = 1
st_ind = 0
if load_saved:
  data = np.load(mod_file)
  temp_t = data['temp_mgrid']
  sal_t = data['sal_mgrid']
  h_t = data['h_mgrid']
  nload = temp_t.shape[1]

  temp_mgrid[:, :nload] = temp_t
  sal_mgrid[:, :nload] = sal_t
  h_mgrid[:nload] = h_t
  st_ind = int(o_prof[nload])
  print(nload, st_ind)


date_fn = np.zeros((len(fn)), dtype='<U8')
for i in range(len(fn)):
  date_fn[i] = fn[i].split('_')[-2].split('-')[0]


f_sv = -1
nc_fid = None

for i in range(st_ind, len(o_prof)):
  print(i / len(o_prof) *100, '%')
  #ind = prof_obs == o_prof[i]

  date_str = m_date[i].strftime('%Y%m%d')
  bool_ind = date_str == date_fn
  f_ind = np.nonzero(bool_ind)[0]
  print(date_str, f_ind)
  if any(bool_ind):
    #with nc.Dataset(fn[int(f_ind)], 'r') as nc_fid:
    if f_sv != f_ind:
      f_sv = f_ind * 1
      if nc_fid != None:
        nc_fid.close()
      nc_fid = nc.Dataset(fn[int(f_ind)], 'r')

    temp_mgrid[:, i] = (nc_fid.variables['temp'][m_hour[i], :, m_node[i]])
    sal_mgrid[:, i] = (nc_fid.variables['salinity'][m_hour[i], :, m_node[i]])
    h_mgrid[i] = (nc_fid.variables['h'][m_node[i]])

  else:
    continue

nc_fid.close()


np.savez(mod_file, temp_mgrid=temp_mgrid, sal_mgrid=sal_mgrid, h_mgrid=h_mgrid)



