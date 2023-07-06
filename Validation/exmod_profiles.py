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
mod_file = in_dir + '/Indices/profile_model_full.npz'
obs_file = in_dir + '/Indices/prof_match_full.npz'
mjas = '/scratch/benbar/JASMIN/Model_Output_V3.02/Hourly/'

#fn = sorted(glob.glob(mjas + '*/SSWRS*hr*RE.nc'))
fn = []
#for yr in range(1993, 1995):
#  fn.extend(sorted(glob.glob(mjas + '*/SSWRS*V1_*hr*' + str(yr) + '*RE.nc')))
#fn = fn[:-10]
#print(fn[-1], len(fn))
for yr in range(1993, 2019):
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

temp_mod = np.ma.zeros(dep_obs.shape) -9999
sal_mod = np.ma.zeros(dep_obs.shape) -9999
h_mod = np.ma.zeros(dep_obs.shape) -9999

load_saved = 0
st_ind = 0
if load_saved:
  data = np.load(mod_file)
  temp_t = data['temp_mod']
  sal_t = data['sal_mod']
  h_t = data['h_mod']
  nload = len(temp_t)

  temp_mod[:nload] = temp_t
  sal_mod[:nload] = sal_t
  h_mod[:nload] = h_t
  st_ind = int(prof_obs[nload])
  print(st_ind)


date_fn = np.zeros((len(fn)), dtype='<U8')
for i in range(len(fn)):
  date_fn[i] = fn[i].split('_')[-2].split('-')[0]


# Fix o_prof and prof_obs
if 0:
  pind = 0
  sti = 0
  for i in range(len(dep_obs) -1):
    if (prof_obs[i] != prof_obs[i + 1]):
      print(i / len(dep_obs) *100, '%')
      o_prof[pind] = pind
      prof_obs[sti:i + 1] = pind
      sti = i
      pind = pind + 1

  o_prof[pind] = pind
  prof_obs[sti:i + 1] = pind
print(len(o_prof), o_prof[-1])

f_sv = -1
nc_fid = None

for i in range(st_ind, len(o_prof)):
  print(i / len(o_prof) *100, '%')
  ind = prof_obs == o_prof[i]

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

    temp_p = (nc_fid.variables['temp'][m_hour[i], :, m_node[i]])
    sal_p = (nc_fid.variables['salinity'][m_hour[i], :, m_node[i]])
    h_p = (nc_fid.variables['h'][m_node[i]])

  else:
    continue

  temp_mod[ind] = np.interp(dep_obs[ind], depth_mod[:, m_node[i]], temp_p)
  sal_mod[ind] = np.interp(dep_obs[ind], depth_mod[:, m_node[i]], sal_p)
  h_mod[ind] = h_p * 1

  #plt.plot(temp_p, depth_mod[:, m_node[i]])
  #plt.plot(temp_mod[ind], dep_obs[ind])
  #plt.show()

nc_fid.close()

temp_mod[dep_obs == -9999] = -9999
sal_mod[dep_obs == -9999] = -9999
h_mod[dep_obs == -9999] = -9999

np.savez(mod_file, temp_mod=temp_mod, sal_mod=sal_mod, h_mod=h_mod)



