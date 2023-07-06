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
import haversine


# Model folder


in_dir = '/scratch/benbar/Processed_Data_V3.02'
mfiles = (in_dir 
    + '/SSWRS_V3.02_NOC_FVCOM_NWEuropeanShelf_01dy_19930101-1200_RE.nc')

match_file = in_dir + '/Indices/index_full.npz'
obs_file = in_dir + '/Indices/prof_match_full.npz'

grid_obs_file = in_dir + '/Indices/grid_match_full.npz'

with nc.Dataset(mfiles, 'r') as nc_fid:
  lat_modg = nc_fid.variables['lat'][:]
  lon_modg = nc_fid.variables['lon'][:]
  siglay_mod = nc_fid.variables['siglay'][:]
  h_mod = nc_fid.variables['h'][:]
  zeta_mod = nc_fid.variables['zeta'][:]

lon_modg[lon_modg > 180] = lon_modg[lon_modg > 180] - 360

x1 = np.min(lon_modg) - 0.2
x2 = np.max(lon_modg) + 0.2
y1 = np.min(lat_modg) - 0.2
y2 = np.max(lat_modg) + 0.2

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

file_checked = data['file_checked']
data.close()

file_checked = file_checked.tolist()

data = np.load(obs_file, allow_pickle=True)
lat_obs = data['lat_obs']
lon_obs = data['lon_obs']
prof_obs = data['prof_obs']
dep_obs = data['dep_obs']
temp_obs = data['temp_obs']
sal_obs = data['sal_obs']
date_obs = data['date_obs']
data.close()

temp_obs = np.ma.masked_where(temp_obs == -9999, temp_obs)
sal_obs = np.ma.masked_where(sal_obs == -9999, sal_obs)
dep_obs = np.ma.masked_where(dep_obs >= 9999, dep_obs)

pind = np.ma.max(prof_obs)

# Reorganise data into a grid with obs interpolated onto model depths
# dep_obs, temp_obs, sal_obs and prof_obs
print(depth_mod.shape, len(o_prof))

temp_grid = np.ma.zeros((depth_mod.shape[0], len(o_prof)))
sal_grid = np.ma.zeros((depth_mod.shape[0], len(o_prof)))
mask_t_grid = np.ma.zeros((depth_mod.shape[0], len(o_prof)))
mask_s_grid = np.ma.zeros((depth_mod.shape[0], len(o_prof)))
dep_grid = np.ma.zeros((depth_mod.shape[0], len(o_prof)))
mask_t = temp_obs.mask.astype(int)
mask_s = sal_obs.mask.astype(int)

st_ind = 0

load_saved = 1
if load_saved:
  data = np.load(grid_obs_file, allow_pickle=True)
  temp_grid_t = data['temp_grid']
  sal_grid_t = data['sal_grid']
  mask_t_grid_t = data['mask_t_grid']
  mask_s_grid_t = data['mask_s_grid']
  dep_grid_t = data['dep_grid']
  data.close()

  nload = temp_grid_t.shape[1]
  temp_grid[:, :nload] = temp_grid_t
  sal_grid[:, :nload] = sal_grid_t
  mask_t_grid[:, :nload] = mask_t_grid_t
  mask_s_grid[:, :nload] = mask_s_grid_t
  dep_grid[:, :nload] = dep_grid_t
  st_ind = nload * 1



for i in range(st_ind, len(o_prof)):
  print(i / len(o_prof) *100, '%', o_date[i])
  ind = prof_obs == o_prof[i]

  temp_grid[:, i] = np.interp(
      depth_mod[:, m_node[i]], dep_obs[ind], temp_obs[ind])
  sal_grid[:, i] = np.interp(
      depth_mod[:, m_node[i]], dep_obs[ind], sal_obs[ind])
  mask_t_grid[:, i] = np.interp(
      depth_mod[:, m_node[i]], dep_obs[ind], mask_t[ind])
  mask_s_grid[:, i] = np.interp(
      depth_mod[:, m_node[i]], dep_obs[ind], mask_s[ind])
  dep_grid[:, i] = depth_mod[:, m_node[i]]

mask_t_grid[mask_t_grid != 0] = 1
mask_s_grid[mask_s_grid != 0] = 1

np.savez(grid_obs_file, lat_obs=lat_obs, lon_obs=lon_obs, o_prof=o_prof, o_date=o_date, dep_grid=dep_grid, temp_grid=temp_grid, sal_grid=sal_grid, mask_t_grid=mask_t_grid, mask_s_grid=mask_s_grid)


