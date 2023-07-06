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


in_dir = '/scratch/benbar/Processed_Data_V3.02'
#match_file = '../Validation/Indices/index.npz'
#obs_file = '../Validation/Indices/prof_match.npz'
mfiles = (in_dir 
    + '/SSWRS_V3.02_NOC_FVCOM_NWEuropeanShelf_01dy_19930101-1200_RE.nc')

match_file = in_dir + '/Indices/index_2019.npz'
obs_file = in_dir + '/Indices/prof_match_2019.npz'

amm15_root = '/scratch/benbar/AMM15/'
amm15_out = '/scratch/benbar/Processed_Data_V3.02/'

amm15_file = amm15_out + 'grid_amm15_cmems_all_2019.npz'


with nc.Dataset(mfiles, 'r') as nc_fid:
  lat_modg = nc_fid.variables['lat'][:]
  lon_modg = nc_fid.variables['lon'][:]
  siglay_mod = nc_fid.variables['siglay'][:]
  h_mod = nc_fid.variables['h'][:]
  zeta_mod = nc_fid.variables['zeta'][:]

lon_modg[lon_modg > 180] = lon_modg[lon_modg > 180] - 360
depth_mod = -h_mod * siglay_mod


file_amm15_s = sorted(glob.glob(amm15_root + 'Salinity/*/*.nc'))
file_amm15_t = sorted(glob.glob(amm15_root + 'Temperature/*/*.nc'))

with nc.Dataset(file_amm15_t[0], 'r') as src_amm15:
  lat1 = src_amm15.variables['lat'][:]
  lon1 = src_amm15.variables['lon'][:]
  dep1 = src_amm15.variables['depth'][:]
  temp = src_amm15.variables['thetao'][0, 0, :, :] # t, d, y, x

lon1_g, lat1_g = np.meshgrid(lon1, lat1)

date_amm15_st = np.zeros((len(file_amm15_t)), dtype=object)
date_amm15_en = np.zeros((len(file_amm15_t)), dtype=object)
for i in range(len(file_amm15_t)):
  date_amm15_st[i] = dt.datetime.strptime(file_amm15_t[i].split('_')[-1], 
      'dm%Y%m%d.nc')
  date_amm15_en[i] = date_amm15_st[i]# + dt.timedelta(days=1)

temp = np.ma.masked_where(temp == -32768, temp)
mask = temp.mask


lon1_r = lon1_g[np.invert(mask)]
lat1_r = lat1_g[np.invert(mask)]
b_ll = np.array([lon1_r, lat1_r]).T


data = np.load(match_file, allow_pickle=True)
m_date = data['m_date']
m_hour = data['m_hour']
m_node = data['m_node']
o_date = data['o_date']
o_prof = data['o_prof']
data.close()

data = np.load(obs_file, allow_pickle=True)
lat_obs = data['lat_obs']
lon_obs = data['lon_obs']
prof_obs = data['prof_obs']
dep_obs = data['dep_obs']
temp_obs = data['temp_obs']
sal_obs = data['sal_obs']
date_obs = data['date_obs']
data.close()

dep_obs = np.ma.masked_where(dep_obs >= 9999, dep_obs)
m_node = m_node.astype(int)

temp_mgrid = np.ma.zeros((depth_mod.shape[0], len(o_prof))) -9999
sal_mgrid = np.ma.zeros((depth_mod.shape[0], len(o_prof))) -9999
mask_mgrid = np.ma.zeros((depth_mod.shape[0], len(o_prof))) +1

ref1 = dt.datetime(1970, 1, 1)

for i in range(len(o_prof)):
  print(i / len(o_prof) *100, '%')

  date_str = m_date[i].strftime('%Y%m%d')
  bool_ind = ((m_date[i] >= date_amm15_st) 
            & (m_date[i] < (date_amm15_en + dt.timedelta(days=1))))
  f_ind = np.nonzero(bool_ind)[0]
  #ti = m_date[i].day -1 # index of day in file
  if len(f_ind) > 1:
    f_ind = f_ind[:1]

  print(date_str, f_ind)
  if any(bool_ind):

    # Calculate index

    rough_dist = (((lon1_g - lon_obs[i]) ** 2 ) 
        + ((lat1_g - lat_obs[i]) ** 2 )) ** 0.5

    yi, xi = np.nonzero(rough_dist == np.min(rough_dist))
    if len(yi) > 1:
      yi = yi[0]
      xi = xi[0]

    if mask[yi, xi]:
      # check if land and do distance calculation to un masked points
      position = np.array([lon_obs[i], lat_obs[i]])
      dist = np.asarray([haversine.dist(pt_1[0], pt_1[1], 
                      position[0], position[1]) 
                      for pt_1 in b_ll])
      mi = np.argmin(dist)
      lon_c = lon1_r[mi]
      lat_c = lat1_r[mi]
      yi, xi = np.nonzero((lon1_g == lon_c) & (lat1_g == lat_c))

    with nc.Dataset(file_amm15_t[int(f_ind)], 'r') as src_amm15:
      time1 = (ref1 + dt.timedelta(
          seconds=int(src_amm15.variables['time'][0])))
      temp1 = src_amm15.variables['thetao'][0, :, yi, xi].squeeze() # t, d, lat, lon
    with nc.Dataset(file_amm15_s[int(f_ind)], 'r') as src_amm15:
      sal1 = src_amm15.variables['so'][0, :, yi, xi].squeeze()
      mask_sal = sal1 == -32768
    temp1 = np.ma.masked_where(mask_sal, temp1)
    sal1 = np.ma.masked_where(mask_sal, sal1)
  else:
    continue

  temp_mgrid[:, i] = np.interp(depth_mod[:, m_node[i]], dep1, temp1)
  sal_mgrid[:, i] = np.interp(depth_mod[:, m_node[i]], dep1, sal1)
  mask_mgrid[:, i] = np.interp(depth_mod[:, m_node[i]], dep1, mask_sal)


  if i % 500 == 0:
    mask_mgrid[mask_mgrid != 0] = 1
    np.savez(amm15_file, temp_mgrid=temp_mgrid, sal_mgrid=sal_mgrid, mask_mgrid=mask_mgrid)

mask_mgrid[mask_mgrid != 0] = 1

np.savez(amm15_file, temp_mgrid=temp_mgrid, sal_mgrid=sal_mgrid, mask_mgrid=mask_mgrid)


