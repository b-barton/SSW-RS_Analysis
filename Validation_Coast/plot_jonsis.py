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


match_file = './Indices/index_jonsis.npz'
obs_file = './Indices/jonsis_match.npz'
mod_file = './Indices/jonsis_model.npz'

data = np.load(match_file, allow_pickle=True)
m_date = data['m_date']
m_hour = data['m_hour']
m_node = data['m_node']
m_sig = data['m_sig']
o_date = data['o_date']
o_loc = data['o_loc'] # location of profile
o_index = data['o_index'] # date index of profile
dep_obs = data['dep_obs']
 
data.close()


data = np.load(obs_file, allow_pickle=True)
lat_obs = data['o_lat']
lon_obs = data['o_lon']
dep_obs = data['dep_obs']
temp_obs = data['temp_obs']
sal_obs = data['sal_obs']
date_obs = data['o_date']
data.close()

temp_obs = np.ma.masked_where((temp_obs == -9999) | (sal_obs == 0), temp_obs)
sal_obs = np.ma.masked_where((sal_obs == -9999) | (sal_obs == 0), sal_obs)

data = np.load(mod_file, allow_pickle=True)
temp_mod = data['temp_mod']
sal_mod = data['sal_mod']
data.close()

temp_mod = np.ma.masked_where((temp_mod == -9999) | (sal_mod == 0), temp_mod)
sal_mod = np.ma.masked_where((sal_mod == -9999) | (sal_mod == 0), sal_mod)
temp_obs = np.ma.masked_where(temp_mod.mask, temp_obs)
sal_obs = np.ma.masked_where(sal_mod.mask, sal_obs)


fig1 = plt.figure(figsize=(10,8))

ax1 = fig1.add_axes([0.1, 0.55, 0.8, 0.35])
ax2 = fig1.add_axes([0.1, 0.1, 0.8, 0.35])

n_loc = np.unique(o_loc)

for i in range(len(n_loc)):
  ind1 = (o_loc == n_loc[i]) & (dep_obs == 10)
  ind2 = (o_loc == n_loc[i]) & (dep_obs == 30)
  if i == 0:
    ax1.plot(date_obs[ind1], temp_obs[ind1], '.k', label='Obs.')
    ax1.plot(date_obs[ind1], temp_mod[ind1], '.r', label='Model')
  else:
    ax1.plot(date_obs[ind1], temp_obs[ind1], '.k')
    ax1.plot(date_obs[ind1], temp_mod[ind1], '.r')

  ax2.plot(date_obs[ind1], sal_obs[ind1], '.k')
  ax2.plot(date_obs[ind1], sal_mod[ind1], '.r')

ax1.legend()

ax1.set_xlim([dt.datetime(1993, 1, 1), dt.datetime(2003, 1, 1)])
ax2.set_xlim([dt.datetime(1993, 1, 1), dt.datetime(2003, 1, 1)])

ax1.set_ylabel('Temperature ($^{\circ}$C)')
ax2.set_ylabel('Salinity')



fig2 = plt.figure(figsize=(10,8))

ax1 = fig2.add_axes([0.1, 0.55, 0.8, 0.35])
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.35])

ax1.plot(sal_obs, temp_obs, '.k')
ax2.plot(sal_mod, temp_mod, '.r')

ax1.set_ylabel('Temperature ($^{\circ}$C)')
ax1.set_ylabel('Temperature ($^{\circ}$C)')
ax1.set_xlabel('Salinity')
#ax1.legend()


fig1.savefig('./Figures/jonsis_time.png')
fig2.savefig('./Figures/jonsis_ts.png')

