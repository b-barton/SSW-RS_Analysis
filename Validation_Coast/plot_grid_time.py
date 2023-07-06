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

in_dir = '/scratch/benbar/Processed_Data_V3.02'

grid_obs_file = in_dir + '/Indices/grid_match_full.npz'
mod_file = in_dir + '/Indices/grid_model_full.npz'
amm7_file = in_dir + '/grid_amm7_all_full.npz'



data = np.load(grid_obs_file, allow_pickle=True)
o_prof = data['o_prof'] 
lat_obs = data['lat_obs']
lon_obs = data['lon_obs']
dep_grid = data['dep_grid']
temp_grid = data['temp_grid']
sal_grid = data['sal_grid']
o_date = data['o_date']
mask_t_grid = data['mask_t_grid']
mask_s_grid = data['mask_s_grid']
data.close()

print(np.min(sal_grid), np.max(sal_grid))
#mask = (sal_grid < 0) 
temp_grid = np.ma.masked_where(mask_t_grid == 1, temp_grid)
sal_grid = np.ma.masked_where(mask_s_grid == 1, sal_grid)
dep_grid = np.ma.masked_where(mask_t_grid == 1, dep_grid)


data = np.load(mod_file)
temp_mgrid = data['temp_mgrid']
sal_mgrid = data['sal_mgrid']
h_mgrid = data['h_mgrid']
data.close()

h_mgrid = np.ma.masked_where(h_mgrid == -9999, h_mgrid)
temp_mgrid = np.ma.masked_where((temp_mgrid == -9999) | (mask_t_grid == 1), temp_mgrid)
sal_mgrid = np.ma.masked_where((sal_mgrid == -9999) | (mask_s_grid == 1), sal_mgrid)

# Load AMM7

data = np.load(amm7_file)
temp_amm7 = data['temp_mgrid']
sal_amm7 = data['sal_mgrid']
mask_amm7 = data['mask_mgrid']
data.close()

print(mask_amm7.shape, np.min(mask_amm7), np.max(mask_amm7))
#mask_amm7 = sal_amm7 < 0
temp_amm7 = np.ma.masked_where((temp_amm7 <= -9999) | (mask_t_grid == 1) | (mask_amm7 == 1), temp_amm7)
sal_amm7 = np.ma.masked_where((sal_amm7 <= -9999) | (mask_s_grid == 1) | (mask_amm7 == 1), sal_amm7)


for i in range(temp_mgrid.shape[0]):
  temp_mgrid[i, :] = np.ma.masked_where(h_mgrid > 200, temp_mgrid[i, :])
  sal_mgrid[i, :] = np.ma.masked_where(h_mgrid > 200, sal_mgrid[i, :])
  temp_grid[i, :] = np.ma.masked_where(h_mgrid > 200, temp_grid[i, :])
  sal_grid[i, :] = np.ma.masked_where(h_mgrid > 200, sal_grid[i, :])
  temp_amm7[i, :] = np.ma.masked_where(h_mgrid > 200, temp_amm7[i, :])
  sal_amm7[i, :] = np.ma.masked_where(h_mgrid > 200, sal_amm7[i, :])


print(np.ma.min(temp_amm7))


# sort data based on date

#ref = dt.datetime(1990, 1, 1)
#o_days = np.zeros((len(date_obs)))
#ddays = np.zeros((len(dtmp)))
#for i in range(len(date_obs)):
#  tmp = date_obs[i] - ref
#  o_days[i] = tmp.days + (tmp.seconds / (24 * 60 * 60))
#for i in range(len(dtmp)):
#  tmp = dtmp[i] - ref
#  ddays[i] = tmp.days + (tmp.seconds / (24 * 60 * 60))

# sort based on date, profile and depth
#o_sort_ind = np.argsort(o_days)
#sort_ind = np.argsort(ddays)

#date_obs = date_obs[o_sort_ind]
#o_prof = o_prof[o_sort_ind]
#lat_obs = lat_obs[o_sort_ind]
#lon_obs = lon_obs[o_sort_ind]

#prof_obs = prof_obs[sort_ind]
#dep_obs = dep_obs[sort_ind]
#temp_obs = temp_obs[sort_ind]
#sal_obs = sal_obs[sort_ind]
#temp_mod = temp_mod[sort_ind]
#sal_mod = sal_mod[sort_ind]
#temp_amm7 = temp_amm7[sort_ind]
#sal_amm7 = sal_amm7[sort_ind]

temp_grid1 = np.ma.min(temp_grid, axis=0)
temp_grid2 = np.ma.max(temp_grid, axis=0)
temp_mgrid1 = np.ma.min(temp_mgrid, axis=0)
temp_mgrid2 = np.ma.max(temp_mgrid, axis=0)
temp_amm71 = np.ma.min(temp_amm7, axis=0)
temp_amm72 = np.ma.max(temp_amm7, axis=0)

sal_grid1 = np.ma.min(sal_grid, axis=0)
sal_grid2 = np.ma.max(sal_grid, axis=0)
sal_mgrid1 = np.ma.min(sal_mgrid, axis=0)
sal_mgrid2 = np.ma.max(sal_mgrid, axis=0)
sal_amm71 = np.ma.min(sal_amm7, axis=0)
sal_amm72 = np.ma.max(sal_amm7, axis=0)

temp_grid_s = np.stack((temp_grid1, temp_grid2), axis=1)
temp_mgrid_s = np.stack((temp_mgrid1, temp_mgrid2), axis=1)
temp_amm7_s = np.stack((temp_amm71, temp_amm72), axis=1)
sal_grid_s = np.stack((sal_grid1, sal_grid2), axis=1)
sal_mgrid_s = np.stack((sal_mgrid1, sal_mgrid2), axis=1)
sal_amm7_s = np.stack((sal_amm71, sal_amm72), axis=1)
o_date_s = np.stack((o_date, o_date), axis=1)
print(temp_grid_s.shape)

#Plot

fig1 = plt.figure(figsize=(8,12))

ax1 = fig1.add_axes([0.08, 0.82, 0.86, 0.13])
ax2 = fig1.add_axes([0.08, 0.67, 0.86, 0.13])
ax3 = fig1.add_axes([0.08, 0.52, 0.86, 0.13])

ax4 = fig1.add_axes([0.08, 0.35, 0.86, 0.13])
ax5 = fig1.add_axes([0.08, 0.2, 0.86, 0.13])
ax6 = fig1.add_axes([0.08, 0.05, 0.86, 0.13])

#fig2 = plt.figure(figsize=(10,8))

#ax4 = fig2.add_axes([0.1, 0.7, 0.8, 0.25])
#ax5 = fig2.add_axes([0.1, 0.4, 0.8, 0.25])
#ax6 = fig2.add_axes([0.1, 0.1, 0.8, 0.25])

tnow = dt.datetime.today()

ntemp = 0
nsal = 0
interval = 10
if 0:
  for i in range(0, len(o_prof), interval):
    print(i / len(o_prof) *100, '%')
    print(dt.datetime.today() - tnow)

    if i == 0:
      ax1.plot(o_date_s[i, :], temp_grid_s[i, :], 'k', label='Observations')
      ax2.plot(o_date_s[i, :], temp_mgrid_s[i, :], 'r', label='SSW-RS')
      ax3.plot(o_date_s[i, :], temp_amm7_s[i, :], 'g', label='AMM7')
    else:
      ax1.plot(o_date_s[i, :], temp_grid_s[i, :], 'k')
      ax2.plot(o_date_s[i, :], temp_mgrid_s[i, :], 'r')
      ax3.plot(o_date_s[i, :], temp_amm7_s[i, :], 'g')

    if i == 0:
      ax4.plot(o_date_s[i, :], sal_grid_s[i, :], 'k', label='Observations')
      ax5.plot(o_date_s[i, :], sal_mgrid_s[i, :], 'r', label='SSW-RS')
      ax6.plot(o_date_s[i, :], sal_amm7_s[i, :], 'g', label='AMM7')
    else:
      ax4.plot(o_date_s[i, :], sal_grid_s[i, :], 'k')
      ax5.plot(o_date_s[i, :], sal_mgrid_s[i, :], 'r')
      ax6.plot(o_date_s[i, :], sal_amm7_s[i, :], 'g')
else:
  ax1.scatter(o_date, temp_grid1, s=0.5)
  ax2.scatter(o_date, temp_mgrid1, s=0.5)
  ax3.scatter(o_date, temp_amm71, s=0.5)
  ax1.scatter(o_date, temp_grid2, s=0.5)
  ax2.scatter(o_date, temp_mgrid2, s=0.5)
  ax3.scatter(o_date, temp_amm72, s=0.5)
  ax4.scatter(o_date, sal_grid1, s=0.5)
  ax5.scatter(o_date, sal_mgrid1, s=0.5)
  ax6.scatter(o_date, sal_amm71, s=0.5)
  ax4.scatter(o_date, sal_grid2, s=0.5)
  ax5.scatter(o_date, sal_mgrid2, s=0.5)
  ax6.scatter(o_date, sal_amm72, s=0.5)

ntemp = np.ma.count(temp_grid_s[:i:interval, 0])
nsal = np.ma.count(sal_grid_s[:i:interval, 0])
print(ntemp, nsal)

ax1.legend(loc='lower right')
ax2.legend(loc='lower right')
ax3.legend(loc='lower right')
ax4.legend(loc='lower right')
ax5.legend(loc='lower right')
ax6.legend(loc='lower right')

ax1.set_ylabel('Temperature ($^{\circ}$C)')
ax2.set_ylabel('Temperature ($^{\circ}$C)')
ax3.set_ylabel('Temperature ($^{\circ}$C)')
ax4.set_ylabel('Salinity')
ax5.set_ylabel('Salinity')
ax6.set_ylabel('Salinity')

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax4.set_xticklabels([])
ax5.set_xticklabels([])

ax1.set_ylim([0, 25])
ax2.set_ylim([0, 25])
ax3.set_ylim([0, 25])
ax4.set_ylim([10, 37])
ax5.set_ylim([10, 37])
ax6.set_ylim([10, 37])


ax1.annotate('(a)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax4.annotate('(d)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax5.annotate('(e)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax6.annotate('(f)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

with open('./number_profiles.txt', 'w') as f:
  f.write('{:.0f} Temp. Profiles'.format(ntemp) + '\n')
  f.write('{:.0f} Sal. Profiles'.format(nsal) + '\n')


fig1.savefig('./Figures/all_grid_time.png')
#fig2.savefig('./Figures/all_grid_time_sal.png')

