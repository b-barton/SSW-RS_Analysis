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
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import gsw as sw


in_dir = '/scratch/benbar/Processed_Data_V3.02'

grid_obs_file = in_dir + '/Indices/grid_match_2019.npz'
match_file = in_dir + '/Indices/index_2019.npz'

mod_file = in_dir + '/Indices/grid_model_2019.npz'
amm7_file = in_dir + '/grid_amm7_all_2019.npz'
amm15_file = in_dir + '/grid_amm15_cmems_all_2019.npz'

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

data = np.load(match_file, allow_pickle=True)
m_node = data['m_node']
data.close()
print(m_node.shape, lat_obs.shape)

print(np.min(temp_grid), np.max(temp_grid))
print(np.min(sal_grid), np.max(sal_grid))
mask_s_grid = (mask_s_grid == 1) | (sal_grid < 1) | (mask_t_grid == 1)
temp_grid = np.ma.masked_where(mask_t_grid == 1, temp_grid)
sal_grid = np.ma.masked_where(mask_s_grid == 1, sal_grid)
dep_grid = np.ma.masked_where(mask_t_grid == 1, dep_grid)

def sa_from_sp(sal, dep, lat, lon):
  p = sw.p_from_z(dep, lat)
  return sw.SA_from_SP(sal ,p, lon, lat)

sal_grid1 = sa_from_sp(sal_grid, dep_grid, 56, 0)
sal_mask = np.isnan(sal_grid1) | np.isinf(sal_grid1) | (sal_grid1 > 40)
sal_grid1[sal_mask] = 0
sal_grid = np.ma.masked_where(sal_grid.mask | sal_mask, sal_grid1)

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
mask_amm7 = (mask_amm7 == 1) | (sal_amm7 < 0) | (temp_amm7 <= -1)
temp_amm7 = np.ma.masked_where((mask_amm7 == 1) | (mask_t_grid == 1), temp_amm7)
sal_amm7 = np.ma.masked_where((mask_amm7 == 1) | (mask_s_grid == 1), sal_amm7)

# Load AMM15

data = np.load(amm15_file)
temp_amm15 = data['temp_mgrid']
sal_amm15 = data['sal_mgrid']
mask_amm15 = data['mask_mgrid']
data.close()

print(mask_amm15.shape, np.min(mask_amm15), np.max(mask_amm15))
mask_amm15 = (mask_amm15 == 1) | (sal_amm15 < 0) | (temp_amm15 <= -1)
temp_amm15 = np.ma.masked_where((mask_amm15 == 1) | (mask_t_grid == 1), temp_amm15)
sal_amm15 = np.ma.masked_where((mask_amm15 == 1) | (mask_s_grid == 1), sal_amm15)

tdiff = (temp_grid - temp_mgrid) > 4

t_mask_all = temp_grid.mask | temp_mgrid.mask | temp_amm7.mask | temp_amm15.mask | tdiff
temp_grid = np.ma.masked_where(t_mask_all, temp_grid)
temp_mgrid = np.ma.masked_where(t_mask_all, temp_mgrid)
temp_amm7 = np.ma.masked_where(t_mask_all, temp_amm7)
temp_amm15 = np.ma.masked_where(t_mask_all, temp_amm15)

s_mask_all = sal_grid.mask | sal_mgrid.mask | sal_amm7.mask | sal_amm15.mask
sal_grid = np.ma.masked_where(s_mask_all, sal_grid)
sal_mgrid = np.ma.masked_where(s_mask_all, sal_mgrid)
sal_amm7 = np.ma.masked_where(s_mask_all, sal_amm7)
sal_amm15 = np.ma.masked_where(s_mask_all, sal_amm15)


for i in range(temp_mgrid.shape[0]):
  temp_mgrid[i, :] = np.ma.masked_where((h_mgrid > 200), temp_mgrid[i, :])
  sal_mgrid[i, :] = np.ma.masked_where((h_mgrid > 200), sal_mgrid[i, :])
  temp_grid[i, :] = np.ma.masked_where((h_mgrid > 200), temp_grid[i, :])
  sal_grid[i, :] = np.ma.masked_where((h_mgrid > 200), sal_grid[i, :])
  temp_amm7[i, :] = np.ma.masked_where((h_mgrid > 200), temp_amm7[i, :])
  sal_amm7[i, :] = np.ma.masked_where((h_mgrid > 200), sal_amm7[i, :])
  temp_amm15[i, :] = np.ma.masked_where((h_mgrid > 200), temp_amm15[i, :])
  sal_amm15[i, :] = np.ma.masked_where((h_mgrid > 200), sal_amm15[i, :])

data = np.load(in_dir + '/coast_distance.npz', allow_pickle=True)
dist = data['dist_c']
lon = data['lon']
lat = data['lat']
data.close()

# Select

ind_d = dep_grid < 200
mnt = np.array([d.month for d in o_date])
year = np.array([d.year for d in o_date])
ind1 = (year == 2019)

dist_g = np.zeros(temp_grid.shape)
for i in range(temp_grid.shape[0]):
  dist_g[i, :] = dist[m_node[i]]
coast = dist_g > 80
print(np.max(dist_g))

print(temp_mgrid.shape)
temp_mgrid = temp_mgrid[:, ind1]
sal_mgrid = sal_mgrid[:, ind1]
temp_grid = temp_grid[:, ind1]
sal_grid = sal_grid[:, ind1]
temp_amm7 = temp_amm7[:, ind1]
sal_amm7 = sal_amm7[:, ind1]
temp_amm15 = temp_amm15[:, ind1]
sal_amm15 = sal_amm15[:, ind1]
o_date = o_date[ind1]
#temp_amm15 = temp_amm15[:, ind_d]
#sal_amm15 = sal_amm15[:, ind_d]

# Only use un-masked

temp_mgrid_nm = np.ma.zeros(temp_mgrid.shape)
sal_mgrid_nm = np.ma.zeros(temp_mgrid.shape)
temp_grid_nm = np.ma.zeros(temp_mgrid.shape)
sal_grid_nm = np.ma.zeros(temp_mgrid.shape)
temp_amm7_nm = np.ma.zeros(temp_mgrid.shape)
sal_amm7_nm = np.ma.zeros(temp_mgrid.shape)
temp_amm15_nm = np.ma.zeros(temp_mgrid.shape)
sal_amm15_nm = np.ma.zeros(temp_mgrid.shape)
o_date_nm = np.zeros((len(o_date)), dtype=object)
c = 0
for i in range(temp_mgrid.shape[1]):
  if all(temp_mgrid[:, i].mask):
    continue

  temp_mgrid_nm[:, c] = temp_mgrid[:, i]
  sal_mgrid_nm[:, c] = sal_mgrid[:, i]
  temp_grid_nm[:, c] = temp_grid[:, i]
  sal_grid_nm[:, c] = sal_grid[:, i]
  temp_amm7_nm[:, c] = temp_amm7[:, i]
  sal_amm7_nm[:, c] = sal_amm7[:, i]
  temp_amm15_nm[:, c] = temp_amm15[:, i]
  sal_amm15_nm[:, c] = sal_amm15[:, i]
  o_date_nm[c] = o_date[i]
  c = c + 1


temp_mgrid = temp_mgrid_nm[:, :c]
sal_mgrid = sal_mgrid_nm[:, :c]
temp_grid = temp_grid_nm[:, :c]
sal_grid = sal_grid_nm[:, :c]
temp_amm7 = temp_amm7_nm[:, :c]
sal_amm7 = sal_amm7_nm[:, :c]
temp_amm15 = temp_amm15_nm[:, :c]
sal_amm15 = sal_amm15_nm[:, :c]
o_date = o_date_nm[:c]


# Make lines between min and max temperature for each profile

temp_grid1 = np.ma.min(temp_grid, axis=0)
temp_grid2 = np.ma.max(temp_grid, axis=0)
temp_mgrid1 = np.ma.min(temp_mgrid, axis=0)
temp_mgrid2 = np.ma.max(temp_mgrid, axis=0)
temp_amm71 = np.ma.min(temp_amm7, axis=0)
temp_amm72 = np.ma.max(temp_amm7, axis=0)
temp_amm151 = np.ma.min(temp_amm15, axis=0)
temp_amm152 = np.ma.max(temp_amm15, axis=0)

sal_grid1 = np.ma.min(sal_grid, axis=0)
sal_grid2 = np.ma.max(sal_grid, axis=0)
sal_mgrid1 = np.ma.min(sal_mgrid, axis=0)
sal_mgrid2 = np.ma.max(sal_mgrid, axis=0)
sal_amm71 = np.ma.min(sal_amm7, axis=0)
sal_amm72 = np.ma.max(sal_amm7, axis=0)
sal_amm151 = np.ma.min(sal_amm15, axis=0)
sal_amm152 = np.ma.max(sal_amm15, axis=0)

temp_grid_s = np.stack((temp_grid1, temp_grid2), axis=1)
temp_mgrid_s = np.stack((temp_mgrid1, temp_mgrid2), axis=1)
temp_amm7_s = np.stack((temp_amm71, temp_amm72), axis=1)
temp_amm15_s = np.stack((temp_amm151, temp_amm152), axis=1)
sal_grid_s = np.stack((sal_grid1, sal_grid2), axis=1)
sal_mgrid_s = np.stack((sal_mgrid1, sal_mgrid2), axis=1)
sal_amm7_s = np.stack((sal_amm71, sal_amm72), axis=1)
sal_amm15_s = np.stack((sal_amm151, sal_amm152), axis=1)
o_date_s = np.stack((o_date, o_date), axis=1)
print(temp_grid_s.shape)

#Plot

fig1 = plt.figure(figsize=(8,6))

ax1 = fig1.add_axes([0.08, 0.525, 0.86, 0.425])
ax2 = fig1.add_axes([0.08, 0.05, 0.86, 0.425])


tnow = dt.datetime.today()

ntemp = 0
nsal = 0
interval = 10
if 0:
  for i in range(0, len(temp_grid_s), interval):
    print(i / len(temp_grid_s) *100, '%')
    print(dt.datetime.today() - tnow)

    if i == 0:
      ax1.plot(o_date_s[i, :], temp_grid_s[i, :], 'k', label='Observations')
      ax1.plot(o_date_s[i, :], temp_mgrid_s[i, :], 'r', label='SSW-RS')
      ax1.plot(o_date_s[i, :], temp_amm7_s[i, :], 'g', label='AMM7')
      ax1.plot(o_date_s[i, :], temp_amm15_s[i, :], 'b', label='AMM15')
    else:
      ax1.plot(o_date_s[i, :], temp_grid_s[i, :], 'k')
      ax1.plot(o_date_s[i, :], temp_mgrid_s[i, :], 'r')
      ax1.plot(o_date_s[i, :], temp_amm7_s[i, :], 'g')
      ax1.plot(o_date_s[i, :], temp_amm15_s[i, :], 'b')

    if i == 0:
      ax2.plot(o_date_s[i, :], sal_grid_s[i, :], 'k', label='Observations')
      ax2.plot(o_date_s[i, :], sal_mgrid_s[i, :], 'r', label='SSW-RS')
      ax2.plot(o_date_s[i, :], sal_amm7_s[i, :], 'g', label='AMM7')
      ax2.plot(o_date_s[i, :], sal_amm15_s[i, :], 'b', label='AMM15')
    else:
      ax2.plot(o_date_s[i, :], sal_grid_s[i, :], 'k')
      ax2.plot(o_date_s[i, :], sal_mgrid_s[i, :], 'r')
      ax2.plot(o_date_s[i, :], sal_amm7_s[i, :], 'g')
      ax2.plot(o_date_s[i, :], sal_amm15_s[i, :], 'b')
else:
  print(o_date_s[:1, 0].shape, np.ma.mean(temp_grid, axis=0).shape)
  ax1.scatter(o_date_s[:1, 0], np.array([[30]]), s=6, color='k', label='Observations')
  ax1.scatter(o_date_s[:1, 0], np.array([[30]]), s=6, color='tab:blue', label='SSW-RS')
  ax1.scatter(o_date_s[:1, 0], np.array([[30]]), s=6, color='tab:orange', label='AMM15')

  ax1.scatter(o_date_s[:, 0], np.ma.mean(temp_grid, axis=0), s=0.2, color='k', zorder=100)
  ax1.scatter(o_date_s[:, 0], np.ma.mean(temp_mgrid+2, axis=0), s=0.2, color='tab:blue')
  ax1.scatter(o_date_s[:, 0], np.ma.mean(temp_amm15-2, axis=0), s=0.2, color='tab:orange')

  ax2.scatter(o_date_s[:, 0], np.ma.mean(sal_grid, axis=0), s=0.2, color='k', zorder=100)
  ax2.scatter(o_date_s[:, 0], np.ma.mean(sal_mgrid+2, axis=0), s=0.2, color='tab:blue')
  ax2.scatter(o_date_s[:, 0], np.ma.mean(sal_amm15-2, axis=0), s=0.2, color='tab:orange')

ntemp = np.ma.count(temp_grid_s[:i:interval, 0])
nsal = np.ma.count(sal_grid_s[:i:interval, 0])
print(ntemp, nsal)

ax1.legend(loc='lower right')
#ax2.legend(loc='lower right')

ax1.set_ylabel('Temperature ($^{\circ}$C)')
ax2.set_ylabel('Salinity (g/kg)')

ax1.set_ylim([0, 25])
ax2.set_ylim([20, 39])

ax1.set_xlim([dt.datetime(2019, 1, 1), dt.datetime(2020, 1, 1)])
ax2.set_xlim([dt.datetime(2019, 1, 1), dt.datetime(2020, 1, 1)])


ax1.annotate('(a)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

#with open('./number_profiles.txt', 'w') as f:
#  f.write('{:.0f} Temp. Profiles'.format(ntemp) + '\n')
#  f.write('{:.0f} Sal. Profiles'.format(nsal) + '\n')


fig1.savefig('./Figures/all_grid_time_2019.png')

