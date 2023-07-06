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


in_dir = '/scratch/benbar/Processed_Data_V3.02'

grid_obs_file = in_dir + '/Indices/grid_match_full.npz'
mod_file = in_dir + '/Indices/grid_model_full.npz'
amm7_file = in_dir + '/grid_amm7_all_full.npz'
amm15_file = in_dir + '/grid_amm15_all_full.npz'
match_file = in_dir + '/Indices/index_full.npz'

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

print(np.min(sal_grid), np.max(sal_grid))
mask_s_grid = (mask_s_grid == 1) | (sal_grid < 1) | (mask_t_grid == 1)
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
#mask_amm15 = (mask_amm15 == 1) | (sal_amm15 < 0) | (temp_amm15 <= -1)
#temp_amm15 = np.ma.masked_where((mask_amm15 == 1) | (mask_t_grid == 1), temp_amm15)
#sal_amm15 = np.ma.masked_where((mask_amm15 == 1) | (mask_s_grid == 1), sal_amm15)

t_mask_all = temp_grid.mask | temp_mgrid.mask | temp_amm7.mask
temp_grid = np.ma.masked_where(t_mask_all, temp_grid)
temp_mgrid = np.ma.masked_where(t_mask_all, temp_mgrid)
temp_amm7 = np.ma.masked_where(t_mask_all, temp_amm7)

s_mask_all = sal_grid.mask | sal_mgrid.mask | sal_amm7.mask
sal_grid = np.ma.masked_where(s_mask_all, sal_grid)
sal_mgrid = np.ma.masked_where(s_mask_all, sal_mgrid)
sal_amm7 = np.ma.masked_where(s_mask_all, sal_amm7)


for i in range(temp_mgrid.shape[0]):
  temp_mgrid[i, :] = np.ma.masked_where(h_mgrid > 200, temp_mgrid[i, :])
  sal_mgrid[i, :] = np.ma.masked_where(h_mgrid > 200, sal_mgrid[i, :])
  temp_grid[i, :] = np.ma.masked_where(h_mgrid > 200, temp_grid[i, :])
  sal_grid[i, :] = np.ma.masked_where(h_mgrid > 200, sal_grid[i, :])
  temp_amm7[i, :] = np.ma.masked_where(h_mgrid > 200, temp_amm7[i, :])
  sal_amm7[i, :] = np.ma.masked_where(h_mgrid > 200, sal_amm7[i, :])
#  temp_amm15[i, :] = np.ma.masked_where(h_mgrid > 200, temp_amm15[i, :])
#  sal_amm15[i, :] = np.ma.masked_where(h_mgrid > 200, sal_amm15[i, :])

data = np.load(in_dir + '/coast_distance.npz', allow_pickle=True)
dist = data['dist_c']
lon = data['lon']
lat = data['lat']
data.close()

# Select

ind_d = dep_grid < 200
mnt = np.array([d.month for d in o_date])
ind1 = ind_d * ((mnt <= 4) | (mnt >= 11))
ind2 = ind_d * ((mnt > 4) & (mnt < 11))

dist_g = np.zeros(temp_grid.shape)
for i in range(temp_grid.shape[0]):
  dist_g[i, :] = dist[m_node[i]]
coast = dist_g > 80
print(np.max(dist_g))

# Point Density Grid

t_int = 0.5
temp_range = np.arange(0, 25, t_int)

s_int = 0.5
sal_range = np.arange(0, 36.5, s_int)

to = ((temp_grid - temp_range[0]) / t_int).astype(np.int64)
tm = ((temp_mgrid - temp_range[0]) / t_int).astype(np.int64)
ta = ((temp_amm7 - temp_range[0]) / t_int).astype(np.int64)
#tn = ((temp_amm15 - temp_range[0]) / t_int).astype(np.int64)

so = ((sal_grid - sal_range[0]) / s_int).astype(np.int64)
sm = ((sal_mgrid - sal_range[0]) / s_int).astype(np.int64)
sa = ((sal_amm7 - sal_range[0]) / s_int).astype(np.int64)
#sn = ((sal_amm15 - sal_range[0]) / s_int).astype(np.int64)

t_valid = ((to < len(temp_range)) & (to >= 0) 
#          & (tm < len(temp_range)) & (tm >= 0) 
#          & (ta < len(temp_range)) & (ta >= 0) 
          & np.invert(to.mask))

s_valid = ((so < len(sal_range)) & (so >= 0) 
#          & (sm < len(sal_range)) & (sm >= 0) 
#          & (sa < len(sal_range)) & (sa >= 0) 
          & np.invert(so.mask))

to = to[t_valid & s_valid]
tm = tm[t_valid & s_valid]
ta = ta[t_valid & s_valid]
#tn = tn[t_valid & s_valid]
ind1t = ind1[t_valid & s_valid]
ind2t = ind2[t_valid & s_valid]
to1 = to[ind1t]
to2 = to[ind2t]
tm1 = tm[ind1t]
tm2 = tm[ind2t]
ta1 = ta[ind1t]
ta2 = ta[ind2t]
#tn1 = tn[ind1t]
#tn2 = tn[ind2t]

so = so[t_valid & s_valid]
sm = sm[t_valid & s_valid]
sa = sa[t_valid & s_valid]
#sn = sn[t_valid & s_valid]
ind1s = ind1[t_valid & s_valid]
ind2s = ind2[t_valid & s_valid]
so1 = so[ind1s]
so2 = so[ind2s]
sm1 = sm[ind1s]
sm2 = sm[ind2s]
sa1 = sa[ind1s]
sa2 = sa[ind2s]
#sn1 = sn[ind1s]
#sn2 = sn[ind2s]

# Regress

#slope, intercept, r_value, p_value, std_err = stats.linregress()
rms_tm1 = mean_squared_error(temp_grid[ind1], temp_mgrid[ind1], squared=False)
rms_tm2 = mean_squared_error(temp_grid[ind2], temp_mgrid[ind2], squared=False)
rms_ta1 = mean_squared_error(temp_grid[ind1], temp_amm7[ind1], squared=False)
rms_ta2 = mean_squared_error(temp_grid[ind2], temp_amm7[ind2], squared=False)
#rms_tn1 = mean_squared_error(temp_grid[ind1], temp_amm15[ind1], squared=False)
#rms_tn2 = mean_squared_error(temp_grid[ind2], temp_amm15[ind2], squared=False)
rms_sm1 = mean_squared_error(sal_grid[ind1], sal_mgrid[ind1], squared=False)
rms_sm2 = mean_squared_error(sal_grid[ind2], sal_mgrid[ind2], squared=False)
rms_sa1 = mean_squared_error(sal_grid[ind1], sal_amm7[ind1], squared=False)
rms_sa2 = mean_squared_error(sal_grid[ind2], sal_amm7[ind2], squared=False)
#rms_sn1 = mean_squared_error(sal_grid[ind1], sal_amm15[ind1], squared=False)
#rms_sn2 = mean_squared_error(sal_grid[ind2], sal_amm15[ind2], squared=False)

rms_tm1 = rms_tm1 / np.ma.count(temp_mgrid[ind1])
rms_tm2 = rms_tm2 / np.ma.count(temp_mgrid[ind2])
rms_ta1 = rms_ta1 / np.ma.count(temp_amm7[ind1])
rms_ta2 = rms_ta2 / np.ma.count(temp_amm7[ind2])
#rms_tn1 = rms_tn1 / np.ma.count(temp_amm15[ind1])
#rms_tn2 = rms_tn2 / np.ma.count(temp_amm15[ind2])
rms_sm1 = rms_sm1 / np.ma.count(sal_mgrid[ind1])
rms_sm2 = rms_sm2 / np.ma.count(sal_mgrid[ind2])
rms_sa1 = rms_sa1 / np.ma.count(sal_amm7[ind1])
rms_sa2 = rms_sa2 / np.ma.count(sal_amm7[ind2])
#rms_sn1 = rms_sn1 / np.ma.count(sal_amm15[ind1])
#rms_sn2 = rms_sn2 / np.ma.count(sal_amm15[ind2])


#Plot


if 1:
  fig2 = plt.figure(figsize=(10,8))

  obs_grid1 = np.zeros((len(sal_range), len(temp_range)))
  obs_grid2 = np.zeros((len(sal_range), len(temp_range)))
  mod_grid1 = np.zeros((len(sal_range), len(temp_range)))
  mod_grid2 = np.zeros((len(sal_range), len(temp_range)))
  amm7_grid1 = np.zeros((len(sal_range), len(temp_range)))
  amm7_grid2 = np.zeros((len(sal_range), len(temp_range)))
  #amm15_grid1 = np.zeros((len(sal_range), len(temp_range)))
  #amm15_grid2 = np.zeros((len(sal_range), len(temp_range)))

  for j in range(len(to1)):
    if to1.mask[j]:
      continue
    obs_grid1[so1[j], to1[j]] = obs_grid1[so1[j], to1[j]] + 1
  for j in range(len(to2)):
    if to2.mask[j]:
      continue
    obs_grid2[so2[j], to2[j]] = obs_grid2[so2[j], to2[j]] + 1
  for j in range(len(sm1)):
    if sm1.mask[j]:
      continue
    mod_grid1[sm1[j], tm1[j]] = mod_grid1[sm1[j], tm1[j]] + 1
  for j in range(len(sm2)):
    if sm2.mask[j]:
      continue
    mod_grid2[sm2[j], tm2[j]] = mod_grid2[sm2[j], tm2[j]] + 1
  for j in range(len(sa1)):
    if sa1.mask[j]:
      continue
    amm7_grid1[sa1[j], ta1[j]] = amm7_grid1[sa1[j], ta1[j]] + 1
  for j in range(len(sa2)):
    if sa2.mask[j]:
      continue
    amm7_grid2[sa2[j], ta2[j]] = amm7_grid2[sa2[j], ta2[j]] + 1
  #for j in range(len(sn1)):
  #  if sn1.mask[j]:
  #    continue
  #  amm15_grid1[sn1[j], tn1[j]] = amm15_grid1[sn1[j], tn1[j]] + 1
  #for j in range(len(sn2)):
  #  if sn2.mask[j]:
  #    continue
  #  amm15_grid2[sn2[j], tn2[j]] = amm15_grid2[sn2[j], tn2[j]] + 1

  obs_grid1 = ((obs_grid1 / np.ma.sum(obs_grid1)) * 100).T
  obs_grid2 = ((obs_grid2 / np.ma.sum(obs_grid2)) * 100).T
  mod_grid1 = ((mod_grid1 / np.ma.sum(mod_grid1)) * 100).T
  mod_grid2 = ((mod_grid2 / np.ma.sum(mod_grid2)) * 100).T
  amm7_grid1 = ((amm7_grid1 / np.ma.sum(amm7_grid1)) * 100).T
  amm7_grid2 = ((amm7_grid2 / np.ma.sum(amm7_grid2)) * 100).T
  #amm15_grid1 = ((amm15_grid1 / np.ma.sum(amm15_grid1)) * 100).T
  #amm15_grid2 = ((amm15_grid2 / np.ma.sum(amm15_grid2)) * 100).T

  obs_grid1 = np.ma.masked_where(obs_grid1 == 0, obs_grid1)
  obs_grid2 = np.ma.masked_where(obs_grid2 == 0, obs_grid2)
  mod_grid1 = np.ma.masked_where(mod_grid1 == 0, mod_grid1)
  mod_grid2 = np.ma.masked_where(mod_grid2 == 0, mod_grid2)
  amm7_grid1 = np.ma.masked_where(amm7_grid1 == 0, amm7_grid1)
  amm7_grid2 = np.ma.masked_where(amm7_grid2 == 0, amm7_grid2)
  #amm15_grid1 = np.ma.masked_where(amm15_grid1 == 0, amm15_grid1)
  #amm15_grid2 = np.ma.masked_where(amm15_grid2 == 0, amm15_grid2)

  ax1 = fig2.add_axes([0.07, 0.60, 0.24, 0.35])
  ax2 = fig2.add_axes([0.38, 0.60, 0.24, 0.35])
  ax3 = fig2.add_axes([0.69, 0.60, 0.24, 0.35])
  #ax4 = fig2.add_axes([0.79, 0.60, 0.17, 0.35])

  ax5 = fig2.add_axes([0.07, 0.15, 0.24, 0.35])
  ax6 = fig2.add_axes([0.38, 0.15, 0.24, 0.35])
  ax7 = fig2.add_axes([0.69, 0.15, 0.24, 0.35])
  #ax8 = fig2.add_axes([0.79, 0.15, 0.17, 0.35])
  cax1 = fig2.add_axes([0.2, 0.08, 0.6, 0.01])


  ax1.pcolormesh(sal_range, temp_range, obs_grid1, vmin=0, vmax=2, zorder=99)
  ax2.pcolormesh(sal_range, temp_range, mod_grid1, vmin=0, vmax=2, zorder=99)
  ax3.pcolormesh(sal_range, temp_range, amm7_grid1, vmin=0, vmax=2, zorder=99)
  #ax4.pcolormesh(sal_range, temp_range, amm15_grid1, vmin=0, vmax=2, zorder=99)

  ax5.pcolormesh(sal_range, temp_range, obs_grid2, vmin=0, vmax=2, zorder=99)
  ax6.pcolormesh(sal_range, temp_range, mod_grid2, vmin=0, vmax=2, zorder=99)
  cs1 = ax7.pcolormesh(sal_range, temp_range, amm7_grid2, vmin=0, vmax=2, zorder=99)
  #cs1 = ax8.pcolormesh(sal_range, temp_range, amm15_grid2, vmin=0, vmax=2, zorder=99)

  #ax1.plot(sal_grid[ind1 & coast], temp_grid[ind1 & coast], 'b.', zorder=100)
  #ax2.plot(sal_mgrid[ind1 & coast], temp_mgrid[ind1 & coast], 'b.', zorder=100)
  #ax4.plot(sal_grid[ind2 & coast], temp_grid[ind2 & coast], 'b.', zorder=100)
  #ax5.plot(sal_mgrid[ind2 & coast], temp_mgrid[ind2 & coast], 'b.', zorder=100)

  ax1.set_title('Winter')
  ax2.set_title('Winter')
  ax3.set_title('Winter')
  #ax4.set_title('Winter')
  ax5.set_title('Summer')
  ax6.set_title('Summer')
  ax7.set_title('Summer')
  #ax8.set_title('Summer')

  ax1.set_xlabel('Salinity Obs.')
  ax5.set_xlabel('Salinity Obs.')
  ax2.set_xlabel('Salinity SSW-RS')
  ax6.set_xlabel('Salinity SSW-RS')
  ax3.set_xlabel('Salinity AMM7')
  ax7.set_xlabel('Salinity AMM7')
  #ax4.set_xlabel('Salinity AMM15')
  #ax8.set_xlabel('Salinity AMM15')

  ax1.set_ylabel('Temperature Obs. ($^{\circ}$C)')
  ax5.set_ylabel('Temperature Obs. ($^{\circ}$C)')
  ax2.set_ylabel('Temperature SSW-RS ($^{\circ}$C)')
  ax6.set_ylabel('Temperature SSW-RS ($^{\circ}$C)')
  ax3.set_ylabel('Temperature AMM7 ($^{\circ}$C)')
  ax7.set_ylabel('Temperature AMM7 ($^{\circ}$C)')
  #ax4.set_ylabel('Temperature AMM15 ($^{\circ}$C)')
  #ax8.set_ylabel('Temperature AMM15 ($^{\circ}$C)')

  ax1.set_xlim([10, 36.5])
  ax2.set_xlim([10, 36.5])
  ax3.set_xlim([10, 36.5])
  #ax4.set_xlim([10, 36.5])
  ax5.set_xlim([10, 36.5])
  ax6.set_xlim([10, 36.5])
  ax7.set_xlim([10, 36.5])
  #ax8.set_xlim([10, 36.5])
  ax1.set_ylim([0, 22])
  ax2.set_ylim([0, 22])
  ax3.set_ylim([0, 22])
  #ax4.set_ylim([0, 22])
  ax5.set_ylim([0, 22])
  ax6.set_ylim([0, 22])
  ax7.set_ylim([0, 22])
  #ax8.set_ylim([0, 22])

  fig2.colorbar(cs1, cax=cax1, extend='max', orientation='horizontal')
  cax1.set_xlabel('Points (%)')

  ax1.annotate('(a) Obs.', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2.annotate('(b) SSW-RS', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax3.annotate('(c) AMM7', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  #ax4.annotate('(d) AMM15', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax5.annotate('(e) Obs.', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax6.annotate('(f) SSW-RS', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax7.annotate('(g) AMM7', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  #ax8.annotate('(h) AMM15', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)



  fig3 = plt.figure(figsize=(10,8))

  ax1 = fig3.add_axes([0.1, 0.55, 0.35, 0.35])
  ax2 = fig3.add_axes([0.55, 0.55, 0.35, 0.35])
  ax3 = fig3.add_axes([0.1, 0.1, 0.35, 0.35])
  ax4 = fig3.add_axes([0.55, 0.1, 0.35, 0.35])

  if 0:
    cax1 = fig3.add_axes([0.91, 0.1, 0.02, 0.8])

    temp_grid1 = np.zeros((len(temp_range), len(temp_range)))
    temp_grid2 = np.zeros((len(temp_range), len(temp_range)))
    sal_grid1 = np.zeros((len(sal_range), len(sal_range)))
    sal_grid2 = np.zeros((len(sal_range), len(sal_range)))

    for j in range(len(to1)):
      if to1.mask[j] | tm1.mask[j]:
        continue
      temp_grid1[to1[j], tm1[j]] = temp_grid1[to1[j], tm1[j]] + 1
    for j in range(len(to2)):
      if to2.mask[j] | tm2.mask[j]:
        continue
      temp_grid2[to2[j], tm2[j]] = temp_grid2[to2[j], tm2[j]] + 1
    for j in range(len(so1)):
      if so1.mask[j] | sm1.mask[j]:
        continue
      sal_grid1[so1[j], sm1[j]] = sal_grid1[so1[j], sm1[j]] + 1
    for j in range(len(so2)):
      if so2.mask[j] | sm2.mask[j]:
        continue
      sal_grid2[so2[j], sm2[j]] = sal_grid2[so2[j], sm2[j]] + 1

    temp_grid1 = (temp_grid1 / np.ma.sum(temp_grid1)) * 100
    temp_grid2 = (temp_grid2 / np.ma.sum(temp_grid2)) * 100
    sal_grid1 = (sal_grid1 / np.ma.sum(sal_grid1)) * 100
    sal_grid2 = (sal_grid2 / np.ma.sum(sal_grid2)) * 100

    temp_grid1 = np.ma.masked_where(temp_grid1 == 0, temp_grid1)
    temp_grid2 = np.ma.masked_where(temp_grid2 == 0, temp_grid2)
    sal_grid1 = np.ma.masked_where(sal_grid1 == 0, sal_grid1)
    sal_grid2 = np.ma.masked_where(sal_grid2 == 0, sal_grid2)

    ax1.plot([4, 20], [4, 20], '-k', zorder=101)
    ax1.pcolormesh(temp_range, temp_range, temp_grid1, vmin=0, vmax=2, zorder=99)

    ax2.plot([23, 35], [23, 35], '-k', zorder=101)
    cs1 = ax2.pcolormesh(sal_range, sal_range, sal_grid1, vmin=0, vmax=2, zorder=99)

    ax3.plot([4, 20], [4, 20], '-k', zorder=101)
    ax3.pcolormesh(temp_range, temp_range, temp_grid2, vmin=0, vmax=2, zorder=99)

    ax4.plot([23, 35], [23, 35], '-k', zorder=101)
    cs1 = ax4.pcolormesh(sal_range, sal_range, sal_grid2, vmin=0, vmax=2, zorder=99)

    fig3.colorbar(cs1, cax=cax1, extend='max')
    cax1.set_ylabel('Points (%)')


  elif 1:
    al = 0.2
    ax1.plot(temp_grid[ind1], temp_mgrid[ind1], '.r', ms=2, alpha=al)
    ax1.plot(temp_grid[ind1], temp_amm7[ind1], '.g', ms=2, alpha=al)
    ax2.plot(sal_grid[ind1], sal_mgrid[ind1], '.r', ms=2, alpha=al)
    ax2.plot(sal_grid[ind1], sal_amm7[ind1], '.g', ms=2, alpha=al)

    ax3.plot(temp_grid[ind2], temp_mgrid[ind2], '.r', ms=2, alpha=al)
    ax3.plot(temp_grid[ind2], temp_amm7[ind2], '.g', ms=2, alpha=al)
    ax4.plot(sal_grid[ind2], sal_mgrid[ind2], '.r', ms=2, alpha=al)
    ax4.plot(sal_grid[ind2], sal_amm7[ind2], '.g', ms=2, alpha=al)

  else:
    al = 0.2

    cax1 = fig3.add_axes([0.91, 0.3, 0.02, 0.4])
    cs1 = ax1.scatter(temp_grid[ind1], temp_mgrid[ind1], c=dist_g[ind1], s=2, marker='o', vmin=0, vmax=200, alpha=al)
    ax1.scatter(temp_grid[ind1], temp_amm7[ind1], c=dist_g[ind1], s=2, marker='o', vmin=0, vmax=200, alpha=al)
    ax2.scatter(sal_grid[ind1], sal_mgrid[ind1], c=dist_g[ind1], s=2, marker='o', vmin=0, vmax=200, alpha=al)
    ax2.scatter(sal_grid[ind1], sal_amm7[ind1], c=dist_g[ind1], s=2, marker='o', vmin=0, vmax=200, alpha=al)

    ax3.scatter(temp_grid[ind2], temp_mgrid[ind2], c=dist_g[ind2], s=2, marker='^', vmin=0, vmax=200, alpha=al)
    ax3.scatter(temp_grid[ind2], temp_amm7[ind2], c=dist_g[ind2], s=2, marker='^', vmin=0, vmax=200, alpha=al)
    ax4.scatter(sal_grid[ind2], sal_mgrid[ind2], c=dist_g[ind2], s=2, marker='^', vmin=0, vmax=200, alpha=al)
    ax4.scatter(sal_grid[ind2], sal_amm7[ind2], c=dist_g[ind2], s=2, marker='^', vmin=0, vmax=200, alpha=al)
    fig2.colorbar(cs1, cax=cax1)
    cax1.set_ylabel('Distance from coast (km)')

  sum_t_ssw = np.ma.count(temp_mgrid)
  sum_s_ssw = np.ma.count(sal_mgrid)
  sum_t_amm7 = np.ma.count(temp_amm7)
  sum_s_amm7 = np.ma.count(sal_amm7)

  sum_t_obs = np.ma.count(np.ma.max(temp_grid, axis=0))
  sum_s_obs = np.ma.count(np.ma.max(sal_grid, axis=0))

  print('Number Points', sum_t_ssw, sum_s_ssw, sum_t_amm7, sum_s_amm7)
  print('Number Profiles', sum_t_obs, sum_s_obs)

  ax1.plot([-2, 40], [-2, 40], 'k-')
  ax2.plot([-2, 40], [-2, 40], 'k-')
  ax3.plot([-2, 40], [-2, 40], 'k-')
  ax4.plot([-2, 40], [-2, 40], 'k-')

  ax1.set_xlim([-1, 25])
  ax1.set_ylim([-1, 25])
  ax2.set_xlim([0, 36])
  ax2.set_ylim([0, 36])
  ax3.set_xlim([-1, 25])
  ax3.set_ylim([-1, 25])
  ax4.set_xlim([0, 36])
  ax4.set_ylim([0, 36])

  ax1.set_title('Winter')
  ax2.set_title('Winter')
  ax3.set_title('Summer')
  ax4.set_title('Summer')

  ax1.set_ylabel('Temperature Model ($^{\circ}$C)')
  ax1.set_xlabel('Temperature Obs ($^{\circ}$C)')
  ax3.set_ylabel('Temperature Model ($^{\circ}$C)')
  ax3.set_xlabel('Temperature Obs ($^{\circ}$C)')

  ax2.set_ylabel('Salinity Model')
  ax2.set_xlabel('Salinity Obs.')
  ax4.set_ylabel('Salinity Model')
  ax4.set_xlabel('Salinity Obs.')


  ax1.annotate('(a)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2.annotate('(b)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax3.annotate('(c)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax4.annotate('(d)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax1.annotate('SSW-RS={:.4f}'.format(rms_tm1), (0.6, 0.18), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax1.annotate('AMM7={:.4f}'.format(rms_ta1), (0.6, 0.08), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax2.annotate('SSW-RS={:.4f}'.format(rms_sm1), (0.6, 0.18), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2.annotate('AMM7={:.4f}'.format(rms_sa1), (0.6, 0.08), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax3.annotate('SSW-RS={:.4f}'.format(rms_tm2), (0.6, 0.18), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax3.annotate('AMM7={:.4f}'.format(rms_ta2), (0.6, 0.08), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax4.annotate('SSW-RS={:.4f}'.format(rms_sm2), (0.6, 0.18), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax4.annotate('AMM7={:.4f}'.format(rms_sa2), (0.6, 0.08), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)


if 0:
  fig4 = plt.figure(figsize=(12,8))

  ax1 = fig4.add_axes([0.1, 0.55, 0.35, 0.35])
  ax2 = fig4.add_axes([0.55, 0.55, 0.35, 0.35])
  ax3 = fig4.add_axes([0.1, 0.1, 0.35, 0.35])
  ax4 = fig4.add_axes([0.55, 0.1, 0.35, 0.35])

  fig5 = plt.figure(figsize=(12,8))

  ax5 = fig5.add_axes([0.1, 0.55, 0.35, 0.35])
  ax6 = fig5.add_axes([0.55, 0.55, 0.35, 0.35])
  ax7 = fig5.add_axes([0.1, 0.1, 0.35, 0.35])
  ax8 = fig5.add_axes([0.55, 0.1, 0.35, 0.35])


  for i in range(0, len(o_prof), 500):
    print(i / len(o_prof) *100, '%')
    ind = prof_obs == o_prof[i]
    if (np.ma.max(dep_obs[ind]) < 50) | (np.ma.max(dep_obs[ind]) > 200):
      continue
    if (date_obs[i].month <= 4) | (date_obs[i].month >= 11):
      ax1.plot(temp_obs[ind], dep_obs[ind], 'k-')
      ax2.plot(temp_mod[ind], dep_obs[ind], 'r-')
      ax5.plot(sal_obs[ind], dep_obs[ind], 'k-')
      ax6.plot(sal_mod[ind], dep_obs[ind], 'r-')
    else:
      ax3.plot(temp_obs[ind], dep_obs[ind], 'k-')
      ax4.plot(temp_mod[ind], dep_obs[ind], 'r-')
      ax7.plot(sal_obs[ind], dep_obs[ind], 'k-')
      ax8.plot(sal_mod[ind], dep_obs[ind], 'r-')


  ax1.set_title('Winter (Nov-Apr)')
  ax2.set_title('Winter (Nov-Apr)')
  ax3.set_title('Summer (May-Oct)')
  ax4.set_title('Summer (May-Oct)')
  ax5.set_title('Winter (Nov-Apr)')
  ax6.set_title('Winter (Nov-Apr)')
  ax7.set_title('Summer (May-Oct)')
  ax8.set_title('Summer (May-Oct)')

  ax1.set_xlabel('Temperature ($^{\circ}$C)')
  ax2.set_xlabel('Temperature ($^{\circ}$C)')
  ax3.set_xlabel('Temperature ($^{\circ}$C)')
  ax4.set_xlabel('Temperature ($^{\circ}$C)')

  ax5.set_xlabel('Salinity')
  ax6.set_xlabel('Salinity')
  ax7.set_xlabel('Salinity')
  ax8.set_xlabel('Salinity')

  ax1.set_ylabel('Depth (m)')
  ax2.set_ylabel('Depth (m)')
  ax3.set_ylabel('Depth (m)')
  ax4.set_ylabel('Depth (m)')
  ax5.set_ylabel('Depth (m)')
  ax6.set_ylabel('Depth (m)')
  ax7.set_ylabel('Depth (m)')
  ax8.set_ylabel('Depth (m)')

  ax1.set_xlim([4, 16])
  ax2.set_xlim([4, 16])
  ax3.set_xlim([4, 20])
  ax4.set_xlim([4, 20])
  ax5.set_xlim([28, 35.5])
  ax6.set_xlim([28, 35.5])
  ax7.set_xlim([30, 35.5])
  ax8.set_xlim([30, 35.5])


  ax1.invert_yaxis()
  ax2.invert_yaxis()
  ax3.invert_yaxis()
  ax4.invert_yaxis()
  ax5.invert_yaxis()
  ax6.invert_yaxis()
  ax7.invert_yaxis()
  ax8.invert_yaxis()

  ax1.set_ylim([200, 0])
  ax2.set_ylim([200, 0])
  ax3.set_ylim([200, 0])
  ax4.set_ylim([200, 0])
  ax5.set_ylim([200, 0])
  ax6.set_ylim([200, 0])
  ax7.set_ylim([200, 0])
  ax8.set_ylim([200, 0])

  ax1.annotate('(a)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2.annotate('(b)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax3.annotate('(c)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax4.annotate('(d)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax5.annotate('(a)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax6.annotate('(b)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax7.annotate('(c)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax8.annotate('(d)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)



fig2.savefig('./Figures/all_grid_ts_equal.png')
fig3.savefig('./Figures/all_grid_vs_equal.png')
#fig4.savefig('./Figures/all_grid_depth_t.png')
#fig5.savefig('./Figures/all_grid_depth_s.png')

