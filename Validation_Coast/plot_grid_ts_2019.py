#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
from functools import partial


in_dir = '/scratch/benbar/Processed_Data_V3.02'

grid_obs_file = in_dir + '/Indices/grid_match_2019.npz'
mod_file = in_dir + '/Indices/grid_model_2019.npz'
amm7_file = in_dir + '/grid_amm7_all_2019.npz'
amm15_file = in_dir + '/grid_amm15_cmems_all_2019.npz'
match_file = in_dir + '/Indices/index_2019.npz'

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

t_mask_all = temp_grid.mask | temp_mgrid.mask | temp_amm7.mask | temp_amm15.mask| tdiff
temp_grid = np.ma.masked_where(t_mask_all, temp_grid)
temp_mgrid = np.ma.masked_where(t_mask_all, temp_mgrid)
temp_amm7 = np.ma.masked_where(t_mask_all, temp_amm7)
temp_amm15 = np.ma.masked_where(t_mask_all, temp_amm15)

s_mask_all = sal_grid.mask | sal_mgrid.mask | sal_amm7.mask | sal_amm15.mask | tdiff
sal_grid = np.ma.masked_where(s_mask_all, sal_grid)
sal_mgrid = np.ma.masked_where(s_mask_all, sal_mgrid)
sal_amm7 = np.ma.masked_where(s_mask_all, sal_amm7)
sal_amm15 = np.ma.masked_where(s_mask_all, sal_amm15)


x_min = -10.05
x_max = 2.05

y_min = 53.98
y_max = 61.02

if 0:
  # Baltic
  x_min = 5.05
  x_max = 20.05

  y_min = 48.98
  y_max = 61.02

sel_ind = np.invert((lon_obs >= x_min) & (lon_obs <= x_max) 
                  & (lat_obs >= y_min) & (lat_obs <= y_max))
do_vs = 1
if do_vs:
  sel_ind = np.zeros(h_mgrid.shape, dtype=bool)

for i in range(temp_mgrid.shape[0]):
  temp_mgrid[i, :] = np.ma.masked_where((h_mgrid > 200) | sel_ind, temp_mgrid[i, :])
  sal_mgrid[i, :] = np.ma.masked_where((h_mgrid > 200) | sel_ind, sal_mgrid[i, :])
  temp_grid[i, :] = np.ma.masked_where((h_mgrid > 200) | sel_ind, temp_grid[i, :])
  sal_grid[i, :] = np.ma.masked_where((h_mgrid > 200) | sel_ind, sal_grid[i, :])
  temp_amm7[i, :] = np.ma.masked_where((h_mgrid > 200) | sel_ind, temp_amm7[i, :])
  sal_amm7[i, :] = np.ma.masked_where((h_mgrid > 200) | sel_ind, sal_amm7[i, :])
  temp_amm15[i, :] = np.ma.masked_where((h_mgrid > 200) | sel_ind, temp_amm15[i, :])
  sal_amm15[i, :] = np.ma.masked_where((h_mgrid > 200) | sel_ind, sal_amm15[i, :])

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

t_int = 0.25
temp_range = np.arange(0, 25, t_int)

s_int = 0.25
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
rms_ta1 = mean_squared_error(temp_grid[ind1], temp_amm15[ind1], squared=False)
rms_ta2 = mean_squared_error(temp_grid[ind2], temp_amm15[ind2], squared=False)
#rms_tn1 = mean_squared_error(temp_grid[ind1], temp_amm15[ind1], squared=False)
#rms_tn2 = mean_squared_error(temp_grid[ind2], temp_amm15[ind2], squared=False)
rms_sm1 = mean_squared_error(sal_grid[ind1], sal_mgrid[ind1], squared=False)
rms_sm2 = mean_squared_error(sal_grid[ind2], sal_mgrid[ind2], squared=False)
rms_sa1 = mean_squared_error(sal_grid[ind1], sal_amm15[ind1], squared=False)
rms_sa2 = mean_squared_error(sal_grid[ind2], sal_amm15[ind2], squared=False)
#rms_sn1 = mean_squared_error(sal_grid[ind1], sal_amm15[ind1], squared=False)
#rms_sn2 = mean_squared_error(sal_grid[ind2], sal_amm15[ind2], squared=False)

rms_tm1 = rms_tm1 / np.ma.count(temp_mgrid[ind1])
rms_tm2 = rms_tm2 / np.ma.count(temp_mgrid[ind2])
rms_ta1 = rms_ta1 / np.ma.count(temp_amm15[ind1])
rms_ta2 = rms_ta2 / np.ma.count(temp_amm15[ind2])
#rms_tn1 = rms_tn1 / np.ma.count(temp_amm15[ind1])
#rms_tn2 = rms_tn2 / np.ma.count(temp_amm15[ind2])
rms_sm1 = rms_sm1 / np.ma.count(sal_mgrid[ind1])
rms_sm2 = rms_sm2 / np.ma.count(sal_mgrid[ind2])
rms_sa1 = rms_sa1 / np.ma.count(sal_amm15[ind1])
rms_sa2 = rms_sa2 / np.ma.count(sal_amm15[ind2])
#rms_sn1 = rms_sn1 / np.ma.count(sal_amm15[ind1])
#rms_sn2 = rms_sn2 / np.ma.count(sal_amm15[ind2])


tl = np.arange(-1, 22.5, 0.05)
sl = np.arange(9, 37, 0.05)
slg, tlg = np.meshgrid(sl, tl)
sa = sw.SA_from_SP(slg, 0, 0, 58)
ct = sw.CT_from_pt(sa, tlg)
rl = sw.sigma0(sa, ct)

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



  ax1 = fig2.add_axes([0.07, 0.60, 0.10, 0.35])
  ax1a = fig2.add_axes([0.17, 0.60, 0.14, 0.35])
  ax2 = fig2.add_axes([0.38, 0.60, 0.10, 0.35])
  ax2a = fig2.add_axes([0.48, 0.60, 0.14, 0.35])
  ax3 = fig2.add_axes([0.69, 0.60, 0.10, 0.35])
  ax3a = fig2.add_axes([0.79, 0.60, 0.14, 0.35])
  #ax4 = fig2.add_axes([0.79, 0.60, 0.17, 0.35])

  ax5 = fig2.add_axes([0.07, 0.15, 0.10, 0.35])
  ax5a = fig2.add_axes([0.17, 0.15, 0.14, 0.35])
  ax6 = fig2.add_axes([0.38, 0.15, 0.10, 0.35])
  ax6a = fig2.add_axes([0.48, 0.15, 0.14, 0.35])
  ax7 = fig2.add_axes([0.69, 0.15, 0.10, 0.35])
  ax7a = fig2.add_axes([0.79, 0.15, 0.14, 0.35])
  #ax8 = fig2.add_axes([0.79, 0.15, 0.17, 0.35])
  cax1 = fig2.add_axes([0.2, 0.08, 0.6, 0.01])

  s_off = 0#36.5

  ax1.pcolormesh(sal_range - s_off, temp_range, obs_grid1, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  ax1a.pcolormesh(sal_range - s_off, temp_range, obs_grid1, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  ax2.pcolormesh(sal_range - s_off, temp_range, mod_grid1, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  ax2a.pcolormesh(sal_range - s_off, temp_range, mod_grid1, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  ax3.pcolormesh(sal_range - s_off, temp_range, amm7_grid1, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  ax3a.pcolormesh(sal_range - s_off, temp_range, amm7_grid1, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  #ax4.pcolormesh(sal_range, temp_range, amm15_grid1, vmin=0, vmax=2, zorder=99)

  ax5.pcolormesh(sal_range - s_off, temp_range, obs_grid2, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  ax5a.pcolormesh(sal_range - s_off, temp_range, obs_grid2, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  ax6.pcolormesh(sal_range - s_off, temp_range, mod_grid2, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  ax6a.pcolormesh(sal_range - s_off, temp_range, mod_grid2, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  ax7.pcolormesh(sal_range - s_off, temp_range, amm7_grid2, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())
  cs1 = ax7a.pcolormesh(sal_range - s_off, temp_range, amm7_grid2, vmin=0.01, vmax=3, zorder=99, norm=colors.LogNorm())  #cs1 = ax8.pcolormesh(sal_range, temp_range, amm15_grid2, vmin=0, vmax=2, zorder=99)

  #ax1.plot(sal_grid[ind1 & coast], temp_grid[ind1 & coast], 'b.', zorder=100)
  #ax2.plot(sal_mgrid[ind1 & coast], temp_mgrid[ind1 & coast], 'b.', zorder=100)
  #ax4.plot(sal_grid[ind2 & coast], temp_grid[ind2 & coast], 'b.', zorder=100)
  #ax5.plot(sal_mgrid[ind2 & coast], temp_mgrid[ind2 & coast], 'b.', zorder=100)

  def add_watermasses(ax, part, slg, tlg, rl, s_off):
    # plot lines
    d_levs = np.arange(5, 30, 1)
    ax.contour(slg - s_off, tlg, rl, d_levs, colors='0.6', linewidths=(0.5), zorder=103)

    mtick = np.array([15, 20, 25, 30, 35]) - s_off
    mlabel = ['15', '20', '25', '30', '35']
    #ax.set_xscale('symlog')
    ax.set_xticks(mtick)
    ax.set_xticklabels(mlabel)

    if part == 1:
      #ax.plot(np.array([30, 30, 10, 10, 30]) - s_off, [0, 22, 22, 0, 0], 'w', lw=1, zorder=100)
      #ax.plot(np.array([30, 30, 10, 10, 30]) - s_off, [0, 22, 22, 0, 0], '--k', lw=1, zorder=101)
      #ax.annotate('BW', (0.60, 0.05), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

      ax.set_xlim([15 - s_off, 30 - s_off])
      ax.set_ylim([3, 20])

    else:
      ax.plot(np.array([34.5, 34.5, 0, 0, 34.5]) - s_off, [0, 22, 22, 0, 0], 'w', lw=1, zorder=100)
      ax.plot(np.array([34.5, 34.5, 0, 0, 34.5]) - s_off, [0, 22, 22, 0, 0], '--k', lw=1, zorder=101)
      ax.annotate('CW', (0.25, 0.05), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

      ax.plot(np.array([35, 35, 37, 37, 35]) - s_off, [0, 22, 22, 0, 0], 'w', lw=1, zorder=100)
      ax.plot(np.array([35, 35, 37, 37, 35]) - s_off, [0, 22, 22, 0, 0], '--k', lw=1, zorder=101)
      ax.annotate('AW', (0.8, 0.05), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

      ax.set_xlim([30 - s_off, 37 - s_off])
      ax.set_ylim([3, 20])
      ax.set_yticks([])


  add_watermasses(ax1, 1, slg, tlg, rl, s_off)
  add_watermasses(ax1a, 2, slg, tlg, rl, s_off)
  add_watermasses(ax2, 1, slg, tlg, rl, s_off)
  add_watermasses(ax2a, 2, slg, tlg, rl, s_off)
  add_watermasses(ax3, 1, slg, tlg, rl, s_off)
  add_watermasses(ax3a, 2, slg, tlg, rl, s_off)
  add_watermasses(ax5, 1, slg, tlg, rl, s_off)
  add_watermasses(ax5a, 2, slg, tlg, rl, s_off)
  add_watermasses(ax6, 1, slg, tlg, rl, s_off)
  add_watermasses(ax6a, 2, slg, tlg, rl, s_off)
  add_watermasses(ax7, 1, slg, tlg, rl, s_off)
  add_watermasses(ax7a, 2, slg, tlg, rl, s_off)


  ax1.set_title('Winter')
  ax2.set_title('Winter')
  ax3.set_title('Winter')
  #ax4.set_title('Winter')
  ax5.set_title('Summer')
  ax6.set_title('Summer')
  ax7.set_title('Summer')
  #ax8.set_title('Summer')

  ax1.set_xlabel('Salinity Obs. (g/kg)')
  ax5.set_xlabel('Salinity Obs. (g/kg)')
  ax2.set_xlabel('Salinity SSW-RS (g/kg)')
  ax6.set_xlabel('Salinity SSW-RS (g/kg)')
  ax3.set_xlabel('Salinity AMM7 (g/kg)')
  ax7.set_xlabel('Salinity AMM7 (g/kg)')
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

  ax1 = fig3.add_axes([0.055, 0.55, 0.17, 0.35])
  ax2 = fig3.add_axes([0.28, 0.55, 0.17, 0.35])
  ax3 = fig3.add_axes([0.505, 0.55, 0.17, 0.35])
  ax4 = fig3.add_axes([0.73, 0.55, 0.17, 0.35])

  ax5 = fig3.add_axes([0.055, 0.1, 0.17, 0.35])
  ax6 = fig3.add_axes([0.28, 0.1, 0.17, 0.35])
  ax7 = fig3.add_axes([0.505, 0.1, 0.17, 0.35])
  ax8 = fig3.add_axes([0.73, 0.1, 0.17, 0.35])

  cax1 = fig3.add_axes([0.91, 0.3, 0.02, 0.4])

  al = 0.2

  cs1 = ax1.scatter(temp_grid[0:2], temp_mgrid[0:2]+30, c=dist_g[0:2], s=2, marker='o', vmin=0, vmax=200)
  ax1.scatter(temp_grid[ind1], temp_mgrid[ind1], c=dist_g[ind1], s=2, marker='o', vmin=0, vmax=200, alpha=al)
  ax2.scatter(temp_grid[ind1], temp_amm15[ind1], c=dist_g[ind1], s=2, marker='o', vmin=0, vmax=200, alpha=al)
  ax3.scatter(sal_grid[ind1], sal_mgrid[ind1], c=dist_g[ind1], s=2, marker='o', vmin=0, vmax=200, alpha=al)
  ax4.scatter(sal_grid[ind1], sal_amm15[ind1], c=dist_g[ind1], s=2, marker='o', vmin=0, vmax=200, alpha=al)

  ax5.scatter(temp_grid[ind2], temp_mgrid[ind2], c=dist_g[ind2], s=2, marker='o', vmin=0, vmax=200, alpha=al)
  ax6.scatter(temp_grid[ind2], temp_amm15[ind2], c=dist_g[ind2], s=2, marker='o', vmin=0, vmax=200, alpha=al)
  ax7.scatter(sal_grid[ind2], sal_mgrid[ind2], c=dist_g[ind2], s=2, marker='o', vmin=0, vmax=200, alpha=al)
  ax8.scatter(sal_grid[ind2], sal_amm15[ind2], c=dist_g[ind2], s=2, marker='o', vmin=0, vmax=200, alpha=al)

  fig2.colorbar(cs1, cax=cax1)
  cax1.set_ylabel('Distance from coast (km)')

  sum_t_ssw = np.ma.count(temp_mgrid)
  sum_s_ssw = np.ma.count(sal_mgrid)
  sum_t_amm15 = np.ma.count(temp_amm15)
  sum_s_amm15 = np.ma.count(sal_amm15)

  sum_t_obs = np.ma.count(np.ma.max(temp_grid, axis=0))
  sum_s_obs = np.ma.count(np.ma.max(sal_grid, axis=0))

  print('Number Points', sum_t_ssw, sum_s_ssw, sum_t_amm15, sum_s_amm15)
  print('Number Profiles', sum_t_obs, sum_s_obs)

  def set_lims(ax, lim):
    ax.plot([-2, 40], [-2, 40], 'k-')

    ax.set_xlim(lim)
    ax.set_ylim(lim)

  set_lims(ax1, [-1, 25])
  set_lims(ax2, [-1, 25])
  set_lims(ax3, [0, 40])
  set_lims(ax4, [0, 40])
  set_lims(ax5, [-1, 25])
  set_lims(ax6, [-1, 25])
  set_lims(ax7, [0, 40])
  set_lims(ax8, [0, 40])

  ax1.set_ylabel('Temperature Model ($^{\circ}$C)')
  ax1.set_xlabel('Temperature Obs ($^{\circ}$C)')
  ax2.set_ylabel('Temperature Model ($^{\circ}$C)')
  ax2.set_xlabel('Temperature Obs ($^{\circ}$C)')

  ax3.set_ylabel('Salinity Model (g/kg)')
  ax3.set_xlabel('Salinity Obs. (g/kg)')
  ax4.set_ylabel('Salinity Model (g/kg)')
  ax4.set_xlabel('Salinity Obs. (g/kg)')

  ax5.set_ylabel('Temperature Model ($^{\circ}$C)')
  ax5.set_xlabel('Temperature Obs ($^{\circ}$C)')
  ax6.set_ylabel('Temperature Model ($^{\circ}$C)')
  ax6.set_xlabel('Temperature Obs ($^{\circ}$C)')

  ax7.set_ylabel('Salinity Model (g/kg)')
  ax7.set_xlabel('Salinity Obs. (g/kg)')
  ax8.set_ylabel('Salinity Model (g/kg)')
  ax8.set_xlabel('Salinity Obs. (g/kg)')



  ax1.annotate('(a) Winter', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2.annotate('(b) Winter', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax3.annotate('(c) Winter', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax4.annotate('(d) Winter', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax5.annotate('(e) Summer', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax6.annotate('(f) Summer', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax7.annotate('(g) Summer', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax8.annotate('(h) Summer', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax1.annotate('SSW-RS\n{:.4f}'.format(rms_tm1), (0.6, 0.05), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2.annotate('AMM15\n{:.4f}'.format(rms_ta1), (0.6, 0.05), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax3.annotate('SSW-RS\n{:.4f}'.format(rms_sm1), (0.6, 0.05), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax4.annotate('AMM15\n{:.4f}'.format(rms_sa1), (0.6, 0.08), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax5.annotate('SSW-RS\n{:.4f}'.format(rms_tm2), (0.6, 0.05), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax6.annotate('AMM15\n{:.4f}'.format(rms_ta2), (0.6, 0.05), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax7.annotate('SSW-RS\n{:.4f}'.format(rms_sm2), (0.6, 0.05), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax8.annotate('AMM15\n{:.4f}'.format(rms_sa2), (0.6, 0.05), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)




if do_vs == False:
  fig2.savefig('./Figures/all_grid_ts_2019.png')
else:
  fig3.savefig('./Figures/all_grid_vs_2019.png')

