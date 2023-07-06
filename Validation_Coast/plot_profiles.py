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

match_file = '../Validation/Indices/index.npz'
obs_file = '../Validation/Indices/prof_match.npz'
mod_file = '../Validation/Indices/profile_model.npz'
date_file = '../Validation/Indices/profile_dates.npz'

amm7_out = '/work/benbar/Validation_Data/'
amm7_file = amm7_out + 'profile_amm7_all.npz'



data = np.load(obs_file, allow_pickle=True)
prof_obs = data['prof_obs']
lat_obs = data['lat_obs']
lon_obs = data['lon_obs']
dep_obs = data['dep_obs']
temp_obs = data['temp_obs']
sal_obs = data['sal_obs']
date_obs = data['date_obs']
data.close()

temp_obs = np.ma.masked_where(temp_obs == -9999, temp_obs)
sal_obs = np.ma.masked_where(sal_obs == -9999, sal_obs)
dep_obs = np.ma.masked_where(dep_obs >= 9999, dep_obs)

data = np.load(match_file, allow_pickle=True)
o_date = data['o_date']
o_prof = data['o_prof']
data.close()

data = np.load(mod_file)
temp_mod = data['temp_mod']
sal_mod = data['sal_mod']
h_mod = data['h_mod']
data.close()

h_mod = np.ma.masked_where(h_mod == -9999, h_mod)
temp_mod = np.ma.masked_where(temp_mod == -9999, temp_mod)
sal_mod = np.ma.masked_where(sal_mod == -9999, sal_mod)
temp_mod = np.ma.masked_where(temp_obs.mask | (h_mod > 200), temp_mod)
sal_mod = np.ma.masked_where(sal_obs.mask | (h_mod > 200), sal_mod)
temp_obs = np.ma.masked_where((h_mod > 200), temp_obs)
sal_obs = np.ma.masked_where((h_mod > 200), sal_obs)

data = np.load(date_file, allow_pickle=True)
dtmp = data['d_obs']
data.close()

# Load AMM7

data = np.load(amm7_file)
temp_amm7 = data['temp_mod']
sal_amm7 = data['sal_mod']
data.close()


temp_amm7 = np.ma.masked_where(temp_amm7 == -9999, temp_amm7)
sal_amm7 = np.ma.masked_where(sal_amm7 == -9999, sal_amm7)
temp_amm7 = np.ma.masked_where(temp_obs.mask | (h_mod > 200), temp_amm7)
sal_amm7 = np.ma.masked_where(sal_obs.mask | (h_mod > 200), sal_amm7)
#temp_obs = np.ma.masked_where(temp_amm7.mask, temp_obs)
#sal_obs = np.ma.masked_where(sal_amm7.mask, sal_obs)
#temp_mod = np.ma.masked_where(temp_amm7.mask, temp_mod)
#sal_mod = np.ma.masked_where(sal_amm7.mask, sal_mod)

print(np.ma.min(temp_amm7))

# Fix o_prof and prof_obs
if 0:
  o_prof = np.zeros((len(date_obs)), dtype=int)

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



# sort data based on date

ref = dt.datetime(1990, 1, 1)
o_days = np.zeros((len(date_obs)))
ddays = np.zeros((len(dtmp)))
for i in range(len(date_obs)):
  tmp = date_obs[i] - ref
  o_days[i] = tmp.days + (tmp.seconds / (24 * 60 * 60))
for i in range(len(dtmp)):
  tmp = dtmp[i] - ref
  ddays[i] = tmp.days + (tmp.seconds / (24 * 60 * 60))

# sort based on date, profile and depth
# np.lexsort
o_sort_ind = np.argsort(o_days)
sort_ind = np.argsort(ddays)

date_obs = date_obs[o_sort_ind]
o_prof = o_prof[o_sort_ind]
lat_obs = lat_obs[o_sort_ind]
lon_obs = lon_obs[o_sort_ind]

prof_obs = prof_obs[sort_ind]
dtmp = dtmp[sort_ind]
dep_obs = dep_obs[sort_ind]
temp_obs = temp_obs[sort_ind]
sal_obs = sal_obs[sort_ind]
temp_mod = temp_mod[sort_ind]
sal_mod = sal_mod[sort_ind]
temp_amm7 = temp_amm7[sort_ind]
sal_amm7 = sal_amm7[sort_ind]


ind_d = dep_obs < 200
mnt = np.array([d.month for d in dtmp])
ind1 = ind_d * ((mnt <= 4) | (mnt >= 11))
ind2 = ind_d * ((mnt > 4) & (mnt < 11))

# Point Density Grid

t_int = 0.5
temp_range = np.arange(0, 22, t_int)

s_int = 0.5
sal_range = np.arange(10, 36.5, s_int)

to = ((temp_obs - temp_range[0]) / t_int).astype(np.int64)
tm = ((temp_mod - temp_range[0]) / t_int).astype(np.int64)
ta = ((temp_amm7 - temp_range[0]) / t_int).astype(np.int64)

so = ((sal_obs - sal_range[0]) / s_int).astype(np.int64)
sm = ((sal_mod - sal_range[0]) / s_int).astype(np.int64)
sa = ((sal_amm7 - sal_range[0]) / s_int).astype(np.int64)

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
ind1t = ind1[t_valid & s_valid]
ind2t = ind2[t_valid & s_valid]
to1 = to[ind1t]
to2 = to[ind2t]
tm1 = tm[ind1t]
tm2 = tm[ind2t]
ta1 = ta[ind1t]
ta2 = ta[ind2t]

so = so[t_valid & s_valid]
sm = sm[t_valid & s_valid]
sa = sa[t_valid & s_valid]
ind1s = ind1[t_valid & s_valid]
ind2s = ind2[t_valid & s_valid]
so1 = so[ind1s]
so2 = so[ind2s]
sm1 = sm[ind1s]
sm2 = sm[ind2s]
sa1 = sa[ind1s]
sa2 = sa[ind2s]


# Regress

#slope, intercept, r_value, p_value, std_err = stats.linregress()
rms_tm1 = mean_squared_error(temp_obs[ind1], temp_mod[ind1], squared=False)
rms_tm2 = mean_squared_error(temp_obs[ind2], temp_mod[ind2], squared=False)
rms_ta1 = mean_squared_error(temp_obs[ind1], temp_amm7[ind1], squared=False)
rms_ta2 = mean_squared_error(temp_obs[ind2], temp_amm7[ind2], squared=False)
rms_sm1 = mean_squared_error(sal_obs[ind1], sal_mod[ind1], squared=False)
rms_sm2 = mean_squared_error(sal_obs[ind2], sal_mod[ind2], squared=False)
rms_sa1 = mean_squared_error(sal_obs[ind1], sal_amm7[ind1], squared=False)
rms_sa2 = mean_squared_error(sal_obs[ind2], sal_amm7[ind2], squared=False)

rms_tm1 = rms_tm1 / np.ma.count(temp_mod[ind1])
rms_tm2 = rms_tm2 / np.ma.count(temp_mod[ind2])
rms_ta1 = rms_ta1 / np.ma.count(temp_amm7[ind1])
rms_ta2 = rms_ta2 / np.ma.count(temp_amm7[ind2])
rms_sm1 = rms_sm1 / np.ma.count(sal_mod[ind1])
rms_sm2 = rms_sm2 / np.ma.count(sal_mod[ind2])
rms_sa1 = rms_sa1 / np.ma.count(sal_amm7[ind1])
rms_sa2 = rms_sa2 / np.ma.count(sal_amm7[ind2])


#Plot


if 1:
  fig2 = plt.figure(figsize=(10,8))

  obs_grid1 = np.zeros((len(sal_range), len(temp_range)))
  obs_grid2 = np.zeros((len(sal_range), len(temp_range)))
  mod_grid1 = np.zeros((len(sal_range), len(temp_range)))
  mod_grid2 = np.zeros((len(sal_range), len(temp_range)))
  amm7_grid1 = np.zeros((len(sal_range), len(temp_range)))
  amm7_grid2 = np.zeros((len(sal_range), len(temp_range)))

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

  obs_grid1 = ((obs_grid1 / np.ma.sum(obs_grid1)) * 100).T
  obs_grid2 = ((obs_grid2 / np.ma.sum(obs_grid2)) * 100).T
  mod_grid1 = ((mod_grid1 / np.ma.sum(mod_grid1)) * 100).T
  mod_grid2 = ((mod_grid2 / np.ma.sum(mod_grid2)) * 100).T
  amm7_grid1 = ((amm7_grid1 / np.ma.sum(amm7_grid1)) * 100).T
  amm7_grid2 = ((amm7_grid2 / np.ma.sum(amm7_grid2)) * 100).T

  obs_grid1 = np.ma.masked_where(obs_grid1 == 0, obs_grid1)
  obs_grid2 = np.ma.masked_where(obs_grid2 == 0, obs_grid2)
  mod_grid1 = np.ma.masked_where(mod_grid1 == 0, mod_grid1)
  mod_grid2 = np.ma.masked_where(mod_grid2 == 0, mod_grid2)
  amm7_grid1 = np.ma.masked_where(amm7_grid1 == 0, amm7_grid1)
  amm7_grid2 = np.ma.masked_where(amm7_grid2 == 0, amm7_grid2)

  ax1 = fig2.add_axes([0.05, 0.55, 0.21, 0.35])
  ax2 = fig2.add_axes([0.36, 0.55, 0.21, 0.35])
  ax3 = fig2.add_axes([0.67, 0.55, 0.21, 0.35])
  ax4 = fig2.add_axes([0.05, 0.1, 0.21, 0.35])
  ax5 = fig2.add_axes([0.36, 0.1, 0.21, 0.35])
  ax6 = fig2.add_axes([0.67, 0.1, 0.21, 0.35])
  cax1 = fig2.add_axes([0.91, 0.1, 0.02, 0.8])


  ax1.pcolormesh(sal_range, temp_range, obs_grid1, vmin=0, vmax=2, zorder=99)
  ax2.pcolormesh(sal_range, temp_range, mod_grid1, vmin=0, vmax=2, zorder=99)
  ax3.pcolormesh(sal_range, temp_range, amm7_grid1, vmin=0, vmax=2, zorder=99)
  ax4.pcolormesh(sal_range, temp_range, obs_grid2, vmin=0, vmax=2, zorder=99)
  ax5.pcolormesh(sal_range, temp_range, mod_grid2, vmin=0, vmax=2, zorder=99)
  cs1 = ax6.pcolormesh(sal_range, temp_range, amm7_grid2, vmin=0, vmax=2, zorder=99)

  #ax1.plot(sal_obs[ind1], temp_obs[ind1], 'k.')
  #ax2.plot(sal_mod[ind1], temp_mod[ind1], 'r.')
  #ax3.plot(sal_obs[ind2], temp_obs[ind2], 'k.')
  #ax4.plot(sal_mod[ind2], temp_mod[ind2], 'r.')

  ax1.set_title('Winter')
  ax2.set_title('Winter')
  ax3.set_title('Winter')
  ax4.set_title('Summer')
  ax5.set_title('Summer')
  ax6.set_title('Summer')

  ax1.set_xlabel('Salinity Obs')
  ax4.set_xlabel('Salinity Obs')
  ax2.set_xlabel('Salinity SSW-RS')
  ax5.set_xlabel('Salinity SSW-RS')
  ax3.set_xlabel('Salinity AMM7')
  ax6.set_xlabel('Salinity AMM7')

  ax1.set_ylabel('Temperature Obs ($^{\circ}$C)')
  ax4.set_ylabel('Temperature Obs ($^{\circ}$C)')
  ax2.set_ylabel('Temperature SSW-RS ($^{\circ}$C)')
  ax5.set_ylabel('Temperature SSW-RS ($^{\circ}$C)')
  ax3.set_ylabel('Temperature AMM7 ($^{\circ}$C)')
  ax6.set_ylabel('Temperature AMM7 ($^{\circ}$C)')

  fig2.colorbar(cs1, cax=cax1, extend='max')
  cax1.set_ylabel('Points (%)')

  ax1.annotate('(a) Obs', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2.annotate('(b) SSW-RS', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax3.annotate('(c) AMM7', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax4.annotate('(d) Obs', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax5.annotate('(e) SSW-RS', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax6.annotate('(f) AMM7', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)



  fig3 = plt.figure(figsize=(10,8))

  ax1 = fig3.add_axes([0.1, 0.55, 0.35, 0.35])
  ax2 = fig3.add_axes([0.55, 0.55, 0.35, 0.35])
  ax3 = fig3.add_axes([0.1, 0.1, 0.35, 0.35])
  ax4 = fig3.add_axes([0.55, 0.1, 0.35, 0.35])
  cax1 = fig3.add_axes([0.91, 0.1, 0.02, 0.8])

  if 0:
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

  else:

    ax1.plot(temp_obs[ind1], temp_mod[ind1], '.r')
    ax1.plot(temp_obs[ind1], temp_amm7[ind1], '.g')
    ax2.plot(sal_obs[ind1], sal_mod[ind1], '.r')
    ax2.plot(sal_obs[ind1], sal_amm7[ind1], '.g')

    ax3.plot(temp_obs[ind2], temp_mod[ind2], '.r')
    ax3.plot(temp_obs[ind2], temp_amm7[ind2], '.g')
    ax4.plot(sal_obs[ind2], sal_mod[ind2], '.r')
    ax4.plot(sal_obs[ind2], sal_amm7[ind2], '.g')

  ax1.set_title('Winter')
  ax2.set_title('Winter')
  ax3.set_title('Summer')
  ax4.set_title('Summer')

  ax1.set_ylabel('Temperature Model ($^{\circ}$C)')
  ax1.set_xlabel('Temperature Obs ($^{\circ}$C)')
  ax3.set_ylabel('Temperature Model ($^{\circ}$C)')
  ax3.set_xlabel('Temperature Obs ($^{\circ}$C)')

  ax2.set_ylabel('Salinity Model')
  ax2.set_xlabel('Salinity Obs')
  ax4.set_ylabel('Salinity Model')
  ax4.set_xlabel('Salinity Obs')


  fig3.colorbar(cs1, cax=cax1, extend='max')
  cax1.set_ylabel('Points (%)')

  ax1.annotate('(a)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2.annotate('(b)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax3.annotate('(c)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax4.annotate('(d)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax1.annotate('SSW-RS={:.4f}'.format(rms_tm1), (0.05, 0.18), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax1.annotate('AMM7={:.4f}'.format(rms_ta1), (0.05, 0.08), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax2.annotate('SSW-RS={:.4f}'.format(rms_sm1), (0.05, 0.18), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2.annotate('AMM7={:.4f}'.format(rms_sa1), (0.05, 0.08), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax3.annotate('SSW-RS={:.4f}'.format(rms_tm2), (0.05, 0.18), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax3.annotate('AMM7={:.4f}'.format(rms_ta2), (0.05, 0.08), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)

  ax4.annotate('SSW-RS={:.4f}'.format(rms_sm2), (0.05, 0.18), xycoords='axes fraction', c='r', bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax4.annotate('AMM7={:.4f}'.format(rms_sa2), (0.05, 0.08), xycoords='axes fraction', c='g', bbox=dict(boxstyle="round", fc="w"), zorder=105)


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



fig2.savefig('./Figures/all_profile_ts.png')
fig3.savefig('./Figures/all_profile_vs.png')
#fig4.savefig('./Figures/all_profile_depth_t.png')
#fig5.savefig('./Figures/all_profile_depth_s.png')

