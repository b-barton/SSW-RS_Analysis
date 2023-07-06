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

match_file = '../Validation/Indices/index.npz'
obs_file = '../Validation/Indices/prof_match.npz'
mod_file = '../Validation/Indices/profile_model.npz'
date_file = '../Validation/Indices/profile_dates.npz'

amm7_out = '/work/benbar/Validation_Data/'
amm7_file = amm7_out + 'profile_amm7.npz'



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

t_int = 0.25
temp_range = np.arange(2, 22, t_int)

s_int = 0.2
sal_range = np.arange(20, 36.5, s_int)

to = ((temp_obs - temp_range[0]) / t_int).astype(np.int64)
tm = ((temp_mod - temp_range[0]) / t_int).astype(np.int64)
ta = ((temp_amm7 - temp_range[0]) / t_int).astype(np.int64)

so = ((sal_obs - sal_range[0]) / s_int).astype(np.int64)
sm = ((sal_mod - sal_range[0]) / s_int).astype(np.int64)
sa = ((sal_amm7 - sal_range[0]) / s_int).astype(np.int64)

t_valid = ((to < len(temp_range)) & (to >= 0) 
          & (tm < len(temp_range)) & (tm >= 0) 
          & (ta < len(temp_range)) & (ta >= 0) 
          & np.invert(to.mask) & np.invert(tm.mask) & np.invert(ta.mask))

s_valid = ((so < len(sal_range)) & (so >= 0) 
          & (sm < len(sal_range)) & (sm >= 0) 
          & (sa < len(sal_range)) & (sa >= 0) 
          & np.invert(so.mask) & np.invert(sm.mask) & np.invert(sa.mask))

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

#Plot

fig1 = plt.figure(figsize=(10,14))

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
for i in range(0, len(o_prof), 2):
  print(i / len(o_prof) *100, '%')
  ind = prof_obs == o_prof[i]

  if i == 0:
    ax1.plot(dtmp[ind], temp_obs[ind], 'k', label='Observations')
    ax2.plot(dtmp[ind], temp_mod[ind], 'r', label='SSW-RS')
    ax3.plot(dtmp[ind], temp_amm7[ind], 'g', label='AMM7')
  else:
    ax1.plot(dtmp[ind], temp_obs[ind], 'k')
    ax2.plot(dtmp[ind], temp_mod[ind], 'r')
    ax3.plot(dtmp[ind], temp_amm7[ind], 'g')

  if i == 0:
    ax4.plot(dtmp[ind], sal_obs[ind], 'k', label='Observations')
    ax5.plot(dtmp[ind], sal_mod[ind], 'r', label='SSW-RS')
    ax6.plot(dtmp[ind], sal_amm7[ind], 'g', label='AMM7')
  else:
    ax4.plot(dtmp[ind], sal_obs[ind], 'k')
    ax5.plot(dtmp[ind], sal_mod[ind], 'r')
    ax6.plot(dtmp[ind], sal_amm7[ind], 'g')


  ntemp = ntemp + int(np.ma.count(temp_obs[ind]) > 0)
  nsal = nsal + int(np.ma.count(sal_obs[ind]) > 0)

  print(dt.datetime.today() - tnow)

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')
ax4.legend(loc='upper right')
ax5.legend(loc='upper right')
ax6.legend(loc='upper right')

ax1.set_ylabel('Temperature ($^{\circ}$C)')
ax2.set_ylabel('Temperature ($^{\circ}$C)')
ax3.set_ylabel('Temperature ($^{\circ}$C)')
ax4.set_ylabel('Salinity')
ax5.set_ylabel('Salinity')
ax6.set_ylabel('Salinity')

ax1.set_yticklabels([])
ax2.set_yticklabels([])
ax4.set_yticklabels([])
ax5.set_yticklabels([])

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


fig1.savefig('./Figures/all_profile_time_temp.png')
fig2.savefig('./Figures/all_profile_time_sal.png')

