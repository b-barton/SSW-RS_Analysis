#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime as dt
import netCDF4 as nc
import glob
import matplotlib.tri as tri
from PyFVCOM.read import ncread as readFVCOM
import gsw as sw


in_dir = '/scratch/benbar/Processed_Data_V3.02/'
match_file = in_dir + 'Indices/index_jonsis.npz'
obs_file = in_dir + 'Indices/jonsis_match.npz'
mod_file = in_dir + 'Indices/jonsis_model.npz'


data = np.load(in_dir + 'transect.npz', allow_pickle=True)

sal_tran = data['sal_tran']# tran, tb, time, dist
temp_tran = data['temp_tran']# tran, tb, time, dist
dist = data['m_dist']
date_list = data['date_list']
ua = data['ua_tran'] # tran, time, dist
va = data['va_tran']

data.close()


sal_tran = np.ma.masked_where(sal_tran == -999, sal_tran)
temp_tran = np.ma.masked_where(temp_tran == -999, temp_tran)
ua = np.ma.masked_where(ua == -999, ua)
va = np.ma.masked_where(va == -999, va)


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


def sa_from_sp(sal, dep, lat, lon):
  p = sw.p_from_z(dep, lat)
  return sw.SA_from_SP(sal ,p, lon, lat)

sal_obs1 = sa_from_sp(sal_obs, dep_obs, 56, 0)
sal_mask = np.isnan(sal_obs1) | np.isinf(sal_obs1) | (sal_obs1 > 40)
sal_obs1[sal_mask] = 0
sal_obs = np.ma.masked_where(sal_obs.mask | sal_mask, sal_obs1)

data = np.load(mod_file, allow_pickle=True)
temp_mod = data['temp_mod']
sal_mod = data['sal_mod']
data.close()

temp_mod = np.ma.masked_where((temp_mod == -9999) | (sal_mod == 0), temp_mod)
sal_mod = np.ma.masked_where((sal_mod == -9999) | (sal_mod == 0), sal_mod)
temp_obs = np.ma.masked_where(temp_mod.mask, temp_obs)
sal_obs = np.ma.masked_where(sal_mod.mask, sal_obs)

# Process

ua_res = np.ma.mean(ua, axis=1)
va_res = np.ma.mean(va, axis=1)
mag_res = (ua_res ** 2 + va_res ** 2) ** 0.5
mag = (ua ** 2 + va ** 2) ** 0.5

date_line = np.unique(date_obs)
sal_line = np.ma.zeros((len(date_line)))
sal_linem = np.ma.zeros((len(date_line)))

for i in range(len(date_line)):
  ind = (dep_obs == 0) & (date_obs == date_line[i])
  sal_line[i] = np.ma.mean(sal_obs[ind])
  sal_linem[i] = np.ma.mean(sal_mod[ind])

sal_line = sal_line[2:]
sal_linem = sal_linem[2:]
date_line = date_line[2:]

#sal_tran[4, :, :, :][sal_tran[4, :, :, :].mask] = sal_tran[5, :, :, :][sal_tran[4, :, :, :].mask] 
#temp_tran[4, :, :, :][temp_tran[4, :, :, :].mask] = temp_tran[5, :, :, :][temp_tran[4, :, :, :].mask] 

#sal_tran[4, :, :, :] = np.ma.mean(sal_tran[4:, :, :, :], axis=0)
#temp_tran[4, :, :, :] = np.ma.mean(temp_tran[4:, :, :, :], axis=0)

#sal_tran_m = sal_tran * 1
#temp_tran_m = temp_tran * 1
#for i in range(sal_tran.shape[2] - 16):
#  sal_tran_m[:, :, i + 8] = np.ma.mean(sal_tran[:, :, i:i + 16], axis=2)
#  temp_tran_m[:, :, i + 8] = np.ma.mean(temp_tran[:, :, i:i + 16], axis=2)



#sal_tran[sal_tran.mask] = sal_tran_m[sal_tran.mask]
#temp_tran[temp_tran.mask] = temp_tran_m[temp_tran.mask]

#sal_tran_mean = np.ma.mean(sal_tran, axis=1)
#for i in range(len(date_list)):
#  sal_tran[:, i, :] = sal_tran[:, i, :] - sal_tran_mean


fig1 = plt.figure(figsize=(12, 7)) 
ax1 = fig1.add_axes([0.1, 0.81, 0.83, 0.14])
ax2 = fig1.add_axes([0.1, 0.62, 0.83, 0.14])
ax3 = fig1.add_axes([0.1, 0.43, 0.83, 0.14])
ax4 = fig1.add_axes([0.1, 0.24, 0.83, 0.14])
ax5 = fig1.add_axes([0.1, 0.05, 0.83, 0.14])
ax1a = fig1.add_axes([0.05, 0.81, 0.04, 0.14])
ax2a = fig1.add_axes([0.05, 0.62, 0.04, 0.14])
ax3a = fig1.add_axes([0.05, 0.43, 0.04, 0.14])
ax4a = fig1.add_axes([0.05, 0.24, 0.04, 0.14])
cax1 = fig1.add_axes([0.94, 0.3, 0.01, 0.4])

cs1 = ax1.pcolormesh(date_list, dist, sal_tran[0, 0, :, :].T, vmin=34, vmax=35.5)
ax2.pcolormesh(date_list, dist, sal_tran[1, 0, :, :].T, vmin=34, vmax=35.5)
ax3.pcolormesh(date_list, dist, sal_tran[2, 0, :, :].T, vmin=34, vmax=35.5)
ax4.pcolormesh(date_list, dist, sal_tran[3, 0, :, :].T, vmin=34, vmax=35.5)

box_x = np.array([dt.datetime(1999, 1, 1), dt.datetime(1999, 1, 1), dt.datetime(2000, 1, 1), dt.datetime(2000, 1, 1), dt.datetime(1999, 1, 1)])
box_y = np.array([0, 160, 160, 0, 0])
ax1.plot(box_x, box_y, 'k')
ax2.plot(box_x, box_y, 'k')
ax3.plot(box_x, box_y, 'k')
ax4.plot(box_x, box_y, 'k')


ax1a.plot(mag_res[0, :], dist)
ax2a.plot(mag_res[1, :], dist)
ax3a.plot(mag_res[2, :], dist)
ax4a.plot(mag_res[3, :], dist)


ax5.plot(date_line, sal_line, '-o', lw=0.5, ms=3, label='Observations')
ax5.plot(date_line, sal_linem, '-o', lw=0.5, ms=3, label='SSW-RS')
ax5.legend(ncol=2)

ax1.set_ylim([0, 160])
ax2.set_ylim([0, 160])
ax3.set_ylim([0, 160])
ax4.set_ylim([0, 160])
ax1a.set_ylim([0, 160])
ax2a.set_ylim([0, 160])
ax3a.set_ylim([0, 160])
ax4a.set_ylim([0, 160])

ax1a.set_xlim([0, 0.25])
ax2a.set_xlim([0, 0.25])
ax3a.set_xlim([0, 0.25])
ax4a.set_xlim([0, 0.25])

ax1.set_yticklabels([])
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])

ax1a.set_ylabel('Distance (km)')
ax2a.set_ylabel('Distance (km)')
ax3a.set_ylabel('Distance (km)')
ax4a.set_ylabel('Distance (km)')
ax5.set_ylabel('Salinity (g/kg)')
ax5.set_xlim([dt.datetime(1993, 1, 1), dt.datetime(2020, 1, 1)])

ax1.annotate('(a) t1', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b) t2', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c) t3', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(d) t4', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax5.annotate('(e) JONSIS', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax1a.annotate('(m/s)', (-1.3, -0.2), xycoords='axes fraction', fontsize=12, zorder=105)
ax2a.annotate('(m/s)', (-1.3, -0.2), xycoords='axes fraction', fontsize=12, zorder=105)
ax3a.annotate('(m/s)', (-1.3, -0.2), xycoords='axes fraction', fontsize=12, zorder=105)
ax4a.annotate('(m/s)', (-1.3, -0.2), xycoords='axes fraction', fontsize=12, zorder=105)


fig1.colorbar(cs1, cax=cax1, extend='both')
cax1.set_ylabel('Salinity (g/kg)')


fig2 = plt.figure(figsize=(12, 6)) 
ax1 = fig2.add_axes([0.05, 0.77, 0.88, 0.19])
ax2 = fig2.add_axes([0.05, 0.53, 0.88, 0.19])
ax3 = fig2.add_axes([0.05, 0.29, 0.88, 0.19])
ax4 = fig2.add_axes([0.05, 0.05, 0.88, 0.19])
cax1 = fig2.add_axes([0.94, 0.3, 0.01, 0.4])

cs1 = ax1.pcolormesh(date_list, dist, sal_tran[0, 0, :, :].T, vmin=34, vmax=35.5)
ax2.pcolormesh(date_list, dist, sal_tran[1, 0, :, :].T, vmin=34, vmax=35.5)
ax3.pcolormesh(date_list, dist, sal_tran[2, 0, :, :].T, vmin=34, vmax=35.5)
ax4.pcolormesh(date_list, dist, sal_tran[3, 0, :, :].T, vmin=34, vmax=35.5)

ax1.set_ylim([0, 160])
ax2.set_ylim([0, 160])
ax3.set_ylim([0, 160])
ax4.set_ylim([0, 160])

dlim = [dt.datetime(1999, 1, 1), dt.datetime(2000, 1, 1)]
ax1.set_xlim(dlim)
ax2.set_xlim(dlim)
ax3.set_xlim(dlim)
ax4.set_xlim(dlim)

ax1.set_ylabel('Distance (km)')
ax2.set_ylabel('Distance (km)')
ax3.set_ylabel('Distance (km)')
ax4.set_ylabel('Distance (km)')

ax1.annotate('(a) t1', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b) t2', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c) t3', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(d) t4', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig2.colorbar(cs1, cax=cax1, extend='both')
cax1.set_ylabel('Salinity (g/kg)')



fig3 = plt.figure(figsize=(12, 6)) 
ax1 = fig3.add_axes([0.05, 0.81, 0.88, 0.15])
ax2 = fig3.add_axes([0.05, 0.62, 0.88, 0.15])
ax3 = fig3.add_axes([0.05, 0.43, 0.88, 0.15])
ax4 = fig3.add_axes([0.05, 0.24, 0.88, 0.15])
ax5 = fig3.add_axes([0.05, 0.05, 0.88, 0.15])
cax1 = fig3.add_axes([0.94, 0.3, 0.01, 0.4])

cs1 = ax1.pcolormesh(date_list, dist, temp_tran[0, 0, :, :].T, vmin=5, vmax=14)
ax2.pcolormesh(date_list, dist, temp_tran[1, 0, :, :].T, vmin=5, vmax=14)
ax3.pcolormesh(date_list, dist, temp_tran[2, 0, :, :].T, vmin=5, vmax=14)
ax4.pcolormesh(date_list, dist, temp_tran[3, 0, :, :].T, vmin=5, vmax=14)
ax5.pcolormesh(date_list, dist, temp_tran[4, 0, :, :].T, vmin=5, vmax=14)

ax1.set_ylim([0, 150])
ax2.set_ylim([0, 150])
ax3.set_ylim([0, 150])
ax4.set_ylim([0, 150])
ax5.set_ylim([0, 150])

ax1.set_ylabel('Distance (km)')
ax2.set_ylabel('Distance (km)')
ax3.set_ylabel('Distance (km)')
ax4.set_ylabel('Distance (km)')
ax5.set_ylabel('Distance (km)')

fig3.colorbar(cs1, cax=cax1, extend='max')
cax1.set_ylabel('Temperature ($^{\circ}$C)')


fig4 = plt.figure(figsize=(12, 6)) 
ax1 = fig4.add_axes([0.05, 0.81, 0.88, 0.15])
ax2 = fig4.add_axes([0.05, 0.62, 0.88, 0.15])
ax3 = fig4.add_axes([0.05, 0.43, 0.88, 0.15])
ax4 = fig4.add_axes([0.05, 0.24, 0.88, 0.15])
ax5 = fig4.add_axes([0.05, 0.05, 0.88, 0.15])
cax1 = fig4.add_axes([0.94, 0.3, 0.01, 0.4])

cs1 = ax1.pcolormesh(date_list, dist, temp_tran[0, 0, :, :].T - temp_tran[0, 1, :, :].T, vmin=-1, vmax=6)
ax2.pcolormesh(date_list, dist, temp_tran[1, 0, :, :].T - temp_tran[1, 1, :, :].T, vmin=-1, vmax=6)
ax3.pcolormesh(date_list, dist, temp_tran[2, 0, :, :].T - temp_tran[2, 1, :, :].T, vmin=-1, vmax=6)
ax4.pcolormesh(date_list, dist, temp_tran[3, 0, :, :].T - temp_tran[3, 1, :, :].T, vmin=-1, vmax=6)
ax5.pcolormesh(date_list, dist, temp_tran[4, 0, :, :].T - temp_tran[4, 1, :, :].T, vmin=-1, vmax=6)


ax1.set_ylim([0, 150])
ax2.set_ylim([0, 150])
ax3.set_ylim([0, 150])
ax4.set_ylim([0, 150])
ax5.set_ylim([0, 150])

ax1.set_ylabel('Distance (km)')
ax2.set_ylabel('Distance (km)')
ax3.set_ylabel('Distance (km)')
ax4.set_ylabel('Distance (km)')
ax5.set_ylabel('Distance (km)')

fig4.colorbar(cs1, cax=cax1, extend='max')
cax1.set_ylabel('Temperature ($^{\circ}$C)')


fig1.savefig('./Figures/sal_hov.png', dpi=150)
fig2.savefig('./Figures/sal_hov_short.png')
fig3.savefig('./Figures/temp_hov.png')
fig4.savefig('./Figures/temp_hov_strat.png')

