#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime as dt
import netCDF4 as nc
import glob
import matplotlib.tri as tri
from PyFVCOM.read import ncread as readFVCOM


in_dir = '/scratch/benbar/Processed_Data_V3.02/'


data = np.load(in_dir + 'transect_hr.npz', allow_pickle=True)

sal_tran = data['sal_tran']# tran, tb, time, dist
temp_tran = data['temp_tran']# tran, tb, time, dist
dist = data['m_dist']
date_list = data['date_list']

data.close()


sal_tran = np.ma.masked_where(sal_tran == -999, sal_tran)
temp_tran = np.ma.masked_where(temp_tran == -999, temp_tran)

for i in range(sal_tran.shape[3]):
  bool_s = sal_tran[4, :, :, i] == sal_tran[4, :, :, -1]
  sal_tran[4, :, :, i] = np.ma.masked_where(bool_s, sal_tran[4, :, :, i])

  bool_s = sal_tran[5, :, :, i] == sal_tran[5, :, :, 0]
  sal_tran[5, :, :, i] = np.ma.masked_where(bool_s, sal_tran[5, :, :, i])

#sal_tran[4, :, :, :] = np.ma.mean(sal_tran[4:, :, :, :], axis=0)
#temp_tran[4, :, :, :] = np.ma.mean(temp_tran[4:, :, :, :], axis=0)



fig1 = plt.figure(figsize=(12, 6)) 
ax1 = fig1.add_axes([0.05, 0.81, 0.88, 0.15])
ax2 = fig1.add_axes([0.05, 0.62, 0.88, 0.15])
ax3 = fig1.add_axes([0.05, 0.43, 0.88, 0.15])
ax4 = fig1.add_axes([0.05, 0.24, 0.88, 0.15])
ax5 = fig1.add_axes([0.05, 0.05, 0.88, 0.15])
cax1 = fig1.add_axes([0.94, 0.3, 0.01, 0.4])

cs1 = ax1.pcolormesh(date_list, dist, sal_tran[0, 0, :, :].T, vmin=34, vmax=35.5)
ax2.pcolormesh(date_list, dist, sal_tran[1, 0, :, :].T, vmin=34, vmax=35.5)
ax3.pcolormesh(date_list, dist, sal_tran[2, 0, :, :].T, vmin=34, vmax=35.5)
ax4.pcolormesh(date_list, dist, sal_tran[3, 0, :, :].T, vmin=34, vmax=35.5)
ax5.pcolormesh(date_list, dist, sal_tran[4, 0, :, :].T, vmin=34, vmax=35.5)

ax1.set_ylim([0, 200])
ax2.set_ylim([0, 200])
ax3.set_ylim([0, 200])
ax4.set_ylim([0, 200])
ax5.set_ylim([0, 200])

dlim = [dt.datetime(2000, 1, 1), dt.datetime(2001, 1, 1)]
ax1.set_xlim(dlim)
ax2.set_xlim(dlim)
ax3.set_xlim(dlim)
ax4.set_xlim(dlim)
ax5.set_xlim(dlim)

ax1.set_ylabel('Distance (km)')
ax2.set_ylabel('Distance (km)')
ax3.set_ylabel('Distance (km)')
ax4.set_ylabel('Distance (km)')
ax5.set_ylabel('Distance (km)')

fig1.colorbar(cs1, cax=cax1, extend='max')
cax1.set_ylabel('Salinity')


fig2 = plt.figure(figsize=(12, 6)) 
ax1 = fig2.add_axes([0.05, 0.81, 0.88, 0.15])
ax2 = fig2.add_axes([0.05, 0.62, 0.88, 0.15])
ax3 = fig2.add_axes([0.05, 0.43, 0.88, 0.15])
ax4 = fig2.add_axes([0.05, 0.24, 0.88, 0.15])
ax5 = fig2.add_axes([0.05, 0.05, 0.88, 0.15])
cax1 = fig2.add_axes([0.94, 0.3, 0.01, 0.4])

cs1 = ax1.pcolormesh(date_list, dist, sal_tran[0, 0, :, :].T - sal_tran[0, 1, :, :].T, vmin=-2, vmax=1)
ax2.pcolormesh(date_list, dist, sal_tran[1, 0, :, :].T - sal_tran[1, 1, :, :].T, vmin=-2, vmax=1)
ax3.pcolormesh(date_list, dist, sal_tran[2, 0, :, :].T - sal_tran[2, 1, :, :].T, vmin=-2, vmax=1)
ax4.pcolormesh(date_list, dist, sal_tran[3, 0, :, :].T - sal_tran[3, 1, :, :].T, vmin=-2, vmax=1)
ax5.pcolormesh(date_list, dist, sal_tran[4, 0, :, :].T - sal_tran[4, 1, :, :].T, vmin=-2, vmax=1)

ax1.set_ylim([0, 150])
ax2.set_ylim([0, 150])
ax3.set_ylim([0, 150])
ax4.set_ylim([0, 150])
ax5.set_ylim([0, 150])

ax1.set_ylabel('Distance (km)')
ax2.set_ylabel('Distance (km)')
ax3.set_ylabel('Distance (km)')
ax4.set_ylabel('Distance (km)')

fig2.colorbar(cs1, cax=cax1, extend='max')
cax1.set_ylabel('Salinity')



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
ax5.pcolormesh(date_list, dist, temp_tran[5, 0, :, :].T, vmin=5, vmax=14)

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


fig1.savefig('./Figures/sal_hov_hr.png', dpi=150)
fig2.savefig('./Figures/sal_hov_strat_hr.png')
fig3.savefig('./Figures/temp_hov_hr.png')
fig4.savefig('./Figures/temp_hov_strat_hr.png')

