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


data = np.load(in_dir + 'forces_transect.npz', allow_pickle=True)

pgf_across = data['pgf_across']# tran, time, dist   Pressure Grad Force
pgf_along = data['pgf_along']# tran, time, dist
bsts_across = data['bsts_across']# tran, time, dist   Bottom Stress
bsts_along = data['bsts_along']# tran, time, dist
cor_across = data['cor_across']# tran, time, dist    Coriolis
cor_along = data['cor_along']# tran, time, dist

dist = data['m_dist']
date_list = data['date_list']

data.close()


pgf_across = np.ma.masked_where(pgf_across == -999, pgf_across)
pgf_along = np.ma.masked_where(pgf_along == -999, pgf_along)
bsts_across = np.ma.masked_where(bsts_across == -999, bsts_across)
bsts_along = np.ma.masked_where(bsts_along == -999, bsts_along)
cor_across = np.ma.masked_where(cor_across == -999, cor_across)
cor_along = np.ma.masked_where(cor_along == -999, cor_along)


# Plot

fig1 = plt.figure(figsize=(12, 6)) 
ax1 = fig1.add_axes([0.05, 0.74, 0.27, 0.18])
ax2 = fig1.add_axes([0.05, 0.51, 0.27, 0.18])
ax3 = fig1.add_axes([0.05, 0.28, 0.27, 0.18])
ax4 = fig1.add_axes([0.05, 0.05, 0.27, 0.18])

ax5 = fig1.add_axes([0.37, 0.74, 0.27, 0.18])
ax6 = fig1.add_axes([0.37, 0.51, 0.27, 0.18])
ax7 = fig1.add_axes([0.37, 0.28, 0.27, 0.18])
ax8 = fig1.add_axes([0.37, 0.05, 0.27, 0.18])

ax9 = fig1.add_axes([0.69, 0.74, 0.27, 0.18])
ax10 = fig1.add_axes([0.69, 0.51, 0.27, 0.18])
ax11 = fig1.add_axes([0.69, 0.28, 0.27, 0.18])
ax12 = fig1.add_axes([0.69, 0.05, 0.27, 0.18])

cax1 = fig1.add_axes([0.05, 0.98, 0.27, 0.01])
cax2 = fig1.add_axes([0.37, 0.98, 0.27, 0.01])
cax3 = fig1.add_axes([0.69, 0.98, 0.27, 0.01])


cs1 = ax1.pcolormesh(date_list, dist, pgf_across[0, :, :].T, vmin=-4e-5, vmax=4e-5)
ax2.pcolormesh(date_list, dist, pgf_across[1, :, :].T, vmin=-4e-5, vmax=4e-5)
ax3.pcolormesh(date_list, dist, pgf_across[2, :, :].T, vmin=-4e-5, vmax=4e-5)
ax4.pcolormesh(date_list, dist, pgf_across[3, :, :].T, vmin=-4e-5, vmax=4e-5)

cs2 = ax5.pcolormesh(date_list, dist, pgf_along[0, :, :].T, vmin=-4e-5, vmax=4e-5)
ax6.pcolormesh(date_list, dist, pgf_along[1, :, :].T, vmin=-4e-5, vmax=4e-5)
ax7.pcolormesh(date_list, dist, pgf_along[2, :, :].T, vmin=-4e-5, vmax=4e-5)
ax8.pcolormesh(date_list, dist, pgf_along[3, :, :].T, vmin=-4e-5, vmax=4e-5)

#cs2 = ax5.pcolormesh(date_list, dist, bsts_across[0, :, :].T, vmin=-2e-6, vmax=2e-6)
#ax6.pcolormesh(date_list, dist, bsts_across[1, :, :].T, vmin=-2e-6, vmax=2e-6)
#ax7.pcolormesh(date_list, dist, bsts_across[2, :, :].T, vmin=-2e-6, vmax=2e-6)
#ax8.pcolormesh(date_list, dist, bsts_across[3, :, :].T, vmin=-2e-6, vmax=2e-6)

cs3 = ax9.pcolormesh(date_list, dist, cor_across[0, :, :].T, vmin=-4e-5, vmax=4e-5)
ax10.pcolormesh(date_list, dist, cor_across[1, :, :].T, vmin=-4e-5, vmax=4e-5)
ax11.pcolormesh(date_list, dist, cor_across[2, :, :].T, vmin=-4e-5, vmax=4e-5)
ax12.pcolormesh(date_list, dist, cor_across[3, :, :].T, vmin=-4e-5, vmax=4e-5)


ax1.set_ylim([0, 160])
ax2.set_ylim([0, 160])
ax3.set_ylim([0, 160])
ax4.set_ylim([0, 160])

ax5.set_ylim([0, 160])
ax6.set_ylim([0, 160])
ax7.set_ylim([0, 160])
ax8.set_ylim([0, 160])

ax9.set_ylim([0, 160])
ax10.set_ylim([0, 160])
ax11.set_ylim([0, 160])
ax12.set_ylim([0, 160])

if 0:
  dlim = [dt.datetime(1999, 1, 1), dt.datetime(2000, 1, 1)]
  ax1.set_xlim(dlim)
  ax2.set_xlim(dlim)
  ax3.set_xlim(dlim)
  ax4.set_xlim(dlim)

  ax5.set_xlim(dlim)
  ax6.set_xlim(dlim)
  ax7.set_xlim(dlim)
  ax8.set_xlim(dlim)

  ax9.set_xlim(dlim)
  ax10.set_xlim(dlim)
  ax11.set_xlim(dlim)
  ax12.set_xlim(dlim)


ax1.set_ylabel('Distance (km)')
ax2.set_ylabel('Distance (km)')
ax3.set_ylabel('Distance (km)')
ax4.set_ylabel('Distance (km)')

ax1.annotate('(a) t1', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b) t2', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c) t3', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(d) t4', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax5.annotate('(e) t1', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax6.annotate('(f) t2', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax7.annotate('(g) t3', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax8.annotate('(h) t4', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax9.annotate('(i) t1', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax10.annotate('(j) t2', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax11.annotate('(k) t3', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax12.annotate('(l) t4', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)


fig1.colorbar(cs1, cax=cax1, extend='both', orientation='horizontal')
cax1.set_xlabel('Pressure Grad. Force')

fig1.colorbar(cs2, cax=cax2, extend='both', orientation='horizontal')
cax2.set_xlabel('Bottom Stress')

fig1.colorbar(cs3, cax=cax3, extend='both', orientation='horizontal')
cax3.set_xlabel('Rotation')



fig1.savefig('./Figures/forces_hov.png')

