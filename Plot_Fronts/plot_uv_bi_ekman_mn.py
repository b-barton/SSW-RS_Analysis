#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PyFVCOM.read import ncread as readFVCOM
import datetime as dt
import netCDF4 as nc
import glob
import matplotlib.tri as tri
import PyFVCOM as fvcom
import gsw as sw
import scipy.stats as stats
import scipy.signal as sig
import df_regress
from plot_uv_bivariate_mn import calc_bivar_uv


out_dir = '/scratch/benbar/Processed_Data_V3.02/'
fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'

data = np.load(out_dir + 'ekman_mn_vel.npz', allow_pickle=True)
date_ek = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
ek_ua = data['ek_u'][:, :]
ek_va = data['ek_v'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_baroclinic_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bc_ua = data['baroclin_u'][:, :]
bc_va = data['baroclin_v'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_barotropic_vel_mn.npz', allow_pickle=True)
date_bt = data['date_list'][:]
bt_ua = data['barotrop_u'][:, :]
bt_va = data['barotrop_v'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_uv_mn1.npz', allow_pickle=True)
date_uv = data['date_list'][:-12]
ua = data['ua'][:-12, :]
va = data['va'][:-12, :] # time, space
lonc = data['lonc']
latc = data['latc']
triangles = data['tri']
data.close()

data = np.load(out_dir + 'stack_uv_mn2.npz', allow_pickle=True)
date2 = data['date_list']
ua2 = data['ua']
va2 = data['va'] # time, space
data.close()

date_uv = np.append(date_uv, date2)
ua = np.append(ua, ua2, axis=0)
va = np.append(va, va2, axis=0) # time, space

print(date_ek[0], date_ek[-1])
print(len(date_ek))

bc_ua_c = fvcom.grid.nodes2elems(bc_ua, triangles)
bc_va_c = fvcom.grid.nodes2elems(bc_va, triangles)

bt_ua_c = fvcom.grid.nodes2elems(bt_ua, triangles)
bt_va_c = fvcom.grid.nodes2elems(bt_va, triangles)

ls_ua = ua - ek_ua - bc_ua_c 
ls_va = va - ek_va - bc_va_c 

loc = np.nonzero((lonc > -4.1) & (lonc < -4) & (latc > 58.7) & (latc < 58.8))[0][0]
print(loc)
fig, axs = plt.subplots(4)
axs[0].plot(date_ek, ek_ua[:, loc])
axs[0].plot(date_bc, bc_ua_c[:, loc])
axs[0].plot(date_uv, ua[:, loc])
axs[1].plot(date_ek, ek_va[:, loc])
axs[1].plot(date_bc, bc_va_c[:, loc])
axs[1].plot(date_uv, va[:, loc])
axs[2].plot(date_uv, ls_ua[:, loc])
axs[2].plot(date_bt, -bt_ua[:, loc])
axs[3].plot(date_uv, ls_va[:, loc])
axs[3].plot(date_bt, -bt_va[:, loc])

fig.savefig('./Figures/uv_part.png')



fig1 = plt.figure(figsize=(10, 14))  # size in inches
#ax1 = fig1.add_axes([0.05, 0.76, 0.35, 0.2])
#ax2 = fig1.add_axes([0.525, 0.76, 0.35, 0.2])
ax3 = fig1.add_axes([0.05, 0.52, 0.35, 0.2])
ax4 = fig1.add_axes([0.525, 0.52, 0.35, 0.2])
ax5 = fig1.add_axes([0.05, 0.28, 0.35, 0.2])
ax6 = fig1.add_axes([0.525, 0.28, 0.35, 0.2])
ax7 = fig1.add_axes([0.05, 0.04, 0.35, 0.2])
ax8 = fig1.add_axes([0.525, 0.04, 0.35, 0.2])
cax1 = fig1.add_axes([0.88, 0.14, 0.01, 0.48])

fig_name = 'vel_ekman_r_map_mn.png'
calc_bivar_uv(ek_ua, ek_va, date_ek, lonc, latc, triangles, out_dir, fig_name, [fig1, ax3, ax4, cax1])

fig_name = 'vel_baroclinic_r_map_mn.png'
calc_bivar_uv(bc_ua_c, bc_va_c, date_bc, lonc, latc, triangles, out_dir, fig_name, [fig1, ax5, ax6, cax1])

fig_name = 'vel_largescale_r_map_mn.png'
calc_bivar_uv(ls_ua, ls_va, date_bc, lonc, latc, triangles, out_dir, fig_name, [fig1, ax7, ax8, cax1])
#calc_bivar_uv(bt_ua_c, bt_va_c, date_bt, lonc, latc, triangles, out_dir, fig_name)

#ax1.annotate('(a)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
#ax2.annotate('(b)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(d)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax5.annotate('(e)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax6.annotate('(f)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax7.annotate('(g)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax8.annotate('(h)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig1.savefig('./Figures/vel_all_r_map_mn.png')

