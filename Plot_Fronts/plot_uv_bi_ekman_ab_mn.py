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


data = np.load(out_dir + 'stack_baroclinic_r_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bcr_ua = data['baroclin_u'][:, :]
bcr_va = data['baroclin_v'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_baroclinic_a_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bca_ua = data['baroclin_u_a'][:, :]
bca_va = data['baroclin_v_a'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_baroclinic_b_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bcb_ua = data['baroclin_u_b'][:, :]
bcb_va = data['baroclin_v_b'][:, :] # time, space
data.close()


data = np.load(out_dir + 'stack_barosteric_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bsr_ua = data['barosteric_u'][:, :]
bsr_va = data['barosteric_v'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_barosteric_thermo_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bsa_ua = data['barosteric_u_a'][:, :]
bsa_va = data['barosteric_v_a'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_barosteric_haline_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bsb_ua = data['barosteric_u_b'][:, :]
bsb_va = data['barosteric_v_b'][:, :] # time, space
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

datafile = (out_dir 
      + '/SSWRS_V3.02_NOC_FVCOM_NWEuropeanShelf_01dy_19930101-1200_RE.nc')
dims = {'time':':10'}
vars = ('nv', 'h')
FVCOM = readFVCOM(datafile, vars, dims=dims)
h = FVCOM['h']

# scale up the thermo and halosteric heights

bsa_ua = bsa_ua * (bsr_ua / (bsa_ua + bsb_ua))
bsa_va = bsa_va * (bsr_va / (bsa_va + bsb_va))
bsb_ua = bsb_ua * (bsr_ua / (bsa_ua + bsb_ua))
bsb_va = bsb_va * (bsr_va / (bsa_va + bsb_va))

bca_ua = bca_ua * np.ma.mean((bcr_ua / (bca_ua + bcb_ua)))
bca_va = bca_va * np.ma.mean((bcr_va / (bca_va + bcb_va)))
bcb_ua = bcb_ua * np.ma.mean((bcr_ua / (bca_ua + bcb_ua)))
bcb_va = bcb_va * np.ma.mean((bcr_va / (bca_va + bcb_va)))


# mask bad data at 120 m

h = np.tile(h, (bsr_ua.shape[0], 1))
h1 = 115
h2 = 222
bsr_ua[(h >= h1) & (h < h2)] = 0
bsr_va[(h >= h1) & (h < h2)] = 0
bsa_ua[(h >= h1) & (h < h2)] = 0
bsa_va[(h >= h1) & (h < h2)] = 0
bsb_ua[(h >= h1) & (h < h2)] = 0
bsb_va[(h >= h1) & (h < h2)] = 0


bcr_ua_c = fvcom.grid.nodes2elems(bcr_ua, triangles)
bcr_va_c = fvcom.grid.nodes2elems(bcr_va, triangles)
bca_ua_c = fvcom.grid.nodes2elems(bca_ua, triangles)
bca_va_c = fvcom.grid.nodes2elems(bca_va, triangles)
bcb_ua_c = fvcom.grid.nodes2elems(bcb_ua, triangles)
bcb_va_c = fvcom.grid.nodes2elems(bcb_va, triangles)

bsr_ua_c = fvcom.grid.nodes2elems(bsr_ua, triangles)
bsr_va_c = fvcom.grid.nodes2elems(bsr_va, triangles)
bsa_ua_c = fvcom.grid.nodes2elems(bsa_ua, triangles)
bsa_va_c = fvcom.grid.nodes2elems(bsa_va, triangles)
bsb_ua_c = fvcom.grid.nodes2elems(bsb_ua, triangles)
bsb_va_c = fvcom.grid.nodes2elems(bsb_va, triangles)

bt_ua_c = fvcom.grid.nodes2elems(bt_ua, triangles)
bt_va_c = fvcom.grid.nodes2elems(bt_va, triangles)

bcr_ua = None
bcr_va = None
bca_ua = None
bca_va = None
bcb_ua = None
bcb_va = None
bsr_ua = None
bsr_va = None
bsa_ua = None
bsa_va = None
bsb_ua = None
bsb_va = None


ls_ua = ua - ek_ua - bcr_ua_c - bsr_ua_c
ls_va = va - ek_va - bcr_va_c - bsr_va_c

bca_ua_c = bca_ua_c + bsa_ua_c
bca_va_c = bca_va_c + bsa_va_c
bcb_ua_c = bcb_ua_c + bsb_ua_c
bcb_va_c = bcb_va_c + bsb_va_c


loc = np.nonzero((lonc > -4.1) & (lonc < -4) & (latc > 58.7) & (latc < 58.8))[0][0]
print(loc)
fig, axs = plt.subplots(4)
axs[0].plot(date_ek, ek_ua[:, loc])
axs[0].plot(date_bc, bcr_ua_c[:, loc])
axs[0].plot(date_uv, ua[:, loc])
axs[1].plot(date_ek, ek_va[:, loc])
axs[1].plot(date_bc, bcr_va_c[:, loc])
axs[1].plot(date_uv, va[:, loc])
axs[2].plot(date_uv, ls_ua[:, loc])
axs[2].plot(date_bt, -bt_ua[:, loc])
axs[3].plot(date_uv, ls_va[:, loc])
axs[3].plot(date_bt, -bt_va[:, loc])

fig.savefig('./Figures/uv_part.png')



fig1 = plt.figure(figsize=(8, 14))  # size in inches
#ax1 = fig1.add_axes([0.05, 0.76, 0.35, 0.2])
#ax2 = fig1.add_axes([0.525, 0.76, 0.35, 0.2])
ax3 = fig1.add_axes([0.05, 0.61, 0.35, 0.17])
ax4 = fig1.add_axes([0.525, 0.61, 0.35, 0.17])
ax5 = fig1.add_axes([0.05, 0.42, 0.35, 0.17])
ax6 = fig1.add_axes([0.525, 0.42, 0.35, 0.17])
ax7 = fig1.add_axes([0.05, 0.23, 0.35, 0.17])
ax8 = fig1.add_axes([0.525, 0.23, 0.35, 0.17])
ax9 = fig1.add_axes([0.05, 0.04, 0.35, 0.17])
ax10 = fig1.add_axes([0.525, 0.04, 0.35, 0.17])

cax1 = fig1.add_axes([0.88, 0.14, 0.01, 0.48])

fig_name = 'vel_ekman_r_map_mn.png'
calc_bivar_uv(ek_ua, ek_va, date_ek, lonc, latc, triangles, out_dir, fig_name, [fig1, ax3, ax4, cax1])

fig_name = 'vel_baroclinic_a_r_map_mn.png'
calc_bivar_uv(bca_ua_c, bca_va_c, date_bc, lonc, latc, triangles, out_dir, fig_name, [fig1, ax5, ax6, cax1])

fig_name = 'vel_baroclinic_b_r_map_mn.png'
calc_bivar_uv(bcb_ua_c, bcb_va_c, date_bc, lonc, latc, triangles, out_dir, fig_name, [fig1, ax7, ax8, cax1])

fig_name = 'vel_largescale_r_map_mn.png'
calc_bivar_uv(ls_ua, ls_va, date_bc, lonc, latc, triangles, out_dir, fig_name, [fig1, ax9, ax10, cax1])
#calc_bivar_uv(bt_ua_c, bt_va_c, date_bt, lonc, latc, triangles, out_dir, fig_name)

#ax1.annotate('(a)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
#ax2.annotate('(b)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c) Ekman', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(d) Ekman', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax5.annotate('(e) T-Driven', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax6.annotate('(f) T-Driven', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax7.annotate('(g) S-Driven', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax8.annotate('(h) S-Driven', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax9.annotate('(i) Eustatic', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax10.annotate('(j) Eustatic', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig1.savefig('./Figures/vel_all_r_map_ab_mn.png')

