#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime as dt
import matplotlib.dates as mdates
import netCDF4 as nc
import glob
import matplotlib.tri as tri
from PyFVCOM.read import ncread as readFVCOM
import PyFVCOM as fvcom


in_dir = '/scratch/benbar/Processed_Data_V3.02/'
fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'


data = np.load(in_dir + 'momentum_transects.npz', allow_pickle=True)
adv = data['adv']
adf = data['adf']
clin_pgf = data['clin_pgf'] # baroclinic [u/v, time, elem]
cor = data['cor'] # coriolis
trop_pgf = data['trop_pgf'] # barotropic
diff = data['diff'] # diffusion
stress = data['stress'] # surface + bottom
du_dt = data['du_dt']
div = data['div']
date_list = data['date_list']
data.close()

# du_dt = -trop_pgf_x + cor_x + stress_u + diff_x + adv_u

data = np.load(in_dir + 'wind_transects.npz', allow_pickle=True)
wind_stress = data['wind_stress']
bot_stress = data['bot_stress']
date_wind = data['date_list']
data.close()

data = np.load(in_dir + 'elem_transect.npz')
dist = data['m_dist']
latt = data['latt']
lont = data['lont']
elem_list = data['elem_list'] # transect, bin
data.close()

print(date_list[-1])

st_date = dt.datetime(1992, 12, 30, 0, 0, 0)
en_date = dt.datetime(1993, 1, 31, 23, 0, 0)
fvg = fvcom.preproc.Model(st_date, en_date, grid=fgrd, 
                    native_coordinates='spherical', zone='30N')
xc = fvg.grid.xc
yc = fvg.grid.yc

# Match times

dind = date_wind != 0
wind_stress = wind_stress[:, dind, :, :]
bot_stress = bot_stress[:, dind, :, :]
date_wind = date_wind[dind]

dind = ((date_wind >= date_list[0]) & (date_wind <= date_list[-1]))
wind_stress = wind_stress[:, dind, :, :]
bot_stress = bot_stress[:, dind, :, :]
date_wind = date_wind[dind]

if 1:
  dind = ((date_list >= date_wind[0]) & (date_list <= date_wind[-1]))
  adv = adv[:, dind, :]
  adf = adf[:, dind, :]
  clin_pgf = clin_pgf[:, dind, :]
  cor = cor[:, dind, :]
  trop_pgf = trop_pgf[:, dind, :]
  diff = diff[:, dind, :]
  stress = stress[:, dind, :]
  du_dt = du_dt[:, dind, :]
  div = div[:, dind, :]
  date_list = date_list[dind]

print(date_list[-1])

# Split transects into bins and transects [u/v, time, transect, bin]

adv_n = np.ma.zeros((adv.shape[0], adv.shape[1], 4, 50))
adf_n = np.ma.zeros((adv.shape[0], adv.shape[1], 4, 50))
clin_pgf_n = np.ma.zeros((adv.shape[0], adv.shape[1], 4, 50))
cor_n = np.ma.zeros((adv.shape[0], adv.shape[1], 4, 50))
trop_pgf_n = np.ma.zeros((adv.shape[0], adv.shape[1], 4, 50))
diff_n = np.ma.zeros((adv.shape[0], adv.shape[1], 4, 50))
stress_n = np.ma.zeros((adv.shape[0], adv.shape[1], 4, 50))
du_dt_n = np.ma.zeros((adv.shape[0], adv.shape[1], 4, 50))
div_n = np.ma.zeros((adv.shape[0], adv.shape[1], 4, 50))

for i in range(4):
  si = i * 50
  ei = (i + 1) * 50

  adv_n[:, :, i, :] = adv[:, :, si:ei]
  adf_n[:, :, i, :] = adf[:, :, si:ei]
  clin_pgf_n[:, :, i, :] = clin_pgf[:, :, si:ei]
  cor_n[:, :, i, :] = cor[:, :, si:ei]
  trop_pgf_n[:, :, i, :] = trop_pgf[:, :, si:ei]
  diff_n[:, :, i, :] = diff[:, :, si:ei]
  stress_n[:, :, i, :] = stress[:, :, si:ei]
  du_dt_n[:, :, i, :] = du_dt[:, :, si:ei]
  div_n[:, :, i, :] = div[:, :, si:ei]

# Mask bad rows
print(dist)
t_dist = np.tile(dist, (2, adv.shape[1], 1))
mask_dist = [(t_dist == 58), (t_dist == 70) | (t_dist == 78), t_dist == 98]
adv = np.ma.array(adv_n)
adf = np.ma.array(adf_n)
clin_pgf = np.ma.array(clin_pgf_n)
cor = np.ma.array(cor_n)
trop_pgf = np.ma.array(trop_pgf_n)
diff = np.ma.array(diff_n)
stress = np.ma.array(stress_n)
du_dt = np.ma.array(du_dt_n)
div = np.ma.array(div_n)

print(t_dist.shape, adv.shape)
for i in range(4 - 1): # last transect does not need masking
  adv[:, :, i, :] = np.ma.masked_where(mask_dist[i], adv[:, :, i, :])
  adf[:, :, i, :] = np.ma.masked_where(mask_dist[i], adf[:, :, i, :])
  clin_pgf[:, :, i, :] = np.ma.masked_where(mask_dist[i], clin_pgf[:, :, i, :])
  cor[:, :, i, :] = np.ma.masked_where(mask_dist[i], cor[:, :, i, :])
  trop_pgf[:, :, i, :] = np.ma.masked_where(mask_dist[i], trop_pgf[:, :, i, :])
  diff[:, :, i, :] = np.ma.masked_where(mask_dist[i], diff[:, :, i, :])
  stress[:, :, i, :] = np.ma.masked_where(mask_dist[i], stress[:, :, i, :])
  du_dt[:, :, i, :] = np.ma.masked_where(mask_dist[i], du_dt[:, :, i, :])
  div[:, :, i, :] = np.ma.masked_where(mask_dist[i], div[:, :, i, :])

# Magnitude

slope_pgf = trop_pgf - clin_pgf
clin_pgf_mag = ((clin_pgf[0, :, :, :] ** 2) + (clin_pgf[1, :, :, :] ** 2)) ** 0.5
slope_pgf_mag = ((slope_pgf[0, :, :, :] ** 2) + (slope_pgf[1, :, :, :] ** 2)) ** 0.5
cor_mag = ((cor[0, :, :, :] ** 2) + (cor[1, :, :, :] ** 2)) ** 0.5
ws_mag = ((wind_stress[0, :, :, :] ** 2) + (wind_stress[1, :, :, :] ** 2)) ** 0.5
adv_mag = ((adv[0, :, :, :] ** 2) + (adv[1, :, :, :] ** 2)) ** 0.5
adf_mag = ((adf[0, :, :, :] ** 2) + (adf[1, :, :, :] ** 2)) ** 0.5
diff_mag = ((diff[0, :, :, :] ** 2) + (diff[1, :, :, :] ** 2)) ** 0.5
div_mag = ((div[0, :, :, :] ** 2) + (div[1, :, :, :] ** 2)) ** 0.5

print(clin_pgf_mag.shape)

# Along and across transect components

# Get x and y values of the transects

xc_tran = xc[elem_list]
yc_tran = yc[elem_list]

# Get angle of each point on the transect
if 1:
  dx = xc_tran[:, -1] - xc_tran[:, 0]
  dy = yc_tran[:, -1] - yc_tran[:, 0]

  angle = np.arctan2(dx, dy) # from North

  # Align components to be along and across the transect

  def align(x_var, y_var, angle):
    across_var = np.zeros_like(x_var)
    along_var = np.zeros_like(x_var)

    for i in range(x_var.shape[1]): 
      var_mag = ((x_var[:, i, :] ** 2) + (y_var[:, i, :] ** 2)) ** 0.5
      var_dir = np.arctan2(x_var[:, i, :], y_var[:, i, :])
      new_dir = var_dir - angle[i] # subtract so North lies along transect
      across_var[:, i, :] = var_mag * np.sin(new_dir) # x direction
      along_var[:, i, :] = var_mag * np.cos(new_dir) # y direction
    return across_var, along_var

  adv_across, adv_along = align(adv[0, ...], adv[1, ...], angle)
  adf_across, adf_along = align(adf[0, ...], adf[1, ...], angle)
  clin_across, clin_along = align(clin_pgf[0, ...], clin_pgf[1, ...], angle)
  cor_across, cor_along = align(cor[0, ...], cor[1, ...], angle)
  trop_across, trop_along = align(trop_pgf[0, ...], trop_pgf[1, ...], angle)
  slope_across, slope_along = align(slope_pgf[0, ...], slope_pgf[1, ...], angle)

  diff_across, diff_along = align(diff[0, ...], diff[1, ...], angle)
  stress_across, stress_along = align(stress[0, ...], stress[1, ...], angle)
  ws_across, ws_along = align(wind_stress[0, ...], wind_stress[1, ...], angle)
  du_dt_across, du_dt_along = align(du_dt[0, ...], du_dt[1, ...], angle)
  div_across, div_along = align(div[0, ...], div[1, ...], angle)



# Plot

fig1 = plt.figure(figsize=(12, 6)) 
ax1 = [None] * 4
ax2 = [None] * 4
ax3 = [None] * 4
ax4 = [None] * 4

ax1[0] = fig1.add_axes([0.05, 0.695, 0.22, 0.185])
ax1[1] = fig1.add_axes([0.05, 0.48, 0.22, 0.185])
ax1[2] = fig1.add_axes([0.05, 0.265, 0.22, 0.185])
ax1[3] = fig1.add_axes([0.05, 0.05, 0.22, 0.185])

ax2[0] = fig1.add_axes([0.2875, 0.695, 0.22, 0.185])
ax2[1] = fig1.add_axes([0.2875, 0.48, 0.22, 0.185])
ax2[2] = fig1.add_axes([0.2875, 0.265, 0.22, 0.185])
ax2[3] = fig1.add_axes([0.2875, 0.05, 0.22, 0.185])

ax3[0] = fig1.add_axes([0.525, 0.695, 0.22, 0.185])
ax3[1] = fig1.add_axes([0.525, 0.48, 0.22, 0.185])
ax3[2] = fig1.add_axes([0.525, 0.265, 0.22, 0.185])
ax3[3] = fig1.add_axes([0.525, 0.05, 0.22, 0.185])

ax4[0] = fig1.add_axes([0.7625, 0.695, 0.22, 0.185])
ax4[1] = fig1.add_axes([0.7625, 0.48, 0.22, 0.185])
ax4[2] = fig1.add_axes([0.7625, 0.265, 0.22, 0.185])
ax4[3] = fig1.add_axes([0.7625, 0.05, 0.22, 0.185])

cax1 = fig1.add_axes([0.05, 0.98, 0.22, 0.01])
cax2 = fig1.add_axes([0.2875, 0.98, 0.22, 0.01])
cax3 = fig1.add_axes([0.525, 0.98, 0.22, 0.01])
cax4 = fig1.add_axes([0.7625, 0.98, 0.22, 0.01])

ann_list1 = ['(a) t1', '(b) t2', '(c) t3', '(d) t4']
ann_list2 = ['(e) t1', '(f) t2', '(g) t3', '(h) t4']
ann_list3 = ['(i) t1', '(j) t2', '(k) t3', '(l) t4']
ann_list4 = ['(m) t1', '(n) t2', '(o) t3', '(p) t4']

mag = 2
for i in range(4):

  if mag == 1:
    cs1 = ax1[i].pcolormesh(date_list, dist, adv_mag[:, i, :].T * 1000, vmin=0, vmax=2e-2)  
    cs2 = ax2[i].pcolormesh(date_list, dist, adf_mag[:, i, :].T * 1000, vmin=0, vmax=2e-4)

    cs3 = ax3[i].pcolormesh(date_list, dist, diff_mag[:, i, :].T * 1000, vmin=0, vmax=3e-3)

    cs4 = ax4[i].pcolormesh(date_list, dist, div_mag[:, i, :].T * 1000, vmin=0, vmax=3e1)

  elif mag == 2:
    cs1 = ax1[i].pcolormesh(date_list, dist, adv_across[:, i, :].T * 1000, vmin=-1e-1, vmax=1e-1)  
    cs2 = ax2[i].pcolormesh(date_list, dist, adf_across[:, i, :].T * 1000, vmin=-1e-1, vmax=1e-1)

    cs3 = ax3[i].pcolormesh(date_list, dist, diff_across[:, i, :].T * 1000, vmin=-3e-2, vmax=3e-2)

    cs4 = ax4[i].pcolormesh(date_list, dist, div_across[:, i, :].T * 1000, vmin=-3e-2, vmax=3e-2)
  else:
    cs1 = ax1[i].pcolormesh(date_list, dist, adv_along[:, i, :].T * 1000, vmin=-1e-2, vmax=1e-2)  
    cs2 = ax2[i].pcolormesh(date_list, dist, adf_along[:, i, :].T * 1000, vmin=-1e-4, vmax=1e-4)

    cs3 = ax3[i].pcolormesh(date_list, dist, diff_along[:, i, :].T * 1000, vmin=-1e-3, vmax=1e-3)

    cs4 = ax4[i].pcolormesh(date_list, dist, div_along[:, i, :].T * 1000, vmin=-1e1, vmax=1e1)


  ax1[i].set_ylim([0, 160])
  ax2[i].set_ylim([0, 160])
  ax3[i].set_ylim([0, 160])
  ax4[i].set_ylim([0, 160])

  ax1[i].annotate(ann_list1[i], (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2[i].annotate(ann_list2[i], (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax3[i].annotate(ann_list3[i], (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax4[i].annotate(ann_list4[i], (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

  if i != 3:
    ax1[i].set_xticks([])
    ax2[i].set_xticks([])
    ax3[i].set_xticks([])
    ax4[i].set_xticks([])
  else:
    ax1[i].xaxis.set_major_locator(mdates.MonthLocator(interval=2))   
    ax1[i].xaxis.set_major_formatter(mdates.DateFormatter('%b'))     
    ax2[i].xaxis.set_major_locator(mdates.MonthLocator(interval=2))   
    ax2[i].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax3[i].xaxis.set_major_locator(mdates.MonthLocator(interval=2))   
    ax3[i].xaxis.set_major_formatter(mdates.DateFormatter('%b')) 
    ax4[i].xaxis.set_major_locator(mdates.MonthLocator(interval=2))   
    ax4[i].xaxis.set_major_formatter(mdates.DateFormatter('%b')) 

  ax2[i].set_yticklabels([])
  ax3[i].set_yticklabels([])
  ax4[i].set_yticklabels([])


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


ax1[0].set_ylabel('Distance (km)')
ax1[1].set_ylabel('Distance (km)')
ax1[2].set_ylabel('Distance (km)')
ax1[3].set_ylabel('Distance (km)')



fig1.colorbar(cs1, cax=cax1, extend='both', orientation='horizontal')
cax1.set_xlabel('Advection (ms$^{-2}$ $\\times10^{-3}$)')

fig1.colorbar(cs2, cax=cax2, extend='both', orientation='horizontal')
cax2.set_xlabel('Diffusion (ms$^{-2}$ $\\times10^{-3}$)')

fig1.colorbar(cs3, cax=cax3, extend='both', orientation='horizontal')
cax3.set_xlabel('G (ms$^{-2}$ $\\times10^{-3}$)')

fig1.colorbar(cs4, cax=cax4, extend='both', orientation='horizontal')
cax4.set_xlabel('Div (ms$^{-2}$ $\\times10^{-3}$)')


if mag == 1:
  fig1.savefig('./Figures/momentum_hov_other.png')
elif mag == 2:
  fig1.savefig('./Figures/momentum_hov_other_across.png')
else:
  fig1.savefig('./Figures/momentum_hov_other_along.png')



