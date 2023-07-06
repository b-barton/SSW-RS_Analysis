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

data = np.load(in_dir + 'transect.npz', allow_pickle=True)

sal_tran = data['sal_tran']# tran, tb, time, dist
date_list_sal = data['date_list']
dist_sal = data['m_dist']

data.close()

sal_tran = np.ma.masked_where(sal_tran == -999, sal_tran)


data = np.load(in_dir + 'momentum_transects.npz', allow_pickle=True)
adv = data['adv']
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

if 0:
  dind = ((date_list >= date_wind[0]) & (date_list <= date_wind[-1]))
  adv = adv[:, dind, :]
  clin_pgf = clin_pgf[:, dind, :]
  cor = cor[:, dind, :]
  trop_pgf = trop_pgf[:, dind, :]
  diff = diff[:, dind, :]
  stress = stress[:, dind, :]
  du_dt = du_dt[:, dind, :]
  div = div[:, dind, :]
  date_list = date_list[dind]

# Make into daily data
n_days = int(np.round(adv.shape[1] / 24))
adv_d = np.ma.zeros((adv.shape[0], n_days, adv.shape[2]))
clin_pgf_d = np.ma.zeros((adv.shape[0], n_days, adv.shape[2]))
cor_d = np.ma.zeros((adv.shape[0], n_days, adv.shape[2]))
trop_pgf_d = np.ma.zeros((adv.shape[0], n_days, adv.shape[2]))
diff_d = np.ma.zeros((adv.shape[0], n_days, adv.shape[2]))
stress_d = np.ma.zeros((adv.shape[0], n_days, adv.shape[2]))
wind_stress_d = np.ma.zeros((wind_stress.shape[0], n_days, wind_stress.shape[2], wind_stress.shape[3]))
bot_stress_d = np.ma.zeros((bot_stress.shape[0], n_days, wind_stress.shape[2], wind_stress.shape[3]))
du_dt_d = np.ma.zeros((adv.shape[0], n_days, adv.shape[2]))
div_d = np.ma.zeros((adv.shape[0], n_days, adv.shape[2]))
date_d = np.zeros((n_days), dtype=object)
for i in range(n_days):
  adv_d[:, i, :] = np.ma.mean(adv[:, i * 24:(i + 1) * 24, :], axis=1)
  clin_pgf_d[:, i, :] = np.ma.mean(clin_pgf[:, i * 24:(i + 1) * 24, :], axis=1)
  cor_d[:, i, :] = np.ma.mean(cor[:, i * 24:(i + 1) * 24, :], axis=1)
  trop_pgf_d[:, i, :] = np.ma.mean(trop_pgf[:, i * 24:(i + 1) * 24, :], axis=1)
  diff_d[:, i, :] = np.ma.mean(diff[:, i * 24:(i + 1) * 24, :], axis=1)
  stress_d[:, i, :] = np.ma.mean(stress[:, i * 24:(i + 1) * 24, :], axis=1)
  wind_stress_d[:, i, :, :] = np.ma.mean(wind_stress[:, i * 24:(i + 1) * 24, :, :], axis=1)
  bot_stress_d[:, i, :, :] = np.ma.mean(bot_stress[:, i * 24:(i + 1) * 24, :, :], axis=1)
  du_dt_d[:, i, :] = np.ma.mean(du_dt[:, i * 24:(i + 1) * 24, :], axis=1)
  div_d[:, i, :] = np.ma.mean(div[:, i * 24:(i + 1) * 24, :], axis=1)
  date_d[i] = date_list[i * 24]
 
print(date_list[-1])

# Split transects into bins and transects [u/v, time, transect, bin]

adv_n = np.ma.zeros((adv_d.shape[0], adv_d.shape[1], 4, 50))
clin_pgf_n = np.ma.zeros((adv_d.shape[0], adv_d.shape[1], 4, 50))
cor_n = np.ma.zeros((adv_d.shape[0], adv_d.shape[1], 4, 50))
trop_pgf_n = np.ma.zeros((adv_d.shape[0], adv_d.shape[1], 4, 50))
diff_n = np.ma.zeros((adv_d.shape[0], adv_d.shape[1], 4, 50))
stress_n = np.ma.zeros((adv_d.shape[0], adv_d.shape[1], 4, 50))
du_dt_n = np.ma.zeros((adv_d.shape[0], adv_d.shape[1], 4, 50))
div_n = np.ma.zeros((adv_d.shape[0], adv_d.shape[1], 4, 50))

for i in range(4):
  si = i * 50
  ei = (i + 1) * 50

  adv_n[:, :, i, :] = adv_d[:, :, si:ei]
  clin_pgf_n[:, :, i, :] = clin_pgf_d[:, :, si:ei]
  cor_n[:, :, i, :] = cor_d[:, :, si:ei]
  trop_pgf_n[:, :, i, :] = trop_pgf_d[:, :, si:ei]
  diff_n[:, :, i, :] = diff_d[:, :, si:ei]
  stress_n[:, :, i, :] = stress_d[:, :, si:ei]
  du_dt_n[:, :, i, :] = du_dt_d[:, :, si:ei]
  div_n[:, :, i, :] = div_d[:, :, si:ei]

# Mask bad rows
print(dist)
t_dist = np.tile(dist, (2, adv_d.shape[1], 1))
mask_dist = [(t_dist == 58), (t_dist == 70) | (t_dist == 78), t_dist == 98]
adv = np.ma.array(adv_n)
clin_pgf = np.ma.array(clin_pgf_n)
cor = np.ma.array(cor_n)
trop_pgf = np.ma.array(trop_pgf_n)
diff = np.ma.array(diff_n)
stress = np.ma.array(stress_n)
du_dt = np.ma.array(du_dt_n)
div = np.ma.array(div_n)
date_list = date_d[:]

print(t_dist.shape, adv.shape)
if 1:
  for i in range(4 - 1): # last transect does not need masking
    adv[:, :, i, :] = np.ma.masked_where(mask_dist[i], adv[:, :, i, :])
    clin_pgf[:, :, i, :] = np.ma.masked_where(mask_dist[i], clin_pgf[:, :, i, :])
    cor[:, :, i, :] = np.ma.masked_where(mask_dist[i], cor[:, :, i, :])
    trop_pgf[:, :, i, :] = np.ma.masked_where(mask_dist[i], trop_pgf[:, :, i, :])
    diff[:, :, i, :] = np.ma.masked_where(mask_dist[i], diff[:, :, i, :])
    stress[:, :, i, :] = np.ma.masked_where(mask_dist[i], stress[:, :, i, :])
    du_dt[:, :, i, :] = np.ma.masked_where(mask_dist[i], du_dt[:, :, i, :])
    div[:, :, i, :] = np.ma.masked_where(mask_dist[i], div[:, :, i, :])

    for ui in range(adv.shape[0]):
      for ti in range(adv.shape[1]):
        vsel = adv.mask[ui, ti, i, :] == 0
        ivsel = np.nonzero(vsel)[0]
        adv[ui, ti, i, :] = np.interp(
                              dist, dist[vsel], adv[ui, ti, i, ivsel])
        clin_pgf[ui, ti, i, :] = np.interp(
                              dist, dist[vsel], clin_pgf[ui, ti, i, ivsel])
        cor[ui, ti, i, :] = np.interp(
                              dist, dist[vsel], cor[ui, ti, i, ivsel])
        trop_pgf[ui, ti, i, :] = np.interp(
                              dist, dist[vsel], trop_pgf[ui, ti, i, ivsel])
        diff[ui, ti, i, :] = np.interp(
                              dist, dist[vsel], diff[ui, ti, i, ivsel])
        stress[ui, ti, i, :] = np.interp(
                              dist, dist[vsel], stress[ui, ti, i, ivsel])
        du_dt[ui, ti, i, :] = np.interp(
                              dist, dist[vsel], du_dt[ui, ti, i, ivsel])
        div[ui, ti, i, :] = np.interp(
                              dist, dist[vsel], div[ui, ti, i, ivsel])

wind_stress = wind_stress_d
bot_stress = bot_stress_d

# running mean
run = 3
adv_r = np.ma.zeros(adv.shape)
clin_pgf_r = np.ma.zeros(adv.shape)
cor_r = np.ma.zeros(adv.shape)
trop_pgf_r = np.ma.zeros(adv.shape)
diff_r = np.ma.zeros(adv.shape)
stress_r = np.ma.zeros(adv.shape)
wind_stress_r = np.ma.zeros(bot_stress_d.shape)
bot_stress_r = np.ma.zeros(bot_stress_d.shape)
du_dt_r = np.ma.zeros(adv.shape)
div_r = np.ma.zeros(adv.shape)
for i in range(adv.shape[1] - run):
  adv_r[:, i, :, :] = np.ma.mean(adv[:, i:i+run, :, :], axis=1)
  clin_pgf_r[:, i, :, :] = np.ma.mean(clin_pgf[:, i:i+run, :, :], axis=1)
  cor_r[:, i, :, :] = np.ma.mean(cor[:, i:i+run, :, :], axis=1)
  trop_pgf_r[:, i, :, :] = np.ma.mean(trop_pgf[:, i:i+run, :, :], axis=1)
  diff_r[:, i, :, :] = np.ma.mean(diff[:, i:i+run, :, :], axis=1)
  stress_r[:, i, :, :] = np.ma.mean(stress[:, i:i+run, :, :], axis=1)
  wind_stress_r[:, i, :, :] = np.ma.mean(wind_stress_d[:, i:i+run, :, :],axis=1)
  bot_stress_r[:, i, :, :] = np.ma.mean(bot_stress_d[:, i:i+run, :, :], axis=1)
  du_dt_r[:, i, :, :] = np.ma.mean(du_dt[:, i:i+run, :, :], axis=1)
  div_r[:, i, :, :] = np.ma.mean(div[:, i:i+run, :, :], axis=1)

adv[:, 1:-1, :, :] = adv_r[:, :-2, :, :]
clin_pgf[:, 1:-1, :, :] = clin_pgf_r[:, :-2, :, :]
cor[:, 1:-1, :, :] = cor_r[:, :-2, :, :]
trop_pgf[:, 1:-1, :, :] = trop_pgf_r[:, :-2, :, :]
diff[:, 1:-1, :, :] = diff_r[:, :-2, :, :]
stress[:, 1:-1, :, :] = stress_r[:, :-2, :, :]
wind_stress[:, 1:-1, :, :] = wind_stress_r[:, :-2, :, :]
bot_stress[:, 1:-1, :, :] = bot_stress_r[:, :-2, :, :]
du_dt[:, 1:-1, :, :] = du_dt_r[:, :-2, :, :]
div[:, 1:-1, :, :] = div_r[:, :-2, :, :]


# Magnitude

slope_pgf = trop_pgf - clin_pgf
trop_pgf_mag = ((trop_pgf[0, :, :, :] ** 2) + (trop_pgf[1, :, :, :] ** 2)) ** 0.5
clin_pgf_mag = ((clin_pgf[0, :, :, :] ** 2) + (clin_pgf[1, :, :, :] ** 2)) ** 0.5
slope_pgf_mag = ((slope_pgf[0, :, :, :] ** 2) + (slope_pgf[1, :, :, :] ** 2)) ** 0.5
cor_mag = ((cor[0, :, :, :] ** 2) + (cor[1, :, :, :] ** 2)) ** 0.5
stress_mag = ((stress[0, :, :, :] ** 2) + (stress[1, :, :, :] ** 2)) ** 0.5
ws_mag = ((wind_stress[0, :, :, :] ** 2) + (wind_stress[1, :, :, :] ** 2)) ** 0.5
bs_mag = ((bot_stress[0, :, :, :] ** 2) + (bot_stress[1, :, :, :] ** 2)) ** 0.5
adv_mag = ((adv[0, :, :, :] ** 2) + (adv[1, :, :, :] ** 2)) ** 0.5
print(clin_pgf_mag.shape)

bs_sub = stress_mag - ws_mag

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

fig1 = plt.figure(figsize=(12, 7)) 
ax1 = [None] * 4
ax2 = [None] * 4
ax3 = [None] * 4
ax4 = [None] * 4
ax5 = [None] * 4
ax6 = [None] * 4
ax7 = [None] * 4

ax1[0] = fig1.add_axes([0.06, 0.86, 0.185, 0.12])
ax1[1] = fig1.add_axes([0.275, 0.86, 0.185, 0.12])
ax1[2] = fig1.add_axes([0.49, 0.86, 0.185, 0.12])
ax1[3] = fig1.add_axes([0.705, 0.86, 0.185, 0.12])

ax2[0] = fig1.add_axes([0.06, 0.725, 0.185, 0.12])
ax2[1] = fig1.add_axes([0.275, 0.725, 0.185, 0.12])
ax2[2] = fig1.add_axes([0.49, 0.725, 0.185, 0.12])
ax2[3] = fig1.add_axes([0.705, 0.725, 0.185, 0.12])

ax3[0] = fig1.add_axes([0.06, 0.59, 0.185, 0.12])
ax3[1] = fig1.add_axes([0.275, 0.59, 0.185, 0.12])
ax3[2] = fig1.add_axes([0.49, 0.59, 0.185, 0.12])
ax3[3] = fig1.add_axes([0.705, 0.59, 0.185, 0.12])

ax4[0] = fig1.add_axes([0.06, 0.455, 0.185, 0.12])
ax4[1] = fig1.add_axes([0.275, 0.455, 0.185, 0.12])
ax4[2] = fig1.add_axes([0.49, 0.455, 0.185, 0.12])
ax4[3] = fig1.add_axes([0.705, 0.455, 0.185, 0.12])

ax5[0] = fig1.add_axes([0.06, 0.32, 0.185, 0.12])
ax5[1] = fig1.add_axes([0.275, 0.32, 0.185, 0.12])
ax5[2] = fig1.add_axes([0.49, 0.32, 0.185, 0.12])
ax5[3] = fig1.add_axes([0.705, 0.32, 0.185, 0.12])

ax6[0] = fig1.add_axes([0.06, 0.185, 0.185, 0.12])
ax6[1] = fig1.add_axes([0.275, 0.185, 0.185, 0.12])
ax6[2] = fig1.add_axes([0.49, 0.185, 0.185, 0.12])
ax6[3] = fig1.add_axes([0.705, 0.185, 0.185, 0.12])

ax7[0] = fig1.add_axes([0.06, 0.05, 0.185, 0.12])
ax7[1] = fig1.add_axes([0.275, 0.05, 0.185, 0.12])
ax7[2] = fig1.add_axes([0.49, 0.05, 0.185, 0.12])
ax7[3] = fig1.add_axes([0.705, 0.05, 0.185, 0.12])


cax1 = fig1.add_axes([0.895, 0.86, 0.01, 0.12])
cax2 = fig1.add_axes([0.895, 0.725, 0.01, 0.12])
cax3 = fig1.add_axes([0.895, 0.59, 0.01, 0.12])
cax4 = fig1.add_axes([0.895, 0.455, 0.01, 0.12])
cax5 = fig1.add_axes([0.895, 0.32, 0.01, 0.12])
cax6 = fig1.add_axes([0.895, 0.185, 0.01, 0.12])
cax7 = fig1.add_axes([0.895, 0.05, 0.01, 0.12])

ann_list1 = ['(a) t1', '(a) t2', '(a) t3', '(a) t4']
ann_list2 = ['(b) t1', '(b) t2', '(b) t3', '(b) t4']
ann_list3 = ['(c) t1', '(c) t2', '(c) t3', '(c) t4']
ann_list4 = ['(d) t1', '(d) t2', '(d) t3', '(d) t4']
ann_list5 = ['(e) t1', '(e) t2', '(e) t3', '(e) t4']
ann_list6 = ['(f) t1', '(f) t2', '(f) t3', '(f) t4']
ann_list7 = ['(g) t1', '(g) t2', '(g) t3', '(g) t4']

mag = 1
for i in range(4):

  cs1 = ax1[i].pcolormesh(date_list_sal, dist_sal, sal_tran[i, 0, :, :].T, vmin=34, vmax=35.5)

  #ax2[i].contour(date_list_sal, dist_sal, sal_tran[i, 0, :, :].T, [35], colors='k', linewidths=0.3, zorder=100)
  #ax3[i].contour(date_list_sal, dist_sal, sal_tran[i, 0, :, :].T, [35], colors='k', linewidths=0.3, zorder=100)
  #ax4[i].contour(date_list_sal, dist_sal, sal_tran[i, 0, :, :].T, [35], colors='k', linewidths=0.3, zorder=100)
  #ax5[i].contour(date_list_sal, dist_sal, sal_tran[i, 0, :, :].T, [35], colors='k', linewidths=0.3, zorder=100)
  #ax6[i].contour(date_list_sal, dist_sal, sal_tran[i, 0, :, :].T, [35], colors='k', linewidths=0.3, zorder=100)

  if mag == 1:
    cs2 = ax2[i].pcolormesh(date_list, dist, cor_mag[:, i, :].T * 1000, vmin=0, vmax=5e-2)  
    cs3 = ax3[i].pcolormesh(date_list, dist, trop_pgf_mag[:, i, :].T * 1000, vmin=0, vmax=5e-2)
    cs4 = ax4[i].pcolormesh(date_list, dist, clin_pgf_mag[:, i, :].T * 1000, vmin=0, vmax=5e-2)
    cs5 = ax5[i].pcolormesh(date_list, dist, ws_mag[:, i, :].T * 1000, vmin=0, vmax=5e-2)
    cs6 = ax6[i].pcolormesh(date_list, dist, adv_mag[:, i, :].T * 1000, vmin=0, vmax=5e-2)
    cs7 = ax7[i].pcolormesh(date_list, dist, bs_sub[:, i, :].T * 1000, vmin=0, vmax=5e-2)

  elif mag == 2:
    cs2 = ax2[i].pcolormesh(date_list, dist, cor_across[:, i, :].T * 1000, vmin=-5e-2, vmax=5e-2)  
    cs3 = ax3[i].pcolormesh(date_list, dist, trop_across[:, i, :].T * 1000, vmin=-5e-2, vmax=5e-2)
    cs4 = ax4[i].pcolormesh(date_list, dist, clin_across[:, i, :].T * 1000, vmin=-2e-2, vmax=2e-2)
    cs5 = ax5[i].pcolormesh(date_list, dist, stress_across[:, i, :].T * 1000, vmin=-2e-2, vmax=2e-2)
    cs6 = ax6[i].pcolormesh(date_list, dist, adv_across[:, i, :].T * 1000, vmin=-2e-2, vmax=2e-2)

  else:
    cs2 = ax2[i].pcolormesh(date_list, dist, cor_along[:, i, :].T * 1000, vmin=-5e-2, vmax=5e-2)  
    cs3 = ax3[i].pcolormesh(date_list, dist, trop_along[:, i, :].T * 1000, vmin=-5e-2, vmax=5e-2)
    cs4 = ax4[i].pcolormesh(date_list, dist, clin_along[:, i, :].T * 1000, vmin=-2e-2, vmax=2e-2)
    cs5 = ax5[i].pcolormesh(date_list, dist, stress_along[:, i, :].T * 1000, vmin=-2e-2, vmax=2e-2)
    cs6 = ax6[i].pcolormesh(date_list, dist, adv_along[:, i, :].T * 1000, vmin=-2e-2, vmax=2e-2)


  ax1[i].set_ylim([0, 160])
  ax2[i].set_ylim([0, 160])
  ax3[i].set_ylim([0, 160])
  ax4[i].set_ylim([0, 160])
  ax5[i].set_ylim([0, 160])
  ax6[i].set_ylim([0, 160])
  ax7[i].set_ylim([0, 160])

  ax1[i].annotate(ann_list1[i], (0.03, 0.83), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax2[i].annotate(ann_list2[i], (0.03, 0.83), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax3[i].annotate(ann_list3[i], (0.03, 0.83), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax4[i].annotate(ann_list4[i], (0.03, 0.83), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax5[i].annotate(ann_list5[i], (0.03, 0.83), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax6[i].annotate(ann_list6[i], (0.03, 0.83), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
  ax7[i].annotate(ann_list7[i], (0.03, 0.83), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

  if i != 0:
    ax1[i].set_yticklabels([])
    ax2[i].set_yticklabels([])
    ax3[i].set_yticklabels([])
    ax4[i].set_yticklabels([])
    ax5[i].set_yticklabels([])
    ax6[i].set_yticklabels([])
    ax7[i].set_yticklabels([])

  ax1[i].set_xticks([])
  ax2[i].set_xticks([])
  ax3[i].set_xticks([])
  ax4[i].set_xticks([])
  ax5[i].set_xticks([])
  ax6[i].set_xticks([])

  ax7[i].xaxis.set_major_locator(mdates.MonthLocator(interval=2))   
  ax7[i].xaxis.set_major_formatter(mdates.DateFormatter('%b')) 

  if 1:
    dlim = [dt.datetime(1999, 1, 1), dt.datetime(2000, 1, 1)]
    ax1[i].set_xlim(dlim)
    ax2[i].set_xlim(dlim)
    ax3[i].set_xlim(dlim)
    ax4[i].set_xlim(dlim)
    ax5[i].set_xlim(dlim)
    ax6[i].set_xlim(dlim)
    ax7[i].set_xlim(dlim)


ax1[0].set_ylabel('Distance\n(km)')
ax2[0].set_ylabel('Distance\n(km)')
ax3[0].set_ylabel('Distance\n(km)')
ax4[0].set_ylabel('Distance\n(km)')
ax5[0].set_ylabel('Distance\n(km)')
ax6[0].set_ylabel('Distance\n(km)')
ax7[0].set_ylabel('Distance\n(km)')


mticks = [-0.05, 0, 0.05]
fig1.colorbar(cs1, cax=cax1, extend='both')
cax1.set_ylabel('Salinity (g/kg)')

fig1.colorbar(cs2, cax=cax2, extend='both')#, ticks=mticks)
cax2.set_ylabel('Coriolis\nForce\n(m/s$^{2}$ $\\times10^{-3}$)')

fig1.colorbar(cs3, cax=cax3, extend='both')#, ticks=mticks)
cax3.set_ylabel('Barotropic\nPGF\n(m/s$^{2}$ $\\times10^{-3}$)')

fig1.colorbar(cs4, cax=cax4, extend='both')
cax4.set_ylabel('Baroclinic\nPGF\n(m/s$^{2}$ $\\times10^{-3}$)')

fig1.colorbar(cs5, cax=cax5, extend='both')
cax5.set_ylabel('Wind\nStress\n(m/s$^{2}$ $\\times10^{-3}$)')

fig1.colorbar(cs6, cax=cax6, extend='both')
cax6.set_ylabel('Advection\n(m/s$^{2}$ $\\times10^{-3}$)')

fig1.colorbar(cs7, cax=cax7, extend='both')
cax7.set_ylabel('Bottom\nStrees\n(m/s$^{2}$ $\\times10^{-3}$)')


if mag == 1:
  fig1.savefig('./Figures/momentum_hov.png', dpi=150)
elif mag == 2:
  fig1.savefig('./Figures/momentum_hov_across.png', dpi=150)
else:
  fig1.savefig('./Figures/momentum_hov_along.png', dpi=150)



