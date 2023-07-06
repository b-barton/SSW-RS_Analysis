#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PyFVCOM.read import MFileReader
from PyFVCOM.plot import Time
from PyFVCOM.grid import unstructured_grid_depths
import PyFVCOM.plot as fvplot
from PyFVCOM.read import ncread as readFVCOM
import glob
import netCDF4 as nc
import datetime as dt
import properscoring as ps
from mpl_toolkits.basemap import Basemap

match_file = './Indices/index.npz'
obs_file = './Indices/prof_match.npz'
mod_file = './Indices/profile_model.npz'

fname = '/scratch/Scottish_waters_FVCOM/SSW_RS/'
folder1 = 'SSW_Reanalysis_v1.1_1995_02'
datafile = (fname + folder1 
    + '/SSWRS_V1.1_NOC_FVCOM_NWEuropeanShelf_01dy_19950220-1200_RE.nc')



data = np.load(obs_file, allow_pickle=True)
prof_obs = data['prof_obs'] # wrong
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

data = np.load(mod_file)
temp_mod = data['temp_mod']
sal_mod = data['sal_mod']
data.close()

temp_mod = np.ma.masked_where(temp_mod == -9999, temp_mod)
sal_mod = np.ma.masked_where(sal_mod == -9999, sal_mod)
temp_mod = np.ma.masked_where(temp_obs.mask, temp_mod)
sal_mod = np.ma.masked_where(sal_obs.mask, sal_mod)
temp_obs = np.ma.masked_where(temp_mod.mask, temp_obs)
sal_obs = np.ma.masked_where(sal_mod.mask, sal_obs)

# Load bathy

# Extract only the first 24 time steps.
dims = {'time':[0, -1]}
#dims = {'time':range(9)}
# List of the variables to extract.
vars = ['lon', 'lat', 'nv', 'zeta', 'h', 'Times']
FVCOM = readFVCOM(datafile, vars, dims=dims)

extra = 0.5
I=np.where(FVCOM['lon'] > 180) # MICDOM: for Scottish shelf domain
FVCOM['lon'][I]=FVCOM['lon'][I]-360 # MICDOM: for Scottish shelf domain
extents = np.array((FVCOM['lon'].min() - extra,
                FVCOM['lon'].max() + extra,
                FVCOM['lat'].min() - extra,
                FVCOM['lat'].max() + extra))

triangles = FVCOM['nv'].transpose() - 1


# Fix o_prof and prof_obs

o_prof = np.zeros((len(date_obs)), dtype=int)

pind = 0
sti = 0
srf_ind = np.zeros((len(date_obs)), dtype=int)

for i in range(len(dep_obs) -1):
  if (prof_obs[i] != prof_obs[i + 1]):
    print(i / len(dep_obs) *100, '%')
    o_prof[pind] = pind
    prof_obs[sti:i + 1] = pind
    srf_ind[pind] = i
    sti = i
    pind = pind + 1

o_prof[pind] = pind
prof_obs[sti:i + 1] = pind
bot_ind = srf_ind *1
bot_ind[:-1] = srf_ind[1:]
bot_ind[-1] = len(dep_obs)

mean_t_obs = np.zeros((len(date_obs)))
mean_t_mod = np.zeros((len(date_obs)))
mean_s_obs = np.zeros((len(date_obs)))
mean_s_mod = np.zeros((len(date_obs)))
din = dep_obs > 40
print(din, len(din), len(o_prof))

for i in range(len(o_prof)):
  print(i / len(o_prof) *100, '%')
  ind = (prof_obs == o_prof[i]) & din
  mean_t_obs[i] = np.ma.mean(temp_obs[ind])
  mean_t_mod[i] = np.ma.mean(temp_mod[ind])
  mean_s_obs[i] = np.ma.mean(sal_obs[ind])
  mean_s_mod[i] = np.ma.mean(sal_mod[ind])

dtmp = np.zeros((len(prof_obs)), dtype=object)

#for i in range(0, len(o_prof)):
#  print(i / len(o_prof) *100, '%')
#  ind = prof_obs == o_prof[i]
#  dtmp[ind] = date_obs[i]

#mnt = np.array([d.month for d in dtmp])
mnt_o = np.array([d.month for d in date_obs])


#Plot

horiz = 0

ax1 = np.zeros((8), dtype=object)

if horiz:
  fig1 = plt.figure(figsize=(6, 12))
  ax1[0] = fig1.add_axes([0.05, 0.77, 0.4, 0.19])
  ax1[1] = fig1.add_axes([0.05, 0.53, 0.4, 0.19])
  ax1[2] = fig1.add_axes([0.05, 0.29, 0.4, 0.19])
  ax1[3] = fig1.add_axes([0.05, 0.05, 0.4, 0.19])
  ax1[4] = fig1.add_axes([0.5, 0.77, 0.4, 0.19])
  ax1[5] = fig1.add_axes([0.5, 0.53, 0.4, 0.19])
  ax1[6] = fig1.add_axes([0.5, 0.29, 0.4, 0.19])
  ax1[7] = fig1.add_axes([0.5, 0.05, 0.4, 0.19])
else:
  fig1 = plt.figure(figsize=(10, 6))
  ax1[0] = fig1.add_axes([0.05, 0.5, 0.19, 0.4])
  ax1[1] = fig1.add_axes([0.27, 0.5, 0.19, 0.4])
  ax1[2] = fig1.add_axes([0.49, 0.5, 0.19, 0.4])
  ax1[3] = fig1.add_axes([0.71, 0.5, 0.19, 0.4])
  ax1[4] = fig1.add_axes([0.05, 0.05, 0.19, 0.4])
  ax1[5] = fig1.add_axes([0.27, 0.05, 0.19, 0.4])
  ax1[6] = fig1.add_axes([0.49, 0.05, 0.19, 0.4])
  ax1[7] = fig1.add_axes([0.71, 0.05, 0.19, 0.4])

axc1 = fig1.add_axes([0.91, 0.1, 0.02, 0.35])
axc2 = fig1.add_axes([0.91, 0.55, 0.02, 0.35])


for i in range(len(ax1)):

  m = Basemap(llcrnrlon=extents[:2].min(),
        llcrnrlat=extents[-2:].min(),
        urcrnrlon=extents[:2].max(),
        urcrnrlat=extents[-2:].max(),
        rsphere=(6378137.00, 6356752.3142),
        resolution='h',
        projection='merc',
        lat_0=extents[-2:].mean(),
        lon_0=extents[:2].mean(),
        lat_ts=extents[-2:].mean(),
        ax=ax1[i])  
  parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), 5)  
  meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), 5) 
  #m.drawmapboundary()
  #m.drawcoastlines(zorder=100)
  m.fillcontinents(color='0.6', zorder=100)
  if (i == 0) | (i == 4):
    m.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)

  m.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)

  x1, y1 = m(FVCOM['lon'], FVCOM['lat'])
  x2, y2 = m(lon_obs, lat_obs)

  CS1 = ax1[i].tripcolor(x1, y1, triangles, FVCOM['h'], vmin=0, vmax=300)


  #if (i == 0) | (i == 4):
  #  mind = mnt_o == 4 #((mnt_o >= 3) & (mnt_o < 6))
  #elif (i == 1) | (i == 5):
  #  mind = mnt_o == 7 #((mnt_o >= 6) & (mnt_o < 9))
  #elif (i == 2) | (i == 6):
  #  mind = mnt_o == 10 #((mnt_o >= 9) & (mnt_o < 12))
  #else:
  #  mind = mnt_o == 1 #((mnt_o >= 12) | (mnt_o < 3))
  if (i == 0) | (i == 4):
    mind = ((mnt_o >= 4) & (mnt_o < 6))
  elif (i == 1) | (i == 5):
    mind = ((mnt_o >= 7) & (mnt_o < 9))
  elif (i == 2) | (i == 6):
    mind = ((mnt_o >= 10) & (mnt_o < 12))
  else:
    mind = ((mnt_o >= 1) & (mnt_o < 3))

  print(np.sum(mind))

  if i < 4:
    CS2 = ax1[i].scatter(x2[mind], y2[mind], 
        c=mean_t_obs[mind], 
        vmin=5, vmax=20, 
        cmap='jet', zorder=100)
  else:
    CS2 = ax1[i].scatter(x2[mind], y2[mind], 
        c=mean_t_mod[mind], 
        vmin=5, vmax=20, 
        cmap='jet', zorder=100)

cb1 = fig1.colorbar(CS1, axc1, extend='max')
axc1.set_ylabel('Depth (m)')
cb2 = fig1.colorbar(CS2, axc2, extend='max')
axc2.set_ylabel('Temperature ($^{\circ}$C)')


ax1[0].annotate('(a) Apr/May Obs', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[1].annotate('(b) Jul/Aug Obs', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[2].annotate('(c) Oct/Nov Obs', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[3].annotate('(d) Jan/Feb Obs', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax1[4].annotate('(e) Apr/May Mod', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[5].annotate('(f) Jul/Aug Mod', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[6].annotate('(g) Oct/Nov Mod', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[7].annotate('(h) Jan/Feb Mod', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)



ax1 = np.zeros((8), dtype=object)

if horiz:
  fig2 = plt.figure(figsize=(6, 12))
  ax1[0] = fig2.add_axes([0.05, 0.77, 0.4, 0.19])
  ax1[1] = fig2.add_axes([0.05, 0.53, 0.4, 0.19])
  ax1[2] = fig2.add_axes([0.05, 0.29, 0.4, 0.19])
  ax1[3] = fig2.add_axes([0.05, 0.05, 0.4, 0.19])
  ax1[4] = fig2.add_axes([0.5, 0.77, 0.4, 0.19])
  ax1[5] = fig2.add_axes([0.5, 0.53, 0.4, 0.19])
  ax1[6] = fig2.add_axes([0.5, 0.29, 0.4, 0.19])
  ax1[7] = fig2.add_axes([0.5, 0.05, 0.4, 0.19])
else:
  fig2 = plt.figure(figsize=(10, 6))
  ax1[0] = fig2.add_axes([0.05, 0.5, 0.19, 0.4])
  ax1[1] = fig2.add_axes([0.27, 0.5, 0.19, 0.4])
  ax1[2] = fig2.add_axes([0.49, 0.5, 0.19, 0.4])
  ax1[3] = fig2.add_axes([0.71, 0.5, 0.19, 0.4])
  ax1[4] = fig2.add_axes([0.05, 0.05, 0.19, 0.4])
  ax1[5] = fig2.add_axes([0.27, 0.05, 0.19, 0.4])
  ax1[6] = fig2.add_axes([0.49, 0.05, 0.19, 0.4])
  ax1[7] = fig2.add_axes([0.71, 0.05, 0.19, 0.4])

axc1 = fig2.add_axes([0.91, 0.1, 0.02, 0.35])
axc2 = fig2.add_axes([0.91, 0.55, 0.02, 0.35])


for i in range(len(ax1)):

  m = Basemap(llcrnrlon=extents[:2].min(),
        llcrnrlat=extents[-2:].min(),
        urcrnrlon=extents[:2].max(),
        urcrnrlat=extents[-2:].max(),
        rsphere=(6378137.00, 6356752.3142),
        resolution='h',
        projection='merc',
        lat_0=extents[-2:].mean(),
        lon_0=extents[:2].mean(),
        lat_ts=extents[-2:].mean(),
        ax=ax1[i])  
  parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), 5)  
  meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), 5) 
  #m.drawmapboundary()
  #m.drawcoastlines(zorder=100)
  m.fillcontinents(color='0.6', zorder=100)
  if (i == 0) | (i == 4):
    m.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)

  m.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)

  x1, y1 = m(FVCOM['lon'], FVCOM['lat'])
  x2, y2 = m(lon_obs, lat_obs)

  CS1 = ax1[i].tripcolor(x1, y1, triangles, FVCOM['h'], vmin=0, vmax=300)


  if (i == 0) | (i == 4):
    mind = ((mnt_o >= 4) & (mnt_o < 6))
  elif (i == 1) | (i == 5):
    mind = ((mnt_o >= 7) & (mnt_o < 9))
  elif (i == 2) | (i == 6):
    mind = ((mnt_o >= 10) & (mnt_o < 12))
  else:
    mind = ((mnt_o >= 1) & (mnt_o < 3))

  print(np.sum(mind))

  if i < 4:
    CS2 = ax1[i].scatter(x2[mind], y2[mind], 
        c=mean_s_obs[mind], 
        vmin=33, vmax=35.5, 
        cmap='jet', zorder=100)
  else:
    CS2 = ax1[i].scatter(x2[mind], y2[mind], 
        c=mean_s_mod[mind], 
        vmin=33, vmax=35.5, 
        cmap='jet', zorder=100)

cb1 = fig2.colorbar(CS1, axc1, extend='both')
axc1.set_ylabel('Depth (m)')
cb2 = fig2.colorbar(CS2, axc2, extend='both')
axc2.set_ylabel('Salinity')


ax1[0].annotate('(a) Apr/May Obs', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[1].annotate('(b) Jul/Aug Obs', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[2].annotate('(c) Oct/Nov Obs', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[3].annotate('(d) Jan/Feb Obs', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax1[4].annotate('(e) Apr/May Mod', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[5].annotate('(f) Jul/Aug Mod', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[6].annotate('(g) Oct/Nov Mod', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[7].annotate('(h) Jan/Feb Mod', (0.05, 1.08), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)


fig1.savefig('./Figures/all_profiles_tmap_bot.png')
fig2.savefig('./Figures/all_profiles_smap_bot.png')
