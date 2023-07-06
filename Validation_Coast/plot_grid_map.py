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
import gsw as sw


in_dir = '/scratch/benbar/Processed_Data_V3.02'

grid_obs_file = in_dir + '/Indices/grid_match_full.npz'
mod_file = in_dir + '/Indices/grid_model_full.npz'
amm7_file = in_dir + '/grid_amm7_all_full.npz'


datafile = (in_dir 
    + '/SSWRS_V3.02_NOC_FVCOM_NWEuropeanShelf_01dy_19930101-1200_RE.nc')


data = np.load(grid_obs_file, allow_pickle=True)
o_prof = data['o_prof'] 
lat_obs = data['lat_obs']
lon_obs = data['lon_obs']
dep_grid = data['dep_grid']
temp_grid = data['temp_grid']
sal_grid = data['sal_grid']
o_date = data['o_date']
mask_t_grid = data['mask_t_grid']
mask_s_grid = data['mask_s_grid']
data.close()

print(np.min(sal_grid), np.max(sal_grid))
mask_s_grid = (mask_s_grid == 1) | (sal_grid < 1) | (mask_t_grid == 1)
temp_grid = np.ma.masked_where(mask_t_grid == 1, temp_grid)
sal_grid = np.ma.masked_where(mask_s_grid == 1, sal_grid)
dep_grid = np.ma.masked_where(mask_t_grid == 1, dep_grid)

def sa_from_sp(sal, dep, lat, lon):
  p = sw.p_from_z(dep, lat)
  return sw.SA_from_SP(sal ,p, lon, lat)

sal_grid1 = sa_from_sp(sal_grid, dep_grid, 56, 0)
sal_mask = np.isnan(sal_grid1) | np.isinf(sal_grid1) | (sal_grid1 > 40)
sal_grid1[sal_mask] = 0
sal_grid = np.ma.masked_where(sal_grid.mask | sal_mask, sal_grid1)


data = np.load(mod_file)
temp_mgrid = data['temp_mgrid']
sal_mgrid = data['sal_mgrid']
h_mgrid = data['h_mgrid']
data.close()

h_mgrid = np.ma.masked_where(h_mgrid == -9999, h_mgrid)
temp_mgrid = np.ma.masked_where((temp_mgrid == -9999) | (mask_t_grid == 1), temp_mgrid)
sal_mgrid = np.ma.masked_where((sal_mgrid == -9999) | (mask_s_grid == 1), sal_mgrid)

if 0:
  for i in range(temp_mgrid.shape[0]):
    temp_mgrid[i, :] = np.ma.masked_where(h_mgrid > 200, temp_mgrid[i, :])
    sal_mgrid[i, :] = np.ma.masked_where(h_mgrid > 200, sal_mgrid[i, :])
    temp_grid[i, :] = np.ma.masked_where(h_mgrid > 200, temp_grid[i, :])
    sal_grid[i, :] = np.ma.masked_where(h_mgrid > 200, sal_grid[i, :])


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
#extents = np.array((FVCOM['lon'].min() - extra,
#                FVCOM['lon'].max() + extra,
#                FVCOM['lat'].min() - extra,
#                FVCOM['lat'].max() + extra))
extents = np.array((-9, 2, 54, 61))


triangles = FVCOM['nv'].transpose() - 1


din = 20
temp_grid = np.ma.masked_where(dep_grid > din, temp_grid)
sal_grid = np.ma.masked_where(dep_grid > din, sal_grid)
temp_mgrid = np.ma.masked_where(dep_grid > din, temp_mgrid)
sal_mgrid = np.ma.masked_where(dep_grid > din, sal_mgrid)

mean_t_obs = np.ma.mean(temp_grid, axis=0)
mean_t_mod = np.ma.mean(temp_mgrid, axis=0)
mean_s_obs = np.ma.mean(sal_grid, axis=0)
mean_s_mod = np.ma.mean(sal_mgrid, axis=0)


mnt_o = np.array([d.month for d in o_date])


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

axc1 = fig1.add_axes([0.91, 0.55, 0.02, 0.35])
axc2 = fig1.add_axes([0.91, 0.1, 0.02, 0.35])


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

#  CS1 = ax1[i].tripcolor(x1, y1, triangles, FVCOM['h'], vmin=0, vmax=300)


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
    CS1 = ax1[i].scatter(x2[mind], y2[mind], 
        c=mean_t_obs[mind], 
        vmin=5, vmax=20, 
        cmap='jet', zorder=100)
  else:
    CS2 = ax1[i].scatter(x2[mind], y2[mind], 
        c=mean_t_mod[mind] - mean_t_obs[mind], 
        vmin=-2, vmax=2, 
        cmap='bwr', zorder=100)
    #CS2 = ax1[i].scatter(x2[mind], y2[mind], 
    #    c=mean_t_mod[mind] - mean_t_obs[mind], 
    #    vmin=-1, vmax=1, 
    #    cmap='jet', zorder=100)

cb1 = fig1.colorbar(CS1, axc1, extend='both')
axc1.set_ylabel('Obs. Temperature ($^{\circ}$C)')
cb2 = fig1.colorbar(CS2, axc2, extend='both')
axc2.set_ylabel('Temperature Difference ($^{\circ}$C)')


ax1[0].annotate('(a) Apr/May Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[1].annotate('(b) Jul/Aug Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[2].annotate('(c) Oct/Nov Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[3].annotate('(d) Jan/Feb Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax1[4].annotate('(e) Apr/May Mod - Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[5].annotate('(f) Jul/Aug Mod - Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[6].annotate('(g) Oct/Nov Mod - Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[7].annotate('(h) Jan/Feb Mod - Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)



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

axc1 = fig2.add_axes([0.91, 0.55, 0.02, 0.35])
axc2 = fig2.add_axes([0.91, 0.1, 0.02, 0.35])


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

  #CS1 = ax1[i].tripcolor(x1, y1, triangles, FVCOM['h'], vmin=0, vmax=300)


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
    CS1 = ax1[i].scatter(x2[mind], y2[mind], 
        c=mean_s_obs[mind], 
        vmin=33, vmax=35.5, 
        cmap='jet', zorder=100)
  else:
    CS2 = ax1[i].scatter(x2[mind], y2[mind], 
        c=mean_s_mod[mind] - mean_s_obs[mind], 
        vmin=-0.5, vmax=0.5, 
        cmap='bwr', zorder=100)
    #CS2 = ax1[i].scatter(x2[mind], y2[mind], 
    #    c=mean_s_mod[mind] - mean_s_obs[mind], 
    #    vmin=-2, vmax=2, 
    #    cmap='jet', zorder=100)

cb1 = fig2.colorbar(CS1, axc1, extend='both')
axc1.set_ylabel('Obs. Salinity (g/kg)')
cb2 = fig2.colorbar(CS2, axc2, extend='both')
axc2.set_ylabel('Salinity Difference (g/kg)')


ax1[0].annotate('(a) Apr/May Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[1].annotate('(b) Jul/Aug Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[2].annotate('(c) Oct/Nov Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[3].annotate('(d) Jan/Feb Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax1[4].annotate('(e) Apr/May Mod - Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[5].annotate('(f) Jul/Aug Mod - Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[6].annotate('(g) Oct/Nov Mod - Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax1[7].annotate('(h) Jan/Feb Mod - Obs', (0.05, 1.02), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)


fig1.savefig('./Figures/all_grid_tmap.png')
fig2.savefig('./Figures/all_grid_smap.png')
