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

out_dir = '/scratch/benbar/Processed_Data_V3.02/'
fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'


st_date = dt.datetime(1993, 1, 1)
en_date = dt.datetime(2018, 12, 1)

data = np.load(out_dir + 'stack_bot_uv_mn1.npz', allow_pickle=True)
date = data['date_list'][:-12]
print(data['date_list'][0], data['date_list'][-1])
ua = data['ub'][:-12, :]
va = data['vb'][:-12, :] # time, space
lonc = data['lonc']
latc = data['latc']
triangles = data['tri']

data.close()

data = np.load(out_dir + 'stack_bot_uv_mn2.npz', allow_pickle=True)
date2 = data['date_list']
ua2 = data['ub']
va2 = data['vb'] # time, space

data.close()

print(date2[0], date2[-1])
date = np.append(date, date2)
ua_bot = np.append(ua, ua2, axis=0)
va_bot = np.append(va, va2, axis=0) # time, space
ua2 = None
va2 = None

ua = np.ma.mean(ua_bot, axis=0)
va = np.ma.mean(va_bot, axis=0)

data = np.load(out_dir + 'front.npz', allow_pickle=True)
tb_rho = data['tb_rho']
date_list = data['date_list']
data.close()

tb_freq = (np.sum((tb_rho > -0.01), axis=0) / len(date_list)) * 100
tb_rho = np.ma.mean(tb_rho, axis=0)

data = np.load(out_dir + 'stack_ts_mn.npz', allow_pickle=True)
sal = data['sal'] # time, space
data.close()
mean_sal = np.ma.mean(sal, axis=0)
sal = None

data = np.load(out_dir + 'stack_ts_bot_mn.npz', allow_pickle=True)
sal = data['sal'] # time, space
data.close()
mean_sal_bot = np.ma.mean(sal, axis=0)
sal = None


data = np.load(out_dir + 'sswrs_xy.npz')
x = data['x']
y = data['y']
xc = data['xc']
yc = data['yc']
data.close()


data = np.load(out_dir + 'coast_distance.npz', allow_pickle=True)
dist = data['dist_c']
lon = data['lon']
lat = data['lat']
data.close()

dist_e = fvcom.grid.nodes2elems(dist, triangles)

trio = tri.Triangulation(x, y, triangles=np.asarray(triangles))

a = tri.LinearTriInterpolator(trio, dist)
dd_dx, dd_dy = a.gradient(x, y)
dd_dx = fvcom.grid.nodes2elems(dd_dx, triangles)
dd_dy = fvcom.grid.nodes2elems(dd_dy, triangles)

angle = np.arctan2(dd_dx, dd_dy) # from North


def align(x_var, y_var, angle):
  # Align components to be along and across the transect

  across_var = np.zeros_like(x_var)
  along_var = np.zeros_like(x_var)

  #for i in range(x_var.shape[0]): 
  var_mag = ((x_var[:] ** 2) + (y_var[:] ** 2)) ** 0.5
  var_dir = np.arctan2(x_var[:], y_var[:])
  new_dir = var_dir - angle # subtract so North lies along transect
  across_var[:] = var_mag * np.sin(new_dir) # x direction
  along_var[:] = var_mag * np.cos(new_dir) # y direction
  return across_var, along_var

# relative to increasing distance not across/along coast
uv_across, uv_along = align(ua, va, angle) 
uvb_across, uvb_along = align(ua_bot, va_bot, angle) 

mag_uvb = np.abs(uvb_along)
print(mag_uvb.shape)
print(np.ma.mean(mag_uvb))
freq_bot = (mag_uvb < 0.01).astype(np.uint8)
bot0_freq = (np.ma.sum(freq_bot, axis=0) / len(date_list)) * 100
bot0_freq = fvcom.grid.elems2nodes(bot0_freq, triangles)

veln_across = fvcom.grid.elems2nodes(uv_across, triangles)
veln_along = fvcom.grid.elems2nodes(uv_along, triangles)

# relative to coast
vel_parallel = veln_across
vel_perpendicular = veln_along
vel_perpendicular = np.abs(vel_perpendicular)

# Plot

fig1 = plt.figure(figsize=(8, 5))  # size in inches
if 0:
  ax1 = fig1.add_axes([0.05, 0.1, 0.27, 0.78])
  ax2 = fig1.add_axes([0.37, 0.1, 0.27, 0.78])
  ax3 = fig1.add_axes([0.69, 0.1, 0.27, 0.78])
  cax1 = fig1.add_axes([0.1, 0.97, 0.49, 0.01])
  cax2 = fig1.add_axes([0.69, 0.97, 0.27, 0.01])
else:
  ax2 = fig1.add_axes([0.05, 0.1, 0.425, 0.77])
  ax3 = fig1.add_axes([0.525, 0.1, 0.425, 0.77])
  cax1 = fig1.add_axes([0.1, 0.97, 0.325, 0.01])
  cax2 = fig1.add_axes([0.575, 0.97, 0.325, 0.01])


if 0:
  #extra = 1
  #extents = np.array((lon.min() -  extra,
  #               lon.max() + extra,
  #               lat.min() - extra,
  #               lat.max() + extra))
  extents = np.array((-9, 2, 54, 61))

  m1 = Basemap(llcrnrlon=extents[:2].min(),
          llcrnrlat=extents[-2:].min(),
          urcrnrlon=extents[:2].max(),
          urcrnrlat=extents[-2:].max(),
          rsphere=(6378137.00, 6356752.3142),
          resolution='h',
          projection='merc',
          lat_0=extents[-2:].mean(),
          lon_0=extents[:2].mean(),
          lat_ts=extents[-2:].mean(),
          ax=ax1)  

  parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), 5)  
  meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), 5) 
  m1.drawmapboundary()
  #m1.drawcoastlines(zorder=100)
  m1.fillcontinents(color='0.6', zorder=100)
  m1.drawparallels(parallels, labels=[1, 0, 0, 0],
                  fontsize=10, linewidth=0)
  m1.drawmeridians(meridians, labels=[0, 0, 0, 1],
                  fontsize=10, linewidth=0)


  mx, my = m1(lon, lat)
  #triangles = nv.transpose() -1

  # axes1

  #ax1.tricontour(mx, my, triangles, dist_c, levels=[25, 50, 75], zorder=100)
  cs1 = ax1.tripcolor(mx, my, triangles, vel_parallel, vmin=-0.1, vmax=0.1, cmap=plt.get_cmap('bwr'), zorder=99)



extents = np.array((-9, 2, 54, 61))

m2 = Basemap(llcrnrlon=extents[:2].min(),
        llcrnrlat=extents[-2:].min(),
        urcrnrlon=extents[:2].max(),
        urcrnrlat=extents[-2:].max(),
        rsphere=(6378137.00, 6356752.3142),
        resolution='h',
        projection='merc',
        lat_0=extents[-2:].mean(),
        lon_0=extents[:2].mean(),
        lat_ts=extents[-2:].mean(),
        ax=ax2)  

parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), 5)  
meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), 5) 
m2.drawmapboundary()
#m1.drawcoastlines(zorder=100)
m2.fillcontinents(color='0.6', zorder=100)
m2.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)
m2.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)


mx, my = m2(lon, lat)
#triangles = nv.transpose() -1

# axes1

#cs2 = ax2.tripcolor(mx, my, triangles, vel_perpendicular, vmin=-0.05, vmax=0.05, cmap=plt.get_cmap('bwr'), zorder=99)
cs2 = ax2.tripcolor(mx, my, triangles, vel_perpendicular, vmin=0, vmax=0.05, zorder=99)
#cs2 = ax2.tripcolor(mx, my, triangles, bot0_freq, vmin=0, vmax=4, zorder=99)
ax2.tricontour(mx, my, triangles, mean_sal, [35], colors='red', zorder=100)
#ax2.tricontour(mx, my, triangles, mean_sal_bot, [35], colors='red', zorder=100)
#ax2.tricontour(mx, my, triangles, tb_rho, [-0.04, 0], zorder=100)
#ax2.tricontour(mx, my, triangles, tb_freq, [50], zorder=99)

fig1.colorbar(cs2, cax=cax1, orientation='horizontal')
cax1.set_xlabel('Bottom Velocity (m/s)')


extents = np.array((-9, 2, 54, 61))

m3 = Basemap(llcrnrlon=extents[:2].min(),
        llcrnrlat=extents[-2:].min(),
        urcrnrlon=extents[:2].max(),
        urcrnrlat=extents[-2:].max(),
        rsphere=(6378137.00, 6356752.3142),
        resolution='h',
        projection='merc',
        lat_0=extents[-2:].mean(),
        lon_0=extents[:2].mean(),
        lat_ts=extents[-2:].mean(),
        ax=ax3)  

parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), 5)  
meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), 5) 
m3.drawmapboundary()
#m1.drawcoastlines(zorder=100)
m3.fillcontinents(color='0.6', zorder=100)
m3.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)
m3.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)


mx, my = m3(lon, lat)
#triangles = nv.transpose() -1

# axes1

cs3 = ax3.tripcolor(mx, my, triangles, tb_rho, vmin=-1, vmax=0, zorder=99)
ax3.tricontour(mx, my, triangles, mean_sal, [35], colors='red', zorder=100)
#ax3.tricontour(mx, my, triangles, tb_rho, [-0.05, 0], zorder=100)
ax3.tricontour(mx, my, triangles, tb_freq, [50], zorder=99)

fig1.colorbar(cs3, cax=cax2, orientation='horizontal')
cax2.set_xlabel('Density Difference (kg/m$^{3}$)')


if 0:
  ax1.annotate('(a) Parallel', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(a)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(b)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig1.savefig('./Figures/vel_map_coast.png')

