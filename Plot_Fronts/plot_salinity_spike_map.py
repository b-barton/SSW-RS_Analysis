#!/usr/bin/env python3

""" Plot a surface from an FVCOM model output.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime as dt
import netCDF4 as nc
import glob
import matplotlib.tri as tri
from PyFVCOM.read import ncread as readFVCOM


in_dir = '/scratch/benbar/Processed_Data_V3.02/'
#mjas = '/scratch/benbar/JASMIN/Model_Output/Daily/'
fn = sorted(glob.glob(in_dir + 'SSWRS*V3.02*dy*' + 'RE.nc'))

data = np.load(in_dir + 'front.npz', allow_pickle=True)

date_list = data['date_list']
lat = data['lat']
lon = data['lon']

st_date = dt.datetime(1993, 1, 1)
en_date = dt.datetime(2018, 12, 1)

data = np.load(in_dir + 'stack_ts_mn.npz', allow_pickle=True)
date_ts = data['date_list']
print(date_ts[0], date_ts[-1])
i_st = np.nonzero(date_ts == st_date)[0][0]
i_en = np.nonzero(date_ts == en_date)[0][0] + 1
date_ts = date_ts[i_st:i_en]

sal = data['sal'][i_st:i_en, :] # time, space

data.close()

# get dates

st_date = dt.datetime(1993, 7, 1)
i_st = np.nonzero(date_ts == st_date)[0][0]
sal1 = sal[i_st, :]
st_date = dt.datetime(1998, 7, 1)
i_st = np.nonzero(date_ts == st_date)[0][0]
sal2 = sal[i_st, :]
st_date = dt.datetime(2007, 7, 1)
i_st = np.nonzero(date_ts == st_date)[0][0]
sal3 = sal[i_st, :]
st_date = dt.datetime(2008, 7, 1)
i_st = np.nonzero(date_ts == st_date)[0][0]
sal4 = sal[i_st, :]
st_date = dt.datetime(2011, 7, 1)
i_st = np.nonzero(date_ts == st_date)[0][0]
sal5 = sal[i_st, :]
st_date = dt.datetime(2012, 7, 1)
i_st = np.nonzero(date_ts == st_date)[0][0]
sal6 = sal[i_st, :]


# Distance

data = np.load(in_dir + 'coast_distance.npz')
dist_c = data['dist_c']
data.close()



dims = {'time':':10'}
# List of the variables to extract.
vars = ('nv', 'h')
FVCOM = readFVCOM(fn[-1], vars, dims=dims)
nv = FVCOM['nv']
h = FVCOM['h']

scot_x = [-7, -7, -2, -2, -7]
scot_y = [58, 59.6, 59.6, 58, 58]



fig1 = plt.figure(figsize=(10, 14))  # size in inches
ax1 = fig1.add_axes([0.05, 0.69, 0.35, 0.27])
ax2 = fig1.add_axes([0.515, 0.69, 0.35, 0.27])
ax3 = fig1.add_axes([0.05, 0.37, 0.35, 0.27])
ax4 = fig1.add_axes([0.515, 0.37, 0.35, 0.27])
ax5 = fig1.add_axes([0.05, 0.05, 0.35, 0.27])
ax6 = fig1.add_axes([0.515, 0.05, 0.35, 0.27])
cax1 = fig1.add_axes([0.405, 0.69, 0.01, 0.27])
cax2 = fig1.add_axes([0.92, 0.69, 0.01, 0.27])
cax3 = fig1.add_axes([0.405, 0.19, 0.01, 0.32])
cax4 = fig1.add_axes([0.87, 0.19, 0.01, 0.32])

#extra = 1
#extents = np.array((lon.min() -  extra,
#               lon.max() + extra,
#               lat.min() - extra,
#               lat.max() + extra))
extents = np.array((-9, 5, 54, 61))

def do_basemap(ax, extents):
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
          ax=ax)  

  parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), 5)  
  meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), 5) 
  m1.drawmapboundary()
  #m1.drawcoastlines(zorder=100)
  #m.fillcontinents(color='0.6', zorder=100)
  m1.drawparallels(parallels, labels=[1, 0, 0, 0],
                  fontsize=10, linewidth=0)
  m1.drawmeridians(meridians, labels=[0, 0, 0, 1],
                  fontsize=10, linewidth=0)

  return m1

m1 = do_basemap(ax1, extents)
m2 = do_basemap(ax2, extents)
m3 = do_basemap(ax3, extents)
m4 = do_basemap(ax4, extents)
m5 = do_basemap(ax5, extents)
m6 = do_basemap(ax6, extents)
#m7 = do_basemap(ax7, extents)
#m8 = do_basemap(ax8, extents)


mx, my = m1(lon, lat)
triangles = nv.transpose() -1

# axes1

#ax1.tricontour(mx, my, triangles, dist_c, levels=[25, 50, 75], zorder=100)
cs1 = ax1.tripcolor(mx, my, triangles, sal1, vmin=32, vmax=35.5, zorder=99)

fig1.colorbar(cs1, cax=cax1, extend='both')
#cax1.set_ylabel('Temperature ($^{\circ}$C)')


# axes2

cs2 = ax2.tripcolor(mx, my, triangles, sal2, vmin=32, vmax=35.5, zorder=99)
#cs2 = ax2.tripcolor(mx, my, triangles, mean_fc, zorder=99)#, vmin=32, vmax=35.5, zorder=99)

cbar = fig1.colorbar(cs2, cax=cax2)#, extend='min')
cax2.set_ylabel('Salinity')



# axes3

cs3 = ax3.tripcolor(mx, my, triangles, sal3, vmin=32, vmax=35.5, zorder=99)

#scx, scy = m1(scot_x, scot_y)
#ax1.plot(scx, scy, '-', zorder=101)

fig1.colorbar(cs3, cax=cax3, extend='max')
#cax3.set_ylabel('Temperature Gradient ($^{\circ}$C/km)')


# axes4

cs4 = ax4.tripcolor(mx, my, triangles, sal4, vmin=32, vmax=35.5, zorder=99)

fig1.colorbar(cs4, cax=cax4, extend='max')
#cax4.set_ylabel('Salinity Gradient (g/kg km)')


# axes5

cs5 = ax5.tripcolor(mx, my, triangles, sal5, vmin=32, vmax=35.5, zorder=99)

#fig1.colorbar(cs5, cax=cax5, extend='max')
#cax5.set_ylabel('Temperature Gradient ($^{\circ}$C km$^{-1}$)')


# axes6

cs6 = ax6.tripcolor(mx, my, triangles, sal6, vmin=32, vmax=35.5, zorder=99)

#fig1.colorbar(cs6, cax=cax6, extend='max')
#cax6.set_ylabel('Salinity Gradient (PSU km$^{-1}$)')


# axes7
#mx, my = m1(lon_15, lat_15)

#cs7 = ax7.pcolormesh(mx, my, mean_t15 * 1000, vmin=0, vmax=4e-2, zorder=99)

#fig1.colorbar(cs5, cax=cax5, extend='max')
#cax5.set_ylabel('Temperature Gradient ($^{\circ}$C km$^{-1}$)')


# axes8

#cs8 = ax8.pcolormesh(mx, my, mean_s15 * 1000, vmin=0, vmax=4e-2, zorder=99)

#fig1.colorbar(cs6, cax=cax6, extend='max')
#cax6.set_ylabel('Salinity Gradient (PSU km$^{-1}$)')



ax1.annotate('(a)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(d)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax5.annotate('(e)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax6.annotate('(f)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
#ax7.annotate('(g)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
#ax8.annotate('(h)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig1.savefig('./Figures/sal_snapshot.png')



