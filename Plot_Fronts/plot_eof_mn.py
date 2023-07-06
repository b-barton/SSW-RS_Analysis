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
import scipy.stats as stats

in_dir = '/scratch/benbar/Processed_Data_V3.02/'
#mjas = '/scratch/benbar/JASMIN/Model_Output/Daily/'
fn = sorted(glob.glob(in_dir + 'SSWRS*V3.02*dy*' + 'RE.nc'))

season = 0
if season:
  sstr = '_season'
else:
  sstr = ''

data = np.load(in_dir + 'eof_data' + sstr + '_mn_temp.npz', allow_pickle=True)

t_var = data['variability']
t_eof = data['space_eof']
t_amp = data['amp_pc']
date_list = data['date']
lat = data['lat']
lon = data['lon']
triangles = data['tri']
data.close()

mask = t_eof == -1e20
t_eof[mask] = 0
t_eof = np.ma.masked_where(mask, t_eof)


data = np.load(in_dir + 'eof_data' + sstr + '_mn_sal.npz', allow_pickle=True)

s_var = data['variability']
s_eof = data['space_eof']
s_amp = data['amp_pc']
data.close()

mask = s_eof == -1e20
s_eof[mask] = 0
s_eof = np.ma.masked_where(mask, s_eof)

data = np.load(in_dir + 'mag_mean_mn.npz', allow_pickle=True)
mag = data['mag']
mag_date = data['date']
data.close()

print(s_var[:2], t_var[:2])

dims = {'time':':10'}
# List of the variables to extract.
vars = ('nv', 'h')
FVCOM = readFVCOM(fn[-1], vars, dims=dims)
nv = FVCOM['nv']
h = FVCOM['h']


data = np.load(in_dir + 'riv_mean_mn.npz', allow_pickle=True)
riv_run = data['riv_max']
riv_date = data['riv_date']
data.close()
print(riv_run.shape, np.ma.max(riv_run))

#st_d = np.nonzero(riv_date >= date_list[0])[0][0]
#en_d = np.nonzero(riv_date >= date_list[-1])[0][0] + 1
#riv_run = riv_run[st_d:en_d, :]
#riv_date = riv_date[st_d:en_d]
print(riv_run.shape, s_amp[:, 1].shape)


# this is bad, calculate eof on monthly mean data
#s_amp_mn = s_amp_mn[6:-6]

#slope1, intercept1, r1, p1, se1 = stats.linregress(mag[:, 0], s_amp_mn)
#print(r1, p1)
#slope2, intercept2, r2, p2, se2 = stats.linregress(riv_run[:, 0], s_amp[:, 1])
#print(r2, p2)

if 0:
  run = 12
  s_amp_s = np.ma.zeros((len(date_list) - run, s_amp.shape[1]))
  date_s = np.empty(len(date_list) - run, dtype=object)

  for i in range(len(date_s)):
    s_amp_s[i, :] = np.ma.mean(s_amp[i:i+run, :], axis=0)
    date_s[i] = date_list[i + int(run//2)]



fig1 = plt.figure(figsize=(12, 6))  # size in inches
ax1 = fig1.add_axes([0.05, 0.3, 0.35, 0.68])
ax2 = fig1.add_axes([0.525, 0.3, 0.35, 0.68])
ax3 = fig1.add_axes([0.05, 0.05, 0.35, 0.18])
ax4 = fig1.add_axes([0.525, 0.05, 0.35, 0.18])
cax1 = fig1.add_axes([0.4, 0.4, 0.01, 0.55])
cax2 = fig1.add_axes([0.88, 0.4, 0.01, 0.55])

#extra = 1
#extents = np.array((lon.min() -  extra,
#               lon.max() + extra,
#               lat.min() - extra,
#               lat.max() + extra))
extents = np.array((-10, 2, 54, 61))

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
#m.fillcontinents(color='0.6', zorder=100)
m1.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)
m1.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)

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
#m.fillcontinents(color='0.6', zorder=100)
m2.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)
m2.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)


mx, my = m1(lon, lat)
triangles = nv.transpose() -1

cs1 = ax1.tripcolor(mx, my, triangles, t_eof[:, 0], cmap=plt.get_cmap('bwr'), zorder=99)

fig1.colorbar(cs1, cax=cax1)
#cax1.set_ylabel('Temperature ($^{\circ}$C m$^{-1}$)')

cs2 = ax2.tripcolor(mx, my, triangles, t_eof[:, 1], cmap=plt.get_cmap('bwr'), zorder=99)

fig1.colorbar(cs2, cax=cax2)
#cax2.set_ylabel('Salinity Gradient (PSU m$^{-1}$)')

ax3.plot(date_list, t_amp[:, 0])
ax4.plot(date_list, t_amp[:, 1])

ax3.annotate('(c) {:.2f} %'.format(t_var[0]), (0.1, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(d) {:.2f} %'.format(t_var[1]), (0.1, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax1.annotate('(a)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)



fig2 = plt.figure(figsize=(12, 6))  # size in inches
ax1 = fig2.add_axes([0.05, 0.3, 0.35, 0.68])
ax2 = fig2.add_axes([0.525, 0.3, 0.35, 0.68])
ax3 = fig2.add_axes([0.05, 0.05, 0.35, 0.18])
ax4 = fig2.add_axes([0.525, 0.05, 0.35, 0.18])
#ax3b = ax3.twinx()
#ax4b = ax4.twinx()
cax1 = fig2.add_axes([0.4, 0.4, 0.01, 0.55])
cax2 = fig2.add_axes([0.88, 0.4, 0.01, 0.55])


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
#m.fillcontinents(color='0.6', zorder=100)
m1.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)
m1.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)

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
#m.fillcontinents(color='0.6', zorder=100)
m2.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)
m2.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)


mx, my = m1(lon, lat)
triangles = nv.transpose() -1

cs1 = ax1.tripcolor(mx, my, triangles, s_eof[:, 0], cmap=plt.get_cmap('bwr'), vmin=-1, vmax=1, zorder=99)

fig2.colorbar(cs1, cax=cax1)
#cax1.set_ylabel('Temperature ($^{\circ}$C m$^{-1}$)')

cs2 = ax2.tripcolor(mx, my, triangles, s_eof[:, 1], cmap=plt.get_cmap('bwr'), vmin=-1, vmax=1, zorder=99)

fig2.colorbar(cs2, cax=cax2)
#cax2.set_ylabel('Salinity Gradient (PSU m$^{-1}$)')

ax3.plot(date_list, s_amp[:, 0], zorder=104)
#ax3b.plot(mag_date, np.ma.mean(mag, axis=1), color='tab:orange', zorder=104)
ax4.plot(date_list, s_amp[:, 1], zorder=104)
#ax4b.plot(riv_date, riv_run[:, 0], '--', color='tab:orange', zorder=104)

#ax3.plot(date_s, s_amp_s[:, 0])
#ax4.plot(date_s, s_amp_s[:, 1])


#ax4b.set_xlim([dt.datetime(1995, 1, 1), dt.datetime(2006, 1, 1)])
#ax4b.set_xticklabels([])
#ax3.tick_params(axis='y', colors='tab:blue')
#ax3b.tick_params(axis='y', colors='tab:orange')
ax3.set_ylabel('PC 1')
#ax3b.set_ylabel('Speed (m/s)', color='tab:orange')

#ax4.tick_params(axis='y', colors='tab:blue')
#ax4b.tick_params(axis='y', colors='tab:orange')
ax4.set_ylabel('PC 2')
#ax4b.set_ylabel('River Flux\n(m$^{3}$/s)', color='tab:orange')

ax3.annotate('(c) {:.2f} %'.format(s_var[0]), (0.1, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(d) {:.2f} %'.format(s_var[1]), (0.1, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

#ax3b.annotate('(c) {:.2f} %'.format(s_var[0]) + ' r = {:.2f}'.format(r1), (0.1, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
#ax4b.annotate('(d) {:.2f} %'.format(s_var[1]) + ' r = {:.2f}'.format(r2), (0.1, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)

ax1.annotate('(a)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)


print('S {:.2f} % {:.2f} % {:.2f} %'.format(s_var[0], s_var[1], s_var[2]))
print('T {:.2f} % {:.2f} % {:.2f} %'.format(t_var[0], t_var[1], t_var[2]))


fig1.savefig('./Figures/eof_temp' + sstr + '_mn.png')
fig2.savefig('./Figures/eof_sal' + sstr + '_mn.png')


