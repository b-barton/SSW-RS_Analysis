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


st_date = dt.datetime(1993, 1, 1)
en_date = dt.datetime(2018, 12, 1)

data = np.load(out_dir + 'stack_uv_mn1.npz', allow_pickle=True)
date = data['date_list'][:-12]
print(data['date_list'][0], data['date_list'][-1])
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

print(date2[0], date2[-1])
date = np.append(date, date2)
ua = np.append(ua, ua2, axis=0)
va = np.append(va, va2, axis=0) # time, space


data = np.load(out_dir + 'coast_distance.npz', allow_pickle=True)
dist = data['dist_c']
lon = data['lon']
lat = data['lat']
data.close()

dist_e = fvcom.grid.nodes2elems(dist, triangles)


season = 0
if season:
  sstr = '_season'
else:
  sstr = ''

data = np.load(out_dir + 'eof_data_mn' + sstr + '_sal.npz', allow_pickle=True)

s_var = data['variability']
s_eof = data['space_eof']
s_amp = data['amp_pc']
date_mn = data['date']
lon = data['lon']
lat = data['lat']

data.close()

mask = s_eof == -1e20
s_eof[mask] = 0
s_eof = np.ma.masked_where(mask, s_eof)


# Mask area

x_min = -9
#x_max = 2
x_max = 2

#y_min = 54
y_max = 61
y_min = 54

#x_min = -10
#x_max = -3
#y_min = 54
#y_max = 61

sel_ind = np.invert((lonc >= x_min) & (lonc <= x_max) 
                  & (latc >= y_min) & (latc <= y_max))

#sel_dist = dist_e < 50
#sel_dist = dist_e < 50

ua = np.ma.array(ua)
va = np.ma.array(va)

for i in range(ua.shape[0]):
  ua[i, :] = np.ma.masked_where(sel_ind == True, ua[i, :])
  va[i, :] = np.ma.masked_where(sel_ind == True, va[i, :])

ua_mn = ua * 1
va_mn = va * 1


# running mean

run = 12
ua_s = np.ma.zeros((len(date) - run, ua_mn.shape[1]))
va_s = np.ma.zeros((len(date) - run, va_mn.shape[1]))
date_s = np.empty(len(date) - run, dtype=object)

for i in range(len(date_s)):
  ua_s[i, :] = np.ma.mean(ua_mn[i:i+run, :], axis=0)
  va_s[i, :] = np.ma.mean(va_mn[i:i+run, :], axis=0)
  date_s[i] = date[i + int(run//2)]

date = date_s

mag = (ua_s ** 2 + va_s ** 2) ** 0.5
#mag = sig.detrend(mag, axis=0, type='linear')


print(date_mn[-1], date[-1])
i_st = np.nonzero(date_mn[0] == date)[0][0]
i_en = np.nonzero(date_mn[-1] == date)[0][0] + 1
#date_mn = date_mn[i_st:]
#s_amp = s_amp[i_st:, :]
date = date[i_st:i_en]
mag = mag[i_st:i_en, :]

print(np.shape(mag), np.shape(s_amp))
print(date[0], date[-1], date_mn[0], date_mn[-1])

r_value1 = np.zeros((ua_mn.shape[1]))
r_value2 = np.zeros((ua_mn.shape[1]))
p_value1 = np.zeros((ua_mn.shape[1]))
p_value2 = np.zeros((ua_mn.shape[1]))

edf_in1, td = df_regress.edf(s_amp[:, 0], efold=1)
edf_in2, td = df_regress.edf(s_amp[:, 1], efold=1)

for i in range(ua_mn.shape[1]):
#  slope1, intercept1, r_value1[i], p_value1[i], se1 = df_regress.run_sig(mag[:, i], s_amp[:, 0], run=12)
#  slope1, intercept1, r_value2[i], p_value2[i], se1 = df_regress.run_sig(mag[:, i], s_amp[:, 1], run=12)
  slope1, intercept1, r_value1[i], p_value1[i], se1 = df_regress.edf_sig(mag[:, i], s_amp[:, 0], edf_in=edf_in1)
  slope1, intercept1, r_value2[i], p_value2[i], se1 = df_regress.edf_sig(mag[:, i], s_amp[:, 1], edf_in=edf_in2)
#  slope1, intercept1, r_value1[i], p_value1[i], se1 = stats.linregress(mag[:, i], s_amp[:, 0])
#  slope1, intercept1, r_value2[i], p_value2[i], se1 = stats.linregress(mag[:, i], s_amp[:, 1])
r_value1[np.isnan(r_value1)] = 0
r_value2[np.isnan(r_value2)] = 0
print(np.max(r_value1))
print(np.max(r_value2))


if ((np.max(r_value1) ** 2) ** 0.5) > ((np.min(r_value1) ** 2) ** 0.5):
  max_ind = r_value1 == np.max(r_value1)
else:
  max_ind = r_value1 == np.min(r_value1)


mag_max = mag[:, max_ind]


np.savez(out_dir + 'mag_mean_mn', mag=mag_max, date=date)



rn1 = fvcom.grid.elems2nodes(r_value1, triangles)
rn2 = fvcom.grid.elems2nodes(r_value2, triangles)
pn1 = fvcom.grid.elems2nodes(p_value1, triangles)
pn2 = fvcom.grid.elems2nodes(p_value2, triangles)

if ((np.max(rn1) ** 2) ** 0.5) > ((np.min(rn1) ** 2) ** 0.5):
  max_indn = rn1 == np.max(rn1)
else:
  max_indn = rn1 == np.min(rn1)

print(lon[max_indn], lat[max_indn])


fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])

ax1.plot(date, np.ma.mean(mag, axis=1))

#plt.show()



fig2 = plt.figure(figsize=(10, 6))  # size in inches
ax1 = fig2.add_axes([0.05, 0.1, 0.40, 0.8])
ax2 = fig2.add_axes([0.5, 0.1, 0.40, 0.8])
cax1 = fig2.add_axes([0.91, 0.3, 0.01, 0.4])

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
#m.fillcontinents(color='0.6', zorder=100)
m1.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)
m1.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)


mx, my = m1(lon, lat)
#triangles = nv.transpose() -1

# axes1

#ax1.tricontour(mx, my, triangles, dist_c, levels=[25, 50, 75], zorder=100)
cs1 = ax1.tripcolor(mx, my, triangles, rn1, vmin=-0.85, vmax=0.85, cmap=plt.get_cmap('bwr'), zorder=99)

rn1a = np.ma.ones(rn1.shape) - 0.5
rn1a[pn1 < 0.05] = 2
ax1.tricontourf(mx, my, triangles, rn1a, levels=[0, 1], vmin=0, vmax=2, cmap=plt.get_cmap('gray'), zorder=100, alpha=0.6)
ax1.tricontour(mx, my, triangles, rn1a, levels=[1], colors='k', linewidths=0.5, zorder=100)

#ax1.scatter(mx[pn1 > 0.05], my[pn1 > 0.05], color='k', s=3, zorder=101)

#ax1.plot(mx[max_indn], 
#        my[max_indn], 'o', ms=30, mew=3, mec='k', mfc='None', zorder=104)


fig2.colorbar(cs1, cax=cax1)
cax1.set_ylabel('R value')


#extra = 1
#extents = np.array((lon.min() -  extra,
#               lon.max() + extra,
#               lat.min() - extra,
#               lat.max() + extra))
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
#m.fillcontinents(color='0.6', zorder=100)
m2.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)
m2.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)


mx, my = m2(lon, lat)
#triangles = nv.transpose() -1

# axes1

#ax1.tricontour(mx, my, triangles, dist_c, levels=[25, 50, 75], zorder=100)
cs2 = ax2.tripcolor(mx, my, triangles, rn2, vmin=-1, vmax=1, cmap=plt.get_cmap('bwr'), zorder=99)

rn2a = np.ma.ones(rn2.shape) - 0.5
rn2a[pn2 < 0.05] = 2
ax2.tricontourf(mx, my, triangles, rn2a, levels=[0, 1], vmin=0, vmax=2, cmap=plt.get_cmap('gray'), zorder=100, alpha=0.6)
ax2.tricontour(mx, my, triangles, rn2a, levels=[1], colors='k', linewidths=0.5, zorder=100)

#ax2.scatter(mx[pn2 > 0.05], my[pn2 > 0.05], color='k', s=3, zorder=101)

#ax1.plot(mx[max_indn], 
#        my[max_indn], 'o', ms=30, mew=3, mec='k', mfc='None', zorder=104)


#fig2.colorbar(cs2, cax=cax1)
#cax1.set_ylabel('R value')

ax1.annotate('(a)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig2.savefig('./Figures/vel_r_map_mn.png')

