#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PyFVCOM.read import FileReader 
from PyFVCOM.read import ncread as readFVCOM
import datetime as dt
import netCDF4 as nc
import glob
import PyFVCOM as fvcom
import pyproj
from shapely.geometry import LineString
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import cascaded_union
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
import scipy.signal as sig
import scipy.stats as stats
import df_regress
from PyFVCOM.grid import elems2nodes
import matplotlib.tri as tri
import matplotlib.colors as colors

#mjas = '/scratch/benbar/JASMIN/Model_Output/Daily/'
out_dir = '/scratch/benbar/Processed_Data_V3.02/'
fn = []
fn.extend(sorted(glob.glob(out_dir + 'SSWRS*V3.02*dy*RE.nc')))


fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'
river_nc_file = fin + 'SSW_Hindcast_riv_xe3_rivers.nc'
river_nml_file = fin + 'SSW_Hindcast_riv_xe3.nml'

# write pyfvcom code to load rivers

st_date = dt.datetime(1992, 12, 30, 0, 0, 0)
en_date = dt.datetime(1993, 1, 31, 23, 0, 0)
fvg = fvcom.preproc.Model(st_date, en_date, grid=fgrd, 
                    native_coordinates='spherical', zone='30N')

x = fvg.grid.x
y = fvg.grid.y
xc = fvg.grid.xc
yc = fvg.grid.yc
lon = fvg.grid.lon
lat = fvg.grid.lat
h = fvg.grid.h
triangles = fvg.grid.triangles

# Extract only the first 24 time steps.
dims = {'time': slice(0, 10)}
# List of the variables to extract.
#vars = ('lon', 'lat', 'latc', 'lonc', 'nv', 'zeta', 'temp', 'salinity', 'ua', 'va', 'siglay', 'h', 'Times', 'time', 'Itime', 'Itime2')
vars = ('Times', 'time', 'Itime', 'Itime2')

#fvf = readFVCOM(fn[-1], vars, dims=dims)
fvf = FileReader(fn[0], list(vars), dims=dims)
#for i in range(1, len(fn)):
#  fvf = FileReader(fn[i], list(vars), dims=dims) >> fvf

fvf.add_river_flow(river_nc_file, river_nml_file)

riv_lat = fvf.river.river_lat
riv_lon = fvf.river.river_lon
riv_x = x[fvf.river.river_nodes]
riv_y = y[fvf.river.river_nodes]
print(np.shape(fvf.river.river_fluxes)) # time, location
riv_lon[riv_lon > 180] = riv_lon[riv_lon > 180] - 360

river_flux = fvf.river.raw_fluxes


# load eof

season = 0
if season:
  sstr = '_season'
else:
  sstr = ''

data = np.load(out_dir + 'eof_data_mn' + sstr + '_sal.npz', allow_pickle=True)

s_var = data['variability']
s_eof = data['space_eof']
s_amp = data['amp_pc']
date_list = data['date']
lon = data['lon']
lat = data['lat']

data.close()

mask = s_eof == -1e20
s_eof[mask] = 0
s_eof = np.ma.masked_where(mask, s_eof)

# load residual current


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
date2 = data['date_list'][:-12]
ua2 = data['ua'][:-12, :]
va2 = data['va'][:-12, :] # time, space

data.close()

print(date2[0], date2[-1])
date = np.append(date, date2)
ua = np.append(ua, ua2, axis=0)
va = np.append(va, va2, axis=0) # time, space

u_mean = np.ma.mean(ua, axis=0)
v_mean = np.ma.mean(va, axis=0)

mag_r = (u_mean ** 2 + v_mean ** 2) ** 0.5
dir_r = np.ma.arctan2(v_mean, u_mean) # y, x

mag_r = np.ma.masked_where(mag_r == -999, mag_r)
dir_r = np.ma.masked_where(dir_r == -999, dir_r)
mag_n = elems2nodes(mag_r, triangles)
u_f = elems2nodes(mag_r * np.ma.cos(dir_r), triangles)
v_f = elems2nodes(mag_r * np.ma.sin(dir_r), triangles)

def plot_streamlines(ax, x, y, lon, lat, m, triangles, u, v):
    lon_g, lat_g, xg, yg = m.makegrid(100, 100, returnxy=True)
    print(np.min(lon_g), np.max(lon_g), np.min(lat_g), np.max(lat_g))
    trio = tri.Triangulation(lon, lat, triangles=np.asarray(triangles))
    interpolator_u = tri.LinearTriInterpolator(trio, u)
    interpolator_v = tri.LinearTriInterpolator(trio, v)

    grid_u = interpolator_u(lon_g, lat_g)
    grid_v = interpolator_v(lon_g, lat_g)

    ax.streamplot(xg, yg, grid_u, grid_v, density=(1, 2), color='w', linewidth=0.5, zorder=102)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

my_cmap = truncate_colormap(plt.get_cmap('GnBu'), 0.2, 1)

# Monthly means

yr = np.array([d.year for d in fvf.river.time_dt])
mnt = np.array([d.month for d in fvf.river.time_dt])
yri = np.unique(yr)
mnti =np.unique(mnt)

flux_mn = np.ma.zeros((len(yri) * len(mnti), river_flux.shape[1]))
date_mn = np.zeros((len(yri) * len(mnti)), dtype=object)
count = 0
for i in range(len(yri)):
  for j in range(len(mnti)):
    ind = (yr == yri[i]) & (mnt == mnti[j])
    flux_mn[count, :] = np.ma.mean(river_flux[ind, :], axis=0)
    date_mn[count] = dt.datetime(yri[i], mnti[j], 1)
    count += 1

# running mean

run = 12
riv_run = np.zeros((len(date_mn) - run, flux_mn.shape[1]))
run_date = np.empty(len(date_mn) - run, dtype=object)
for i in range(len(run_date)):
  riv_run[i, :] = np.ma.mean(flux_mn[i:i+run, :], axis=0)
  run_date[i] = date_mn[i + run // 2]

st_d = np.nonzero(run_date >= date_list[0])[0][0]
en_d = np.nonzero(run_date >= date_list[-1])[0][0] + 1
riv_run = riv_run[st_d:en_d, :]
riv_date = run_date[st_d:en_d]

#riv_run = sig.detrend(riv_run, axis=0, type='linear')

# Select area

extents = np.array((-10, 2, 54, 61))
sel_riv = (riv_lon > extents[0]) & (riv_lon < extents[1]) & (riv_lat > extents[2]) & (riv_lon < extents[3])

riv_run = riv_run[:, sel_riv]
riv_lon = riv_lon[sel_riv]
riv_lat = riv_lat[sel_riv]



r_value1 = np.zeros((riv_run.shape[1]))
r_value2 = np.zeros((riv_run.shape[1]))
p_value1 = np.zeros((riv_run.shape[1]))
p_value2 = np.zeros((riv_run.shape[1]))

for i in range(riv_run.shape[1]):
#  slope1, intercept1, r_value1[i], p_value1[i], se1 = df_regress.run_sig(riv_run[:, i], s_amp[:, 0], run=12)
#  slope1, intercept1, r_value2[i], p_value2[i], se1 = df_regress.run_sig(riv_run[:, i], s_amp[:, 1], run=12)
  slope1, intercept1, r_value1[i], p_value1[i], se1 = df_regress.edf_sig(riv_run[:, i], s_amp[:, 0])
  slope1, intercept1, r_value2[i], p_value2[i], se1 = df_regress.edf_sig(riv_run[:, i], s_amp[:, 1])

r_value1[np.isnan(r_value1)] = 0
r_value2[np.isnan(r_value2)] = 0

slope1, intercept1, r_value3, p_value3, se1 = df_regress.edf_sig(np.ma.sum(riv_run, axis=1), s_amp[:, 0])

print(np.max(r_value1), np.min(r_value1))
print(np.max(r_value2), np.min(r_value2))
print(r_value3)



if ((np.max(r_value2) ** 2) ** 0.5) > ((np.min(r_value2) ** 2) ** 0.5):
  max_ind = r_value2 == np.max(r_value2)
else:
  max_ind = r_value2 == np.min(r_value2)

riv_max = riv_run[:, max_ind]

#riv_max = np.sum(riv_run[:, r_value > 0.6], axis=1)[:, np.newaxis]


np.savez(out_dir + 'riv_mean_mn', riv_max=riv_max, riv_date=riv_date)
print(riv_lon[max_ind], riv_lat[max_ind])
#np.savez(out_dir + 'riv_rvalue', mxr, myr, r_value2, p_value2, lon, lat, triangles, h)


one = 0
if one:
  fig1 = plt.figure(figsize=(8, 8))  # size in inches
  ax2 = fig1.add_axes([0.05, 0.1, 0.8, 0.8])
  cax1 = fig1.add_axes([0.87, 0.3, 0.01, 0.4])

else:
  fig1 = plt.figure(figsize=(8, 14))  # size in inches
  ax1 = fig1.add_axes([0.05, 0.8, 0.35, 0.17])
  ax2 = fig1.add_axes([0.525, 0.8, 0.35, 0.17])
  cax1 = fig1.add_axes([0.41, 0.8, 0.01, 0.17])
  cax2 = fig1.add_axes([0.91, 0.8, 0.01, 0.17])

if one == 0:


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


  if 0:
    mx, my = m1(lon, lat)
    mxr, myr = m1(riv_lon, riv_lat)

  # axes1

  #ax1.tricontour(mx, my, triangles, h, levels=[5, 100, 200], zorder=100)
    ax1.tripcolor(mx, my, triangles, h, zorder=99)
    cs1 = ax1.scatter(mxr, myr, c='k', zorder=102, s=10)
    cs1 = ax1.scatter(mxr[p_value1 < 0.05], myr[p_value1 < 0.05], c=r_value1[p_value1 < 0.05], vmin=-0.8, vmax=0.8, cmap=plt.get_cmap('bwr'), zorder=103, s=10)

  else:
    x1, y1 = m1(lon, lat)

    ax1.tricontour(x1, y1, triangles, h, levels=[100], colors='k', zorder=102)
    cs1 = ax1.tripcolor(x1, y1, triangles, mag_n, vmin=0, vmax=0.2, zorder=99)
    fig1.colorbar(cs1, cax=cax1, extend='max')
    cax1.set_ylabel('Residual Current (m/s)')

    plot_streamlines(ax1, x, y, lon, lat, m1, triangles, u_f, v_f)

    ax1.annotate('(a)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)


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
mxr, myr = m2(riv_lon, riv_lat)
#triangles = nv.transpose() -1

# axes1

#ax1.tricontour(mx, my, triangles, h, levels=[5, 100, 200], zorder=100)
ax2.tricontour(mx, my, triangles, h, levels=[100], colors='k', zorder=102)
ax2.tripcolor(mx, my, triangles, h, cmap=my_cmap, vmin=0, vmax=300, zorder=99)
cs2 = ax2.scatter(mxr, myr, c='k', zorder=102, s=10)
cs2 = ax2.scatter(mxr[p_value2 < 0.05], myr[p_value2 < 0.05], c=r_value2[p_value2 < 0.05], vmin=-0.8, vmax=0.8, cmap=plt.get_cmap('bwr'), zorder=103, s=10)

fig1.colorbar(cs2, cax=cax2, extend='both')
cax2.set_ylabel('R value')

ax2.annotate('(b)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)


fig1.savefig('./Figures/riv_r_map_mn.png')

