#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PyFVCOM.read import ncread as readFVCOM
from PyFVCOM.read import MFileReader
from PyFVCOM.plot import Time
import datetime as dt
import netCDF4 as nc
import glob
import matplotlib.tri as tri
import PyFVCOM as fvcom
import gsw as sw
import xarray as xr
from scipy import signal


def calc_rho(sal, tp, depth, lon, lat, prac=False):
  pres = sw.p_from_z(depth, lat)
  if prac:
    sa = sw.SA_from_SP(sp, pres, lon, lat)
  else:
    sa = sal
  ct = sw.CT_from_pt(sa, tp)
  rho = sw.rho(sa, ct, 0)
  return rho



mjas = '/scratch/benbar/JASMIN/Model_Output_V3.02/Daily/'
fn_dy = []
for yr in range(1999, 2000):
  fn_dy.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V3.02*dy*RE.nc')))

out_dir = '/scratch/benbar/Processed_Data_V3.02/'

fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'

dims = {'time':':10'}
# List of the variables to extract.
vars = ('lon', 'lat', 'latc', 'lonc', 'nv', 'zeta', 'temp', 'salinity', 'ua', 'va', 'siglay', 'siglev', 'h', 'h_center', 'Itime', 'Itime2', 'nbve', 'nbsn', 'nbe', 'ntsn', 'ntve', 'art1', 'art2')
FVCOM = readFVCOM(fn_dy[0], vars, dims=dims)

# Create the triangulation table array (with Python indexing
# [zero-based])
triangles = FVCOM['nv'].transpose() - 1
# Find the domain extents.
I=np.where(FVCOM['lon'] > 180) # MICDOM: for Scottish shelf domain
FVCOM['lon'][I]=FVCOM['lon'][I]-360 # MICDOM: for Scottish shelf domain
I=np.where(FVCOM['lonc'] > 180) # MICDOM: for Scottish shelf domain
FVCOM['lonc'][I]=FVCOM['lonc'][I]-360 # MICDOM: for Scottish shelf domain
extents = np.array((FVCOM['lon'].min(),
                  FVCOM['lon'].max(),
                   FVCOM['lat'].min(),
                   FVCOM['lat'].max()))

lon = FVCOM['lon']
lat = FVCOM['lat']
lonc = FVCOM['lonc']
latc = FVCOM['latc']
siglay_mod = FVCOM['siglay'][:]
siglev_mod = FVCOM['siglev'][:]
h_mod = FVCOM['h'][:] # h_mod is positive
hc_mod = FVCOM['h_center'][:]

nv = FVCOM['nv'] -1 # nodes around elem

if 1:
  st_date = dt.datetime(1992, 12, 30, 0, 0, 0)
  en_date = dt.datetime(1993, 1, 31, 23, 0, 0)

  fvg = fvcom.preproc.Model(st_date, en_date, grid=fgrd, 
                      native_coordinates='spherical', zone='30N')
  x = fvg.grid.x
  y = fvg.grid.y
  xc = fvg.grid.xc
  yc = fvg.grid.yc
  np.savez(out_dir + 'sswrs_xy.npz', x=x, y=y, xc=xc, yc=yc)

else:
  data = np.load(out_dir + 'sswrs_xy.npz')
  x = data['x']
  y = data['y']
  xc = data['xc']
  yc = data['yc']
  data.close()

trio = tri.Triangulation(x, y, triangles=np.asarray(triangles))

# constants

f = sw.f(lat)
fc = sw.f(latc)
g = sw.grav(56, 0)
rho_0 = 1025

#for t1 in range(len(FVCOM['zeta'][:, 0]) - 1):
t1 = 0

H = h_mod + FVCOM['zeta'][t1, :]
Hc = hc_mod + fvcom.grid.nodes2elems(FVCOM['zeta'][t1, :], triangles)
depth_mod = -H * siglay_mod # should be negative
depthlev_mod = -H * siglev_mod # should be negative
d_depth = depthlev_mod[:-1, :] - depthlev_mod[1:, :]

rho = calc_rho(FVCOM['salinity'][t1, :, :], FVCOM['temp'][t1, :, :], 
              depth_mod, lon, lat) 

thermal_wind_u = np.zeros(rho.shape)
thermal_wind_v = np.zeros(rho.shape)

for i in range(rho.shape[0]-1, -1, -1): # loop over depth from bottom upwards
  print(i)
  a = tri.LinearTriInterpolator(trio, rho[i, :])
  drho_dx, drho_dy = a.gradient(x, y)

  tw_u = drho_dy * (g / (rho_0 * f))
  tw_v = -drho_dx * (g / (rho_0 * f))

  # integrate upwards

  if i == rho.shape[0]-1:
    thermal_wind_u[i, :] = 0 + tw_u
    thermal_wind_v[i, :] = 0 + tw_v
  else:
    thermal_wind_u[i, :] = thermal_wind_u[i +1, :] + tw_u
    thermal_wind_v[i, :] = thermal_wind_v[i +1, :] + tw_v

clin_u = np.sum(thermal_wind_u * -d_depth, axis=0)
clin_v = np.sum(thermal_wind_v * -d_depth, axis=0)

# Plot

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_axes([0.05, 0.1, 0.35, 0.8])
ax2 = fig1.add_axes([0.55, 0.1, 0.35, 0.8])
cax1 = fig1.add_axes([0.41, 0.3, 0.01, 0.4])
cax2 = fig1.add_axes([0.91, 0.3, 0.01, 0.4])

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


cs1 = ax1.tripcolor(mx, my, triangles, clin_u, vmin=-5, vmax=5, zorder=99)

fig1.colorbar(cs1, cax=cax1)
cax1.set_ylabel('u Velocity (ms$^{-1}$)')

cs2 = ax2.tripcolor(mx, my, triangles, (clin_u**2 + clin_v**2)**0.5, vmin=0, vmax=10, zorder=99)

fig1.colorbar(cs2, cax=cax2)
cax2.set_ylabel('V Velocity (ms$^{-1}$)')


fig1.savefig('./Figures/thermal_wind.png')

