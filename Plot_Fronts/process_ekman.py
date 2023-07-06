#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
from PyFVCOM.read import ncread as readFVCOM
from PyFVCOM import physics
from PyFVCOM.read import MFileReader
import datetime as dt
import netCDF4 as nc
import glob
import matplotlib.tri as tri
import PyFVCOM as fvcom
import gsw as sw
import xarray as xr
import sys


"""
Details of the 2D momentum balance equation are in the FVCOM manual page 9.
"""


def calc_wind_tau(forcing_file, use_date):
  # wind stress

  with nc.Dataset(forcing_file, 'r') as dataset:
    Itime = dataset.variables['Itime'][:]
    Itime2 = dataset.variables['Itime2'][:]

    ref = dt.datetime(1858, 11, 17)
    air_date = np.zeros((len(Itime)), dtype=object)
    for i in range(len(air_date)):
      air_date[i] = (ref + dt.timedelta(days=int(Itime[i])) 
          + dt.timedelta(seconds=int(Itime2[i]/1000)))

    d_ind = np.nonzero(air_date >= use_date)[0][0]

    u_wind = dataset.variables['uwind_speed'][d_ind, :]
    v_wind = dataset.variables['vwind_speed'][d_ind, :]
    air_temp = dataset.variables['air_temperature'][d_ind, :] + 273.15
    air_pres = dataset.variables['air_pressure'][d_ind, :]
    rel_humid = dataset.variables['relative_humidity'][d_ind, :]

  air_specific = 287.058 #/ 1000 # J/kg K to J/g K # not needed
  specific_humid = rel_humid * 2.541e6 * np.exp(-5415.0 / air_temp) * (18/29)
  air_rho = (((air_pres / (air_specific * air_temp)) * (1 + specific_humid)) 
        / (1 + (specific_humid * 1.609)))
  air_rho_elem = fvcom.grid.nodes2elems(air_rho, triangles)
  Cd = 0.0015
  tau_wind_u = air_rho_elem * Cd * (((u_wind ** 2) + (v_wind ** 2)) ** 0.5) * u_wind
  tau_wind_v = air_rho_elem * Cd * (((u_wind ** 2) + (v_wind ** 2)) ** 0.5) * v_wind

  return tau_wind_u, tau_wind_v, air_pres



mjas = '/scratch/benbar/JASMIN/'
fn_dy = sorted(glob.glob(mjas + 'Model_Output_V3.02/Daily/1999/SSWRS_V3.02*dy*RE.nc'))
fn_hr = sorted(glob.glob(mjas + 'Model_Output_V3.02/Hourly/1999/SSWRS_V3.02*hr*RE.nc'))
forcing_dir = '/scratch/benbar/Forcing/'
fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'
out_dir = '/scratch/benbar/Processed_Data_V3.02/'
print(len(fn_hr))

dims = {'time':':'}
# List of the variables to extract.
vars = ('lon', 'lat', 'latc', 'lonc', 'nv', 'zeta', 'temp', 'salinity', 'ua', 'va', 'siglay', 'siglev', 'h', 'h_center', 'Itime', 'Itime2', 'nbve', 'nbsn', 'nbe', 'ntsn', 'ntve', 'art1', 'art2')
FVCOM = readFVCOM(fn_hr[-1], vars, dims=dims)

# Find the domain extents.
I=np.where(FVCOM['lon'] > 180) # MICDOM: for Scottish shelf domain
FVCOM['lon'][I]=FVCOM['lon'][I]-360 # MICDOM: for Scottish shelf domain
I=np.where(FVCOM['lonc'] > 180) # MICDOM: for Scottish shelf domain
FVCOM['lonc'][I]=FVCOM['lonc'][I]-360 # MICDOM: for Scottish shelf domain


lon = FVCOM['lon']
lat = FVCOM['lat']
lonc = FVCOM['lonc']
latc = FVCOM['latc']
siglay_mod = FVCOM['siglay'][:]
siglev_mod = FVCOM['siglev'][:]
h_mod = FVCOM['h'][:] # h_mod is positive
hc_mod = FVCOM['h_center'][:]

siglev_mod.mask = np.zeros(siglev_mod.shape, dtype=bool)
siglev_mod[siglev_mod < -1] = -1

triangles = FVCOM['nv'].transpose() - 1
nv = FVCOM['nv'] -1 # nodes around elem

st_date = dt.datetime(1992, 12, 30, 0, 0, 0)
en_date = dt.datetime(1993, 1, 31, 23, 0, 0)

fvg = fvcom.preproc.Model(st_date, en_date, grid=fgrd, 
                    native_coordinates='spherical', zone='30N')
x = fvg.grid.x
y = fvg.grid.y
xc = fvg.grid.xc
yc = fvg.grid.yc
trio = tri.Triangulation(x, y, triangles=np.asarray(triangles))

# Load transect elemts
data = np.load(out_dir + 'elem_transect.npz')

elem_list = data['elem_list']
data.close()

# Setup Greens's theory gradient

varlist = ['temp', 'salinity']
dims = {'siglay': [0]}
fvcom_files = MFileReader(fn_dy[0], variables=varlist, dims=dims)
fvcom_files.grid.lon[fvcom_files.grid.lon > 180] = (fvcom_files.grid.lon[
fvcom_files.grid.lon > 180] - 360)


# read different variables in the loop

vars = ('zeta', 'v', 'u', 'Times', 'h', 'tauc')
dims = {'time':':'}

f = sw.f(lat)
fc = sw.f(latc)
g = sw.grav(56, 0)
rho_0 = 1025

isize = FVCOM['ua'].shape[1]
print(isize, len(FVCOM['zeta'][:, 0]) * len(fn_hr))

ekman_u = np.zeros(lon.shape)
ekman_v = np.zeros(lon.shape)
date_list = np.zeros((len(FVCOM['zeta'][:, 0]) * len(fn_hr)), dtype=object)
count = 0

for f in range(1):#st_f, len(fn_dy)):
  FVCOM = readFVCOM(fn_hr[f], vars, dims=dims)

  for t1 in range(1):#len(FVCOM['zeta'][:, 0])):

    date_list[count] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-4].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S.%f')

    print(t1, date_list[count])

    H = (h_mod + FVCOM['zeta'][t1, :])
    Hc = (hc_mod + fvcom.grid.nodes2elems(FVCOM['zeta'][t1, :], triangles))
    print(np.mean(Hc))

    # Wind

    tau_wind_u, tau_wind_v, air_pres = calc_wind_tau(forcing_dir 
          + 'SSW_Hindcast_metforcing_ERA5_1999.nc', date_list[count])

    wind_stress_x = tau_wind_u / (rho_0 * Hc)
    wind_stress_y = tau_wind_v / (rho_0 * Hc)

    ekman_u = wind_stress_y / f
    ekman_v = wind_stress_x / -f
    count = count + 1


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


cs1 = ax1.tripcolor(mx, my, triangles, ekman_u, vmin=0, vmax=1e-5, zorder=99)

fig1.colorbar(cs1, cax=cax1)
cax1.set_ylabel('U Velocity (ms$^{-1}$)')

cs2 = ax2.tripcolor(mx, my, triangles, (ekman_u**2 + ekman_v**2)**0.5, vmin=0, vmax=1e-5, zorder=99)

fig1.colorbar(cs2, cax=cax2)
cax2.set_ylabel('V Velocity (ms$^{-1}$)')


fig1.savefig('./Figures/ekman.png')

