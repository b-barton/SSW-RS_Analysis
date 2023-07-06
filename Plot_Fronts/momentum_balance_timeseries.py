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

"""
For filtering tide:

import numpy as np
from oceans import lanc

freq = 1./40  # Hours
window_size = 96+1+96
pad = np.zeros(window_size) * np.NaN

wt = lanc(window_size, freq)
res = np.convolve(wt, df['v'], mode='same')

df['low'] = res
df['high'] = df['v'] - df['low']


"""

def calc_rho(sp, tp, depth, lon, lat):
  pres = sw.p_from_z(depth, lat)
  sa = sw.SA_from_SP(sp, pres, lon, lat)
  ct = sw.CT_from_pt(sa, tp)
  rho = sw.rho(sa, ct, 0)
  return rho

fn_dy = sorted(glob.glob('/scratch/Scottish_waters_FVCOM/SSW_RS/SSW_Reanalysis_v1.1_1993_12_25/SSWRS_V1.1*dy*RE.nc'))
fn_hr = sorted(glob.glob('/scratch/Scottish_waters_FVCOM/SSW_RS/SSW_Reanalysis_v1.1_1993_12_25/SSWRS_V1.1*hr*RE.nc'))
#fn_hr = sorted(glob.glob('/scratch/Scottish_waters_FVCOM/SSW_RS/SSW_Reanalysis_v1.1_1997_12_06/out_file.nc'))

print(len(fn_hr))
out_dir = '/scratch/benbar/Processed_Data/'

forcing_dir = '/scratch/benbar/Forcing/'

fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'

# Extract only the first 24 time steps.
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
nbve = FVCOM['nbve'].transpose() -1 # elems around node
nbsn = FVCOM['nbsn'].transpose() -1 # nodes around node
nbe = FVCOM['nbe'].transpose() -1 # elems around elem
ntsn = FVCOM['ntsn'] # number nodes around node
ntve = FVCOM['ntve'] # number of elems around node

art1 = FVCOM['art1'] # area of node-base control volume
art2 = FVCOM['art2'] # area of elements around node

st_date = dt.datetime(1992, 12, 30, 0, 0, 0)
en_date = dt.datetime(1993, 1, 31, 23, 0, 0)

fvg = fvcom.preproc.Model(st_date, en_date, grid=fgrd, 
                    native_coordinates='spherical', zone='30N')
x = fvg.grid.x
y = fvg.grid.y
xc = fvg.grid.xc
yc = fvg.grid.yc

trio = tri.Triangulation(x, y, triangles=np.asarray(triangles))

# Load multiple files to make timersries we can filter

# Positions we're interested in plotting. The find_nearest_point 
# function will find the closest node in the unstructured grid.
# lon, lat pairs
xy = np.array(((-9, 48.8), (4, 61), (10, 57.9), (10, 58.5))) 

dims = {'siglay': [-1]}
varlist = ['zeta', 'temp', 'salinity', 'ua', 'va', 'v', 'u', 'Times']

fvcom1 = MFileReader(fn_hr[0], variables=varlist, dims=dims)
node_ind = np.array([fvcom1.closest_node(i) for i in xy]).flatten().tolist()
ele_ind = np.array([fvcom1.closest_element(i) for i in xy]).flatten().tolist()

dims = {'siglay': [0], 'node': node_ind, 'nele': ele_ind}
fvcom1 = MFileReader(fn_hr, variables=varlist, dims=dims)

dates = np.zeros((len(fvcom1.data.Times)), dtype=object)
for i in range(len(fvcom1.data.Times)):
  dates[i] = dt.datetime.strptime(''.join(fvcom1.data.Times[i].astype(str))[:-7],'%Y-%m-%dT%H:%M:%S')

for c, ind in enumerate(node_ind):
  
  # Filtering of the time series
  fs=1/24/3600 #1 day in Hz (sampling frequency)

  nyquist = fs / 2 # 0.5 times the sampling frequency
  cutoff=0.1 # fraction of nyquist frequency, here  it is 5 days
  print('cutoff= ',1/cutoff*nyquist*24*3600,' days') #cutoff=  4.999999999999999  days
  b, a = signal.butter(5, cutoff, btype='lowpass') #low pass filter
  dzeta_filt = signal.filtfilt(b, a, fvcom1.data.zeta[:, c])
  dzeta_filt = np.array(dzeta_filt).transpose()

  plt.plot(dates, dzeta_filt)
  plt.plot(dates, fvcom1.data.zeta)
  plt.show()


vars = ('zeta', 'temp', 'salinity', 'ua', 'va', 'v', 'u', 'Times', 'h', 'tauc')
dims = {'time':':'}

FVCOM = readFVCOM(fn_dy[0], vars, dims=dims)
t1 = 1

date_list = np.zeros((len(FVCOM['zeta'][:, 0]) -1), dtype=object)
du_dt = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
dv_dt = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
momentum_full_x = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
momentum_full_y = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))

for t1 in range(len(FVCOM['zeta'][:, 0]) - 1):
  date_list[t1] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-4].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S.%f')
  next_time = dt.datetime.strptime(''.join(FVCOM['Times'][t1 + 1, :-4].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S.%f')
  t_diff = next_time - date_list[t1]
  sec_diff = (t_diff.days * 24 * 60 * 60) + t_diff.seconds

  H = h_mod + FVCOM['zeta'][t1, :]
  Hc = hc_mod + fvcom.grid.nodes2elems(FVCOM['zeta'][t1, :], triangles)
  depth_mod = -H * siglay_mod # should be negative
  depthlev_mod = -H * siglev_mod # should be negative
  d_depth = depthlev_mod[:-1, :] - depthlev_mod[1:, :]

  rho = calc_rho(FVCOM['salinity'][t1, :, :], FVCOM['temp'][t1, :, :], 
              depth_mod, lon, lat)  
  pres = sw.p_from_z(depth_mod * -1, lat)
  pres_pa = pres * 10000# convert to Pa

  # variables

  f = sw.f(lat)
  fc = sw.f(latc)
  g = sw.grav(56, 0)
  rho_0 = 1025

  a = tri.LinearTriInterpolator(trio, pres_pa[-1, :])
  dpb_dx, dpb_dy = a.gradient(x, y)

  rho_prime = rho - rho_0
  drho_dx = np.zeros(rho.shape)
  drho_dy = np.zeros(rho.shape)
  for z in range(len(siglay_mod)):
    a = tri.LinearTriInterpolator(trio, rho[z, :])
    drho_dx[z, :], drho_dy[z, :] = a.gradient(x, y)
  rho_prime_bot = rho_prime[-1, :]

  a = tri.LinearTriInterpolator(trio, -h_mod)
  dh_dx, dh_dy = a.gradient(x, y)

  tau_bot = FVCOM['tauc'][t1, :]

  #Cd_bot_u = (((tau_bot ** 2) / 2) ** 0.5) / (-rho_0 * np.abs(FVCOM['u'][t1, -1, :]) * FVCOM['u'][t1, -1, :])
  #Cd_bot_v = (((tau_bot ** 2) / 2) ** 0.5) / (-rho_0 * np.abs(FVCOM['v'][t1, -1, :]) * FVCOM['v'][t1, -1, :])

  Cd_bot = 0.0015
  tau_bot_u = -rho_0 * Cd_bot * np.abs(FVCOM['u'][t1, -1, :]) * FVCOM['u'][t1, -1, :]
  tau_bot_v = -rho_0 * Cd_bot * np.abs(FVCOM['v'][t1, -1, :]) * FVCOM['v'][t1, -1, :]


  # Wind

  with nc.Dataset(forcing_dir 
        + 'SSW_Hindcast_metforcing_ERA5_1993.nc', 'r') as dataset:
    Itime = dataset.variables['Itime'][:]
    Itime2 = dataset.variables['Itime2'][:]

    ref = dt.datetime(1858, 11, 17)
    air_date = np.zeros((len(Itime)), dtype=object)
    for i in range(len(air_date)):
      air_date[i] = (ref + dt.timedelta(days=int(Itime[i])) 
          + dt.timedelta(seconds=int(Itime2[i]/1000)))

    d_ind = np.nonzero(air_date >= dt.datetime(1997, 12, 6))[0][t1]

    u_wind = dataset.variables['uwind_speed'][d_ind, :]
    v_wind = dataset.variables['vwind_speed'][d_ind, :]
    air_temp = dataset.variables['air_temperature'][d_ind, :] + 273.15
    air_pres = dataset.variables['air_pressure'][d_ind, :]
    rel_humid = dataset.variables['relative_humidity'][d_ind, :]

  air_specific = 287.058 # J/kg K
  specific_humid = rel_humid * 2.541e6 * np.exp(-5415.0 / air_temp) * (18/29)
  air_rho = (((air_pres / (air_specific * air_temp)) * (1 + specific_humid)) 
        / (1 + (specific_humid * 1.609)))
  air_rho_elem = fvcom.grid.nodes2elems(air_rho, triangles)
  Cd = 0.0015
  tau_wind_u = air_rho_elem * Cd * np.abs(u_wind) * u_wind
  tau_wind_v = air_rho_elem * Cd * np.abs(v_wind) * v_wind

  # Momentum Balance x terms

  coriolis_x = fc * FVCOM['va'][t1, :]
  bot_pres_grad_x = (-1 / rho_0) * dpb_dx
  horiz_dens_grad_x = (g / (rho_0 * H)) * np.sum(drho_dx * ((d_depth ** 2) / 2), axis=0)
  bot_stress = -tau_bot_u / (rho_0 * Hc)
  wind_stress_x = tau_wind_u / (rho_0 * Hc)

  momentum_node_x = bot_pres_grad_x + horiz_dens_grad_x 
  momentum_elem_x = fvcom.grid.nodes2elems(momentum_node_x, triangles)

  momentum_full_x[t1, :] = coriolis_x - momentum_elem_x + bot_stress + wind_stress_x
  du_dt[t1, :] = (FVCOM['ua'][t1 + 1, :] - FVCOM['ua'][t1, :]) / sec_diff

  #rho_variability = np.zeros(rho.shape)
  #for z in range(len(siglay_mod)):
  #  rho_variability[z, :] = ((drho_prime_dx * ((d_depth[z, :] ** 2) / 2))
  #      + (drho_prime_dx[z, :] * d_depth[z, :] * (-h_mod)))

  #bot_buoy_x = ((g * rho_prime_bot) / rho_0) * dh_dx


  # Momentum Balance y terms

  coriolis_y = -fc * FVCOM['ua'][t1, :]
  bot_pres_grad_y = (-1 / rho_0) * dpb_dy
  horiz_dens_grad_y = (g / (rho_0 * H)) * np.sum(drho_dy * ((d_depth ** 2) / 2), axis=0)
  bot_stress = -tau_bot_v / (rho_0 * Hc)
  wind_stress_y = tau_wind_v / (rho_0 * Hc)

  momentum_node_y = bot_pres_grad_y + horiz_dens_grad_y 
  momentum_elem_y = fvcom.grid.nodes2elems(momentum_node_y, triangles)

  momentum_full_y[t1, :] = coriolis_y - momentum_elem_y + bot_stress + wind_stress_y

  dv_dt[t1, :] = (FVCOM['va'][t1 + 1, :] - FVCOM['va'][t1, :]) / sec_diff


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


cs1 = ax1.tripcolor(mx, my, triangles, momentum_full_x[0], vmin=-0.00005, vmax=0.00005, zorder=99)

fig1.colorbar(cs1, cax=cax1)
cax1.set_ylabel('Acceleration (ms$^{-2}$)')

cs2 = ax2.tripcolor(mx, my, triangles, du_dt[0], vmin=-0.00003, vmax=0.00003, zorder=99)

fig1.colorbar(cs2, cax=cax2)
cax2.set_ylabel('Acceleration (ms$^{-2}$)')




fig2 = plt.figure(figsize=(10, 5))
ax1 = fig2.add_axes([0.1, 0.55, 0.8, 0.35])
ax2 = fig2.add_axes([0.1, 0.05, 0.8, 0.35])

xy = np.array(((-9, 48.8), (4, 61), (10, 57.9), (10, 58.5)))  
ele_ind = np.array([fvg.closest_element(i) for i in xy]).flatten().tolist()

ax1.plot(date_list, momentum_full_x[:, ele_ind[0]])
ax1.plot(date_list, du_dt[:, ele_ind[0]])

ax2.plot(date_list, momentum_full_y[:, ele_ind[0]])
ax2.plot(date_list, dv_dt[:, ele_ind[0]])

fig1.savefig('./Figures/momentum_balance.png')
fig2.savefig('./Figures/momentum_time.png')
