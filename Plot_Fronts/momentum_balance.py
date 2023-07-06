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
import xarray as xr
import sys


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

mjas = '/scratch/benbar/JASMIN/'
fn_dy = sorted(glob.glob(mjas + 'Model_Output_V3.02/Daily/1993/SSWRS_V3.02*dy*RE.nc'))
fn_hr = sorted(glob.glob(mjas + 'Model_Output_V3.02/Hourly/1993/SSWRS_V3.02*hr*RE.nc'))
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

vars = ('zeta', 'temp', 'salinity', 'ua', 'va', 'v', 'u', 'Times', 'h', 'tauc')
dims = {'time':':'}

FVCOM = readFVCOM(fn_dy[0], vars, dims=dims)
t1 = 1

date_list = np.zeros((len(FVCOM['zeta'][:, 0]) -1), dtype=object)
du_dt = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
dv_dt = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
momentum_full_x = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
momentum_full_y = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))

wind_stress_x = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
wind_stress_y = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
bot_stress_x = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
bot_stress_y = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
coriolis_x = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
coriolis_y = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
momentum_elem_x = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
momentum_elem_y = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
du_dx = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
du_dy = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
dv_dx = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
dv_dy = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))


for t1 in range(3):#len(FVCOM['zeta'][:, 0]) - 1):
  date_list[t1] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-4].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S.%f')
  next_time = dt.datetime.strptime(''.join(FVCOM['Times'][t1 + 1, :-4].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S.%f')
  t_diff = next_time - date_list[t1]
  sec_diff = (t_diff.days * 24 * 60 * 60) + t_diff.seconds

  H = h_mod + FVCOM['zeta'][t1, :]
  Hc = hc_mod + fvcom.grid.nodes2elems(FVCOM['zeta'][t1, :], triangles)
  depth_mod = -H * siglay_mod # should be negative
  depthlev_mod = -H * siglev_mod # should be negative
  d_depth = depthlev_mod[:-1, :] - depthlev_mod[1:, :]
  print((depth_mod / H).shape)
  rho = calc_rho(FVCOM['salinity'][t1, :, :], FVCOM['temp'][t1, :, :], 
              depth_mod, lon, lat) * 1000 # convert to g/m^3
  pres = sw.p_from_z(depth_mod * -1, lat)
  pres_pa = pres * 10000# convert to Pa

  # variables

  f = sw.f(lat)
  fc = sw.f(latc)
  g = sw.grav(56, 0)
  rho_0 = 1025 * 1000 # convert to g/m^3

  a = tri.LinearTriInterpolator(trio, pres_pa[-1, :])
  dpb_dx, dpb_dy = a.gradient(x, y)

  rho_prime = rho - rho_0
  drho_dx = np.zeros(rho.shape)
  drho_dy = np.zeros(rho.shape)
  for z in range(len(siglay_mod)):
    a = tri.LinearTriInterpolator(trio, rho[z, :])
    drho_dx[z, :], drho_dy[z, :] = a.gradient(x, y)
  rho_prime_bot = rho_prime[-1, :]

  drho_dx = np.ma.masked_where(np.isnan(drho_dx), drho_dx)
  drho_dy = np.ma.masked_where(np.isnan(drho_dy), drho_dy)

  a = tri.LinearTriInterpolator(trio, -h_mod)
  dh_dx, dh_dy = a.gradient(x, y)

  u_node = fvcom.grid.elems2nodes(FVCOM['ua'][t1, :], triangles)
  v_node = fvcom.grid.elems2nodes(FVCOM['va'][t1, :], triangles)

  a = tri.LinearTriInterpolator(trio, u_node)
  du_dx_n, du_dy_n = a.gradient(xc, yc)
  a = tri.LinearTriInterpolator(trio, v_node)
  dv_dx_n, dv_dy_n = a.gradient(xc, yc)

  du_dx[t1, :] = fvcom.grid.nodes2elems(du_dx_n, triangles)
  du_dy[t1, :] = fvcom.grid.nodes2elems(du_dy_n, triangles)
  dv_dx[t1, :] = fvcom.grid.nodes2elems(dv_dx_n, triangles)
  dv_dy[t1, :] = fvcom.grid.nodes2elems(dv_dy_n, triangles)

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

    d_ind = np.nonzero(air_date >= dt.datetime(1993, 1, 1))[0][t1]

    u_wind = dataset.variables['uwind_speed'][d_ind, :]
    v_wind = dataset.variables['vwind_speed'][d_ind, :]
    air_temp = dataset.variables['air_temperature'][d_ind, :] + 273.15
    air_pres = dataset.variables['air_pressure'][d_ind, :]
    rel_humid = dataset.variables['relative_humidity'][d_ind, :]

  air_specific = 287.058 / 1000 # J/kg K to J/g K
  specific_humid = rel_humid * 2.541e6 * np.exp(-5415.0 / air_temp) * (18/29)
  air_rho = (((air_pres / (air_specific * air_temp)) * (1 + specific_humid)) 
        / (1 + (specific_humid * 1.609)))
  air_rho_elem = fvcom.grid.nodes2elems(air_rho, triangles)
  Cd = 0.0015
  tau_wind_u = air_rho_elem * Cd * np.abs(u_wind) * u_wind
  tau_wind_v = air_rho_elem * Cd * np.abs(v_wind) * v_wind

  # Momentum Balance x terms

  coriolis_x[t1, :] = fc * FVCOM['va'][t1, :]
  #bot_pres_grad_x = (-1 / rho_0) * dpb_dx 
  bot_pres = pres_pa[-1, :] - np.ma.sum(rho * g * d_depth, axis=0) + air_pres
  a = tri.LinearTriInterpolator(trio, bot_pres)
  bot_pres_grad_x, bot_pres_grad_y = a.gradient(x, y)
  #horiz_dens_grad_x = (g / (rho_0 * H)) * np.ma.sum(drho_dx * ((d_depth ** 2) / 2), axis=0)
  
  if 0:
    # Basic pressure gradient
    bot_pres = g * H * np.ma.mean(rho, axis=0)
    a = tri.LinearTriInterpolator(trio, bot_pres)
    horiz_dens_grad_x, horiz_dens_grad_y = a.gradient(x, y)
  elif 0:
    horiz_dens_grad_x = np.ma.zeros(bot_pres_grad_x.shape)
    horiz_dens_grad_y = np.ma.zeros(bot_pres_grad_x.shape)
    for i in range(depth_mod.shape[1]):
      horiz_dens_grad_x[i] = np.ma.sum(g * drho_dx[:, i] * d_depth[:, i], axis=0)
      horiz_dens_grad_y[i] = np.ma.sum(g * drho_dy[:, i] * d_depth[:, i], axis=0)
  else:
    a = tri.LinearTriInterpolator(trio, FVCOM['zeta'][t1, :])
    z_grad_x, z_grad_y = a.gradient(x, y)
    horiz_dens_grad_x = z_grad_x * g
    horiz_dens_grad_y = z_grad_y * g

  bot_stress_x[t1, :] = -tau_bot_u / (rho_0 * Hc)
  wind_stress_x[t1, :] = tau_wind_u / (rho_0 * Hc)

  #momentum_node_x = bot_pres_grad_x + horiz_dens_grad_x 
  #momentum_node_x = (1 / rho_0) * (horiz_dens_grad_x)# + bot_pres_grad_x)
  momentum_node_x = horiz_dens_grad_x 
  momentum_elem_x[t1, :] = fvcom.grid.nodes2elems(momentum_node_x, triangles)

  u_v_momentum = (FVCOM['ua'][t1, :] * du_dx) + (FVCOM['va'][t1, :] * du_dy)
  momentum_full_x[t1, :] = bot_stress_x[t1, :] + wind_stress_x[t1, :] + coriolis_x[t1, :] - momentum_elem_x[t1, :] #- u_v_momentum[t1, :]
  du_dt[t1, :] = (FVCOM['ua'][t1 + 1, :] - FVCOM['ua'][t1, :]) / sec_diff
  print(du_dt[t1, 1000], momentum_elem_x[t1, 1000], momentum_full_x[t1, 1000])
  #rho_variability = np.zeros(rho.shape)
  #for z in range(len(siglay_mod)):
  #  rho_variability[z, :] = ((drho_prime_dx * ((d_depth[z, :] ** 2) / 2))
  #      + (drho_prime_dx[z, :] * d_depth[z, :] * (-h_mod)))

  #bot_buoy_x = ((g * rho_prime_bot) / rho_0) * dh_dx


  # Momentum Balance y terms

  coriolis_y[t1, :] = -fc * FVCOM['ua'][t1, :]
  if 0:
    bot_pres_grad_y = (-1 / rho_0) * dpb_dy
    horiz_dens_grad_y = (g / (rho_0 * H)) * np.ma.sum(drho_dy * ((d_depth ** 2) / 2), axis=0)
  
  bot_stress_y[t1, :] = -tau_bot_v / (rho_0 * Hc)
  wind_stress_y[t1, :] = tau_wind_v / (rho_0 * Hc)

#  momentum_node_y = bot_pres_grad_y + horiz_dens_grad_y 
  #momentum_node_y = (1 / rho_0) * (horiz_dens_grad_y)#
  momentum_node_y = horiz_dens_grad_y 
  momentum_elem_y[t1, :] = fvcom.grid.nodes2elems(momentum_node_y, triangles)

  momentum_full_y[t1, :] =  bot_stress_y[t1, :] + wind_stress_y[t1, :] + coriolis_y[t1, :] - momentum_elem_y[t1, :]

  dv_dt[t1, :] = (FVCOM['va'][t1 + 1, :] - FVCOM['va'][t1, :]) / sec_diff
  print(dv_dt[t1, 1000], momentum_elem_y[t1, 1000], momentum_full_y[t1, 1000])


#print(np.mean(drho_dx), np.mean(d_depth), np.mean(dpb_dx))
#print(np.mean(momentum_full_y), np.mean(dv_dt))
momentum_full_y = np.ma.masked_where(np.isnan(momentum_full_y), momentum_full_y)
momentum_full_x = np.ma.masked_where(np.isnan(momentum_full_x), momentum_full_x)


date_list = date_list[:t1]
du_dt = du_dt[:t1, :]
dv_dt = dv_dt[:t1, :]
momentum_full_x = momentum_full_x[:t1, :]
momentum_full_y = momentum_full_y[:t1, :]

wind_stress_x = wind_stress_x[:t1, :]
wind_stress_y = wind_stress_y[:t1, :]
bot_stress_x = bot_stress_x[:t1, :]
bot_stress_y = bot_stress_y[:t1, :]
coriolis_x = coriolis_x[:t1, :]
coriolis_y = coriolis_y[:t1, :]
momentum_elem_x = momentum_elem_x[:t1, :]
momentum_elem_y = momentum_elem_y[:t1, :]


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


cs1 = ax1.tripcolor(mx, my, triangles, momentum_full_x[t1-1], vmin=-0.000003, vmax=0.000003, zorder=99)
ax1.set_title('Momentum Balance')

fig1.colorbar(cs1, cax=cax1)
cax1.set_ylabel('Acceleration (ms$^{-2}$)')

cs2 = ax2.tripcolor(mx, my, triangles, du_dt[t1-1], vmin=-0.000003, vmax=0.000003, zorder=99)
ax2.set_title('Du/Dt')

fig1.colorbar(cs2, cax=cax2)
cax2.set_ylabel('Acceleration (ms$^{-2}$)')




fig2 = plt.figure(figsize=(10, 5))
ax1 = fig2.add_axes([0.1, 0.55, 0.8, 0.35])
#ax2 = fig2.add_axes([0.1, 0.05, 0.8, 0.35])
ax2 = fig2.add_axes([0.1, 0.22, 0.8, 0.15])
ax3 = fig2.add_axes([0.1, 0.05, 0.8, 0.15])

xy = np.array(((-9, 48.8), (4, 61), (10, 57.9), (10, 58.5)))  
ele_ind = np.array([fvg.closest_element(i) for i in xy]).flatten().tolist()

ax1.plot(date_list, momentum_full_x[:, ele_ind[2]], label='Momentum')
ax1.plot(date_list, wind_stress_x[:, ele_ind[2]], label='Wind Stress')
ax1.plot(date_list, bot_stress_x[:, ele_ind[2]], label='Bottom Stress')
ax1.plot(date_list, coriolis_x[:, ele_ind[2]], label='Coriolis')
ax1.plot(date_list, momentum_elem_x[:, ele_ind[2]] * -1, label='Pressure Grad')
ax1.plot(date_list, du_dt[:, ele_ind[2]], label='Du/Dt')


ax2.plot(date_list, momentum_full_y[:, ele_ind[0]])
ax3.plot(date_list, dv_dt[:, ele_ind[0]])

ax1.legend()

fig1.savefig('./Figures/momentum_balance.png')
fig2.savefig('./Figures/momentum_time.png')
