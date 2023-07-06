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


def calc_rho(sp, tp, depth, lon, lat):
  # density 
  pres = sw.p_from_z(depth, lat)
  sa = sw.SA_from_SP(sp, pres, lon, lat)
  ct = sw.CT_from_pt(sa, tp)
  rho = sw.rho(sa, ct, pres)
  print('This should be False:', np.ma.is_masked(pres))
  return rho

def calc_wind_tau(forcing_file, use_date, t1):
  # wind stress

  with nc.Dataset(forcing_file, 'r') as dataset:
    Itime = dataset.variables['Itime'][:]
    Itime2 = dataset.variables['Itime2'][:]

    ref = dt.datetime(1858, 11, 17)
    air_date = np.zeros((len(Itime)), dtype=object)
    for i in range(len(air_date)):
      air_date[i] = (ref + dt.timedelta(days=int(Itime[i])) 
          + dt.timedelta(seconds=int(Itime2[i]/1000)))

    d_ind = np.nonzero(air_date >= use_date)[0][t1]

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
  tau_wind_u = air_rho_elem * Cd * (((u_wind ** 2) + (v_wind ** 2)) ** 0.5) * u_wind
  tau_wind_v = air_rho_elem * Cd * (((u_wind ** 2) + (v_wind ** 2)) ** 0.5) * v_wind

  return tau_wind_u, tau_wind_v, air_pres

def calc_uv(ua, va, trio, triangles, fvcom_files):
  # partial momentum terms


  if 0:
    u_node = fvcom.grid.elems2nodes(ua, triangles)
    v_node = fvcom.grid.elems2nodes(va, triangles)
    a = tri.LinearTriInterpolator(trio, u_node)
    du_dx_n, du_dy_n = a.gradient(xc, yc)
    a = tri.LinearTriInterpolator(trio, v_node)
    dv_dx_n, dv_dy_n = a.gradient(xc, yc)

    du_dx = fvcom.grid.nodes2elems(du_dx_n, triangles)
    du_dy = fvcom.grid.nodes2elems(du_dy_n, triangles)
    dv_dx = fvcom.grid.nodes2elems(dv_dx_n, triangles)
    dv_dy = fvcom.grid.nodes2elems(dv_dy_n, triangles)

  else: 
    # Use Greens
    du = physics.green_gauss_gradient_method(ua, fvcom_files)
    du_dx = du[:, 0]
    du_dy = du[:, 1]
    dv = physics.green_gauss_gradient_method(va, fvcom_files)
    dv_dx = dv[:, 0]
    dv_dy = dv[:, 1]


  u_v_momentum_x = ((ua * du_dx) + (va * du_dy))
  u_v_momentum_y = ((ua * dv_dx) + (va * dv_dy))
  return u_v_momentum_x, u_v_momentum_y

def calc_pres_grad(zeta, x, y, g, trio, triangles, fvcom_files):
  # pressure gradient force

  if 1:
    a = tri.LinearTriInterpolator(trio, zeta)
    z_grad_x, z_grad_y = a.gradient(x, y)
    horiz_dens_grad_x = z_grad_x * g
    horiz_dens_grad_y = z_grad_y * g

    pres_grad_elem_x = fvcom.grid.nodes2elems(horiz_dens_grad_x, triangles)
    pres_grad_elem_y = fvcom.grid.nodes2elems(horiz_dens_grad_y, triangles)

  else: 
    # Use Greens
    zeta_e = fvcom.grid.nodes2elems(zeta, triangles)
    dzeta = physics.green_gauss_gradient_method(zeta_e, fvcom_files)
    z_grad_x = dzeta[:, 0]
    z_grad_y = dzeta[:, 1]

    pres_grad_elem_x = z_grad_x * g
    pres_grad_elem_y = z_grad_y * g

  return pres_grad_elem_x, pres_grad_elem_y

def buoy_slope_pres_grad(rho, rho_0, g, d_depth, air_pres, pres_pa, x, y, trio):
  buoy_pres = np.ma.sum(rho * d_depth, axis=0) * g
  a = tri.LinearTriInterpolator(trio, buoy_pres)
  buoy_pres_grad_x, buoy_pres_grad_y = a.gradient(x, y)
  buoy_pres_grad_x = buoy_pres_grad_x * (1 / rho_0)
  buoy_pres_grad_y = buoy_pres_grad_y * (1 / rho_0)
  buoy_pres_elem_x = fvcom.grid.nodes2elems(buoy_pres_grad_x, triangles)
  buoy_pres_elem_y = fvcom.grid.nodes2elems(buoy_pres_grad_y, triangles)

  slope_pres = pres_pa[-1, :] - buoy_pres + air_pres
  a = tri.LinearTriInterpolator(trio, slope_pres)
  slope_pres_grad_x, slope_pres_grad_y = a.gradient(x, y)
  slope_pres_elem_x = fvcom.grid.nodes2elems(slope_pres_grad_x, triangles)
  slope_pres_elem_y = fvcom.grid.nodes2elems(slope_pres_grad_y, triangles)

  return (slope_pres_elem_x, slope_pres_elem_y, 
          buoy_pres_elem_x, buoy_pres_elem_y)


def viscosity_friction(u, v, x, y, trio, fvcom_files, ah=1000):
  # Ah in range 100 to 10000 m^2/s
  # Ah is horizontal thermal diffusion coefficient
  if 0:
    a = tri.LinearTriInterpolator(trio, u)
    u_grad_x, u_grad_y = a.gradient(x, y)
    a = tri.LinearTriInterpolator(trio, u_grad_x)
    u_grad2_x, _ = a.gradient(x, y)
    a = tri.LinearTriInterpolator(trio, u_grad_y)
    _, u_grad2_y = a.gradient(x, y)
  else: 
    # Use Greens
    du = physics.green_gauss_gradient_method(u, fvcom_files)
    u_grad_x = du[:, 0]
    u_grad_y = du[:, 1]
    du2 = physics.green_gauss_gradient_method(u_grad_x, fvcom_files)
    u_grad2_x = du2[:, 0]
    du2 = physics.green_gauss_gradient_method(u_grad_y, fvcom_files)
    u_grad2_y = du2[:, 0]

  fric_x = ah * (u_grad2_x + u_grad2_y)

  if 0:
    a = tri.LinearTriInterpolator(trio, v)
    v_grad_x, v_grad_y = a.gradient(x, y)
    a = tri.LinearTriInterpolator(trio, v_grad_x)
    v_grad2_x, _ = a.gradient(x, y)
    a = tri.LinearTriInterpolator(trio, v_grad_y)
    _, v_grad2_y = a.gradient(x, y)
  else: 
    # Use Greens
    dv = physics.green_gauss_gradient_method(v, fvcom_files)
    v_grad_x = dv[:, 0]
    v_grad_y = dv[:, 1]
    dv2 = physics.green_gauss_gradient_method(v_grad_x, fvcom_files)
    v_grad2_x = dv2[:, 0]
    dv2 = physics.green_gauss_gradient_method(v_grad_y, fvcom_files)
    v_grad2_y = dv2[:, 0]

  fric_x = ah * (v_grad2_x + v_grad2_y)
  return fric_x, fric_y


mjas = '/dssgfs01/scratch/benbar/SSW_RS/'
fn_dy = sorted(glob.glob(mjas + 'SSW_RS_v1.2_2013_12_12/Turb/SSW_Hindcast_avg*.nc'))
fn_hr = sorted(glob.glob(mjas + 'SSW_RS_v1.2_2013_12_12/Turb/SSW_Hindcast_0*.nc'))
forcing_dir = '/dssgfs01/scratch/benbar/Forcing/'
fin = '/dssgfs01/working/benbar/SSW_RS/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'
#out_dir = '/scratch/benbar/Processed_Data/'


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

# Setup Greens's theory gradient

varlist = ['temp', 'salinity']
dims = {'siglay': [0]}
fvcom_files = MFileReader(fn_dy[0], variables=varlist, dims=dims)
fvcom_files.grid.lon[fvcom_files.grid.lon > 180] = (fvcom_files.grid.lon[
fvcom_files.grid.lon > 180] - 360)


# read different variables in the loop

vars = ('zeta', 'temp', 'salinity', 'ua', 'va', 'v', 'u', 'Times', 'h', 'tauc')
dims = {'time':':'}

FVCOM = readFVCOM(fn_hr[0], vars, dims=dims)
print(len(fn_hr))

xy = np.array(((-9, 48.8), (4, 61), (10, 57.9), (10, 58.5)))  
ele_ind = np.array([fvg.closest_element(i) for i in xy]).flatten().tolist()

ind = ele_ind[2]
isize = FVCOM['ua'].shape[1]
print(isize)


date_list = np.zeros((len(FVCOM['zeta'][:, 0]) -1), dtype=object)
du_dt = np.zeros((FVCOM['ua'].shape[0] -1, isize))
dv_dt = np.zeros((FVCOM['ua'].shape[0] -1, isize))
momentum_full_x = np.zeros((FVCOM['ua'].shape[0] -1, isize))
momentum_full_y = np.zeros((FVCOM['ua'].shape[0] -1, isize))

wind_stress_x = np.zeros((FVCOM['ua'].shape[0] -1, isize))
wind_stress_y = np.zeros((FVCOM['ua'].shape[0] -1, isize))
bot_stress_x = np.zeros((FVCOM['ua'].shape[0] -1, isize))
bot_stress_y = np.zeros((FVCOM['ua'].shape[0] -1, isize))
coriolis_x = np.zeros((FVCOM['ua'].shape[0] -1, isize))
coriolis_y = np.zeros((FVCOM['ua'].shape[0] -1, isize))
u_v_advection_x = np.zeros((FVCOM['ua'].shape[0] -1, isize))
u_v_advection_y = np.zeros((FVCOM['ua'].shape[0] -1, isize))
pres_grad_elem_x = np.zeros((FVCOM['ua'].shape[0] -1, isize))
pres_grad_elem_y = np.zeros((FVCOM['ua'].shape[0] -1, isize))
slope_grad_elem_x = np.zeros((FVCOM['ua'].shape[0] -1, isize))
slope_grad_elem_y = np.zeros((FVCOM['ua'].shape[0] -1, isize))
buoy_grad_elem_x = np.zeros((FVCOM['ua'].shape[0] -1, isize))
buoy_grad_elem_y = np.zeros((FVCOM['ua'].shape[0] -1, isize))


for t1 in range(2):#len(FVCOM['zeta'][:, 0]) - 1):

  date_list[t1] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-4].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S.%f')
  next_time = dt.datetime.strptime(''.join(FVCOM['Times'][t1 + 1, :-4].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S.%f')
  t_diff = next_time - date_list[t1]
  sec_diff = (t_diff.days * 24 * 60 * 60) + t_diff.seconds

  print(t1, date_list[t1])

  H = (h_mod + FVCOM['zeta'][t1, :])
  Hc = (hc_mod + fvcom.grid.nodes2elems(FVCOM['zeta'][t1, :], triangles))
  depth_mod = -H * siglay_mod # should be negative (siglay is negative)
  depthlev_mod = -H * siglev_mod # should be negative (siglev is negative)

  depth_mod1 = -h_mod * siglay_mod # should be negative (siglay is negative)
  depthlev_mod1 = -h_mod * siglev_mod # should be negative (siglev is negative)
  d_depth = (depthlev_mod1[1:, :] - depthlev_mod1[:-1, :])
  print(np.mean(depthlev_mod1[-1, :]), np.ma.mean(d_depth))

  # constants 

  f = sw.f(lat)
  fc = sw.f(latc)
  g = sw.grav(56, 0)
  rho_0 = 1025 * 1000 # convert to g/m^3

  # variables

  rho = calc_rho(FVCOM['salinity'][t1, :, :], FVCOM['temp'][t1, :, :], 
              -depth_mod, lon, lat) * 1000 # convert to g/m^3
  pres = sw.p_from_z(-depth_mod, lat) * 10000 # convert to Pa

  Cd_bot = 0.0015
  #tau_bot_u = -rho_0 * Cd_bot * np.abs(FVCOM['u'][t1, -1, :]) * FVCOM['u'][t1, -1, :]
  #tau_bot_v = -rho_0 * Cd_bot * np.abs(FVCOM['v'][t1, -1, :]) * FVCOM['v'][t1, -1, :]
  tau_bot_u = -rho_0 * Cd_bot * (((FVCOM['u'][t1, -1, :] ** 2) + (FVCOM['v'][t1, -1, :] ** 2)) ** 0.5) * FVCOM['u'][t1, -1, :]
  tau_bot_v = -rho_0 * Cd_bot * (((FVCOM['u'][t1, -1, :] ** 2) + (FVCOM['v'][t1, -1, :] ** 2)) ** 0.5) * FVCOM['v'][t1, -1, :]


  # Wind

  use_date = dt.datetime.strptime(''.join(
      FVCOM['Times'][t1, :-4].astype(str)).replace(
      'T', ' '), '%Y-%m-%d %H:%M:%S.%f')
  tau_wind_u, tau_wind_v, air_pres= calc_wind_tau(forcing_dir 
        + 'SSW_Hindcast_metforcing_ERA5_2014.nc', use_date, t1)

  # u and v changes

  u_v_advection_x[t1, :], u_v_advection_y[t1, :] = calc_uv(FVCOM['ua'][t1, :], 
      FVCOM['va'][t1, :], trio, triangles, fvcom_files)

  (slope_grad_elem_x[t1, :], slope_grad_elem_y[t1, :], 
      buoy_grad_elem_x[t1, :], buoy_grad_elem_y[t1, :]) = buoy_slope_pres_grad(rho, rho_0, g, d_depth, air_pres, pres, x, y, trio)

  pres_grad_elem_x[t1, :], pres_grad_elem_y[t1, :] = calc_pres_grad(
      FVCOM['zeta'][t1, :], x, y, g, trio, triangles, fvcom_files)
#  slope_grad_elem_x[t1, :], slope_grad_elem_y[t1, :] = calc_pres_grad(
 #     FVCOM['zeta'][t1, :], x, y, g, trio, triangles, fvcom_files)

  #pres_grad_elem_x[t1, :] = slope_grad_elem_x[t1, :] + buoy_grad_elem_x[t1, :]
  #pres_grad_elem_y[t1, :] = slope_grad_elem_y[t1, :] + buoy_grad_elem_y[t1, :]

  # Momentum Balance x terms

  coriolis_x[t1, :] = np.mean(fc * FVCOM['v'][t1, :, :], axis=0)
  bot_stress_x[t1, :] = -tau_bot_u / (rho_0 * Hc)
  wind_stress_x[t1, :] = tau_wind_u / (rho_0 * Hc)


  momentum_full_x[t1, :] = bot_stress_x[t1, :] + wind_stress_x[t1, :] + coriolis_x[t1, :] - pres_grad_elem_x[t1, :] - u_v_advection_x[t1, :]

  #du_dt1 = (FVCOM['ua'][t1 + 1, :] - FVCOM['ua'][t1, :]) / sec_diff
  du_dt[t1, :] = np.mean((FVCOM['u'][t1 + 1, :, :] - FVCOM['u'][t1, :, :]) / sec_diff, axis=0)
  print(momentum_full_x[t1, 0], du_dt[t1, 0])

  # Momentum Balance y terms

  coriolis_y[t1, :] = np.mean(-fc * FVCOM['u'][t1, :, :], axis=0)
  bot_stress_y[t1, :] = -tau_bot_v / (rho_0 * Hc)
  wind_stress_y[t1, :] = tau_wind_v / (rho_0 * Hc)


  momentum_full_y[t1, :] =  bot_stress_y[t1, :] + wind_stress_y[t1, :] + coriolis_y[t1, :] - pres_grad_elem_y[t1, :] - u_v_advection_y[t1, :]

  dv_dt[t1, :] = np.mean((FVCOM['v'][t1 + 1, :, :] - FVCOM['v'][t1, :, :]) / sec_diff, axis=0)


momentum_full_y = np.ma.masked_where(np.isnan(momentum_full_y), momentum_full_y)
momentum_full_x = np.ma.masked_where(np.isnan(momentum_full_x), momentum_full_x)

t2 = t1 + 1

date_list = date_list[:t2]
du_dt = du_dt[:t2, :]
dv_dt = dv_dt[:t2, :]
momentum_full_x = momentum_full_x[:t2, :]
momentum_full_y = momentum_full_y[:t2, :]

wind_stress_x = wind_stress_x[:t2, :]
wind_stress_y = wind_stress_y[:t2, :]
bot_stress_x = bot_stress_x[:t2, :]
bot_stress_y = bot_stress_y[:t2, :]
coriolis_x = coriolis_x[:t2, :]
coriolis_y = coriolis_y[:t2, :]
pres_grad_elem_x = pres_grad_elem_x[:t2, :]
pres_grad_elem_y = pres_grad_elem_y[:t2, :]
buoy_grad_elem_x = buoy_grad_elem_x[:t2, :]
buoy_grad_elem_y = buoy_grad_elem_y[:t2, :]
slope_grad_elem_x = slope_grad_elem_x[:t2, :]
slope_grad_elem_y = slope_grad_elem_y[:t2, :]
u_v_advection_x = u_v_advection_x[:t2, :]
u_v_advection_y = u_v_advection_y[:t2, :]


# Plot

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_axes([0.05, 0.08, 0.27, 0.8])
ax2 = fig1.add_axes([0.37, 0.08, 0.27, 0.8])
ax3 = fig1.add_axes([0.69, 0.08, 0.27, 0.8])
cax1 = fig1.add_axes([0.3, 0.9, 0.4, 0.01])

extents = np.array((-9, 2, 54, 61))

def add_map(ax1):
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
  return m1

m1 = add_map(ax1)
m1 = add_map(ax2)
m1 = add_map(ax3)

mx, my = m1(lon, lat)

#mag_buoy = ((buoy_grad_elem_x ** 2) 
#    + (buoy_grad_elem_y ** 2)) ** 0.5
#mag_slope = ((slope_grad_elem_x ** 2) 
#    + (slope_grad_elem_y ** 2)) ** 0.5
mag_pres = ((pres_grad_elem_x ** 2) 
    + (pres_grad_elem_y ** 2)) ** 0.5
mag_stress = (((wind_stress_x ** 2) 
    + (wind_stress_y ** 2)) ** 0.5 + ((bot_stress_x ** 2) 
    + (bot_stress_y ** 2)) ** 0.5)
mag_corr = ((coriolis_x ** 2) 
    + (coriolis_y ** 2)) ** 0.5

cs1 = ax1.tripcolor(mx, my, triangles, mag_pres[0, :], vmin=0, vmax=1e-4, zorder=99)

fig1.colorbar(cs1, cax=cax1, orientation='horizontal')
cax1.set_xlabel('Acceleration (ms$^{-2}$)')

ax2.tripcolor(mx, my, triangles, mag_stress[0, :], vmin=0, vmax=1e-4, zorder=99)

ax3.tripcolor(mx, my, triangles, mag_corr[0, :], vmin=0, vmax=1e-4, zorder=99)

#ax2.plot(mx[59419], my[59419], 'or', zorder=101)


ax1.annotate('Pressure Force', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('Wind Stress Force', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('Rotation Force', (0.03, 0.88), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)





fig2 = plt.figure(figsize=(10, 5))
ax1 = fig2.add_axes([0.1, 0.55, 0.8, 0.35])
ax2 = fig2.add_axes([0.1, 0.05, 0.8, 0.35])

xy = np.array(((-9, 48.8), (4, 61), (10, 57.9), (10, 58.5)))  
ele_ind = np.array([fvg.closest_element(i) for i in xy]).flatten().tolist()

print(ele_ind[2])

ax1.plot(date_list, du_dt[:, 1], label='Du/Dt')
ax1.plot(date_list, momentum_full_x[:, 1], label='Momentum')
ax1.plot(date_list, wind_stress_x[:, 1], '--', label='Wind Stress')
ax1.plot(date_list, bot_stress_x[:, 1], '--', label='Bottom Stress')
ax1.plot(date_list, coriolis_x[:, 1], '--', label='Coriolis')
ax1.plot(date_list, pres_grad_elem_x[:, 1] * -1, '--', label='Pressure Grad')

ax2.plot(date_list, dv_dt[:, 1], label='Du/Dt')
ax2.plot(date_list, momentum_full_y[:, 1], label='Momentum')
ax2.plot(date_list, wind_stress_y[:, 1], '--', label='Wind Stress')
ax2.plot(date_list, bot_stress_y[:, 1], '--', label='Bottom Stress')
ax2.plot(date_list, coriolis_y[:, 1], '--', label='Coriolis')
ax2.plot(date_list, pres_grad_elem_y[:, 1] * -1, '--', label='Pressure Grad')


ax1.legend()

ind = ele_ind[2] # 59419

fig3 = plt.figure(figsize=(10, 8))
axes = fig3.subplots(nrows=3, ncols=2)

axes[0, 0].plot(date_list, u_v_advection_x[:, ind])
axes[0, 0].plot(date_list, u_v_advection_y[:, ind])

axes[0, 1].plot(date_list, coriolis_x[:, ind])
axes[0, 1].plot(date_list, coriolis_y[:, ind])

axes[1, 0].plot(date_list, pres_grad_elem_x[:, ind])
axes[1, 0].plot(date_list, pres_grad_elem_y[:, ind])

#axes[1, 1].plot(date_list, slope_grad_elem_x[:, ind])
#axes[1, 1].plot(date_list, slope_grad_elem_y[:, ind])
axes[1, 1].plot(date_list, buoy_grad_elem_x[:, ind])
axes[1, 1].plot(date_list, buoy_grad_elem_y[:, ind])

#axes[1, 1].plot(date_list, diff_x)
#axes[1, 1].plot(date_list, diff_y)

axes[2, 0].plot(date_list, wind_stress_x[:, ind])
axes[2, 0].plot(date_list, wind_stress_y[:, ind])
axes[2, 0].plot(date_list, bot_stress_x[:, ind])
axes[2, 0].plot(date_list, bot_stress_y[:, ind])

axes[2, 1].plot(date_list, du_dt[:, ind])
axes[2, 1].plot(date_list, dv_dt[:, ind])

#axes[2, 1].plot(date_list, momentum_full_x[:, ind])
#axes[2, 1].plot(date_list, momentum_full_y[:, ind])
axes[2, 1].plot(date_list, bot_stress_x[:, ind] + wind_stress_x[:, ind] + coriolis_x[:, ind] - pres_grad_elem_x[:, ind])
axes[2, 1].plot(date_list, bot_stress_y[:, ind] + wind_stress_y[:, ind] + coriolis_y[:, ind] - pres_grad_elem_y[:, ind])

axes[0, 0].set_ylabel('Advection')
axes[0, 1].set_ylabel('Coriolis')
axes[1, 0].set_ylabel('BTrop PGF')
axes[1, 1].set_ylabel('Diffusion')
axes[2, 0].set_ylabel('Stress')
#axes[2, 1].set_ylabel('Divergence')
axes[2, 1].set_ylabel('dVel_dt')

axf = axes.flatten()
for ax in axf:
  ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))   #to get a tick every 15 minutes
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))     #optional formatting


fig1.savefig('./Figures/forces_balance.png')
#fig2.savefig('./Figures/momentum_time_greens.png')

fig3.savefig('./Figures/forces_greens.png')

