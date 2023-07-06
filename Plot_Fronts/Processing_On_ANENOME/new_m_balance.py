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



def calc_rho(sp, tp, depth, lon, lat):
  # density 

  pres = sw.p_from_z(depth, lat)
  sa = sw.SA_from_SP(sp, pres, lon, lat)
  ct = sw.CT_from_pt(sa, tp)
  rho = sw.rho(sa, ct, 0)
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
  tau_wind_u = air_rho_elem * Cd * np.abs(u_wind) * u_wind
  tau_wind_v = air_rho_elem * Cd * np.abs(v_wind) * v_wind

  return tau_wind_u, tau_wind_v

def calc_uv(ua, va, trio, triangles):
  # partial momentum terms

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

  u_v_momentum_x = ((ua * du_dx) + (va * du_dy))
  u_v_momentum_y = ((ua * dv_dx) + (va * dv_dy))
  return u_v_momentum_x, u_v_momentum_y

def calc_pres_grad(zeta, x, y, g, trio, triangles):
  # pressure gradient force

  a = tri.LinearTriInterpolator(trio, FVCOM['zeta'][t1, :])
  z_grad_x, z_grad_y = a.gradient(x, y)
  horiz_dens_grad_x = z_grad_x * g
  horiz_dens_grad_y = z_grad_y * g

  pres_grad_elem_x = fvcom.grid.nodes2elems(horiz_dens_grad_x, triangles)
  pres_grad_elem_y = fvcom.grid.nodes2elems(horiz_dens_grad_y, triangles)

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

#def viscosity():


mjas = '/scratch/benbar/JASMIN/'
fn_dy = sorted(glob.glob(mjas + 'Model_Output_V3.02/Daily/1993/SSWRS_V3.02*dy*RE.nc'))
forcing_dir = '/scratch/benbar/Forcing/'
fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'
out_dir = '/scratch/benbar/Processed_Data/'


dims = {'time':':'}
# List of the variables to extract.
vars = ('lon', 'lat', 'latc', 'lonc', 'nv', 'zeta', 'temp', 'salinity', 'ua', 'va', 'siglay', 'siglev', 'h', 'h_center', 'Itime', 'Itime2', 'nbve', 'nbsn', 'nbe', 'ntsn', 'ntve', 'art1', 'art2')
FVCOM = readFVCOM(fn_dy[0], vars, dims=dims)

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

# read different variables in the loop

vars = ('zeta', 'temp', 'salinity', 'ua', 'va', 'v', 'u', 'Times', 'h', 'tauc')
dims = {'time':':'}

FVCOM = readFVCOM(fn_dy[0], vars, dims=dims)

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
pres_grad_elem_x = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
pres_grad_elem_y = np.zeros((FVCOM['ua'].shape[0] -1, FVCOM['ua'].shape[1]))
slope_grad_elem_x = np.zeros((FVCOM['ua'].shape[0] -1, isize))
slope_grad_elem_y = np.zeros((FVCOM['ua'].shape[0] -1, isize))
buoy_grad_elem_x = np.zeros((FVCOM['ua'].shape[0] -1, isize))
buoy_grad_elem_y = np.zeros((FVCOM['ua'].shape[0] -1, isize))


for t1 in range(4):#len(FVCOM['zeta'][:, 0]) - 1):

  date_list[t1] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-4].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S.%f')
  next_time = dt.datetime.strptime(''.join(FVCOM['Times'][t1 + 1, :-4].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S.%f')
  t_diff = next_time - date_list[t1]
  sec_diff = (t_diff.days * 24 * 60 * 60) + t_diff.seconds

  print(t1, date_list[t1])

  H = h_mod + FVCOM['zeta'][t1, :]
  Hc = hc_mod + fvcom.grid.nodes2elems(FVCOM['zeta'][t1, :], triangles)
  depth_mod = -H * siglay_mod # should be negative
#  depthlev_mod = -H * siglev_mod # should be negative

  depth_mod1 = -h_mod * siglay_mod # should be negative (siglay is negative)
  depthlev_mod1 = -h_mod * siglev_mod # should be negative (siglev is negative)
  d_depth = (depthlev_mod1[1:, :] - depthlev_mod1[:-1, :])

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
  tau_bot_u = -rho_0 * Cd_bot * np.abs(FVCOM['u'][t1, -1, :]) * FVCOM['u'][t1, -1, :]
  tau_bot_v = -rho_0 * Cd_bot * np.abs(FVCOM['v'][t1, -1, :]) * FVCOM['v'][t1, -1, :]

  # Wind

  use_date = dt.datetime.strptime(''.join(
      FVCOM['Times'][t1, :-4].astype(str)).replace(
      'T', ' '), '%Y-%m-%d %H:%M:%S.%f')
  tau_wind_u, tau_wind_v = calc_wind_tau(forcing_dir 
        + 'SSW_Hindcast_metforcing_ERA5_1993.nc', use_date, t1)

  # u and v changes

  u_v_momentum_x, u_v_momentum_y = calc_uv(FVCOM['ua'][t1, :], 
      FVCOM['va'][t1, :], trio, triangles)

  (slope_grad_elem_x[t1, :], slope_grad_elem_y[t1, :], 
      buoy_grad_elem_x[t1, :], buoy_grad_elem_y[t1, :]) = buoy_slope_pres_grad(rho, rho_0, g, d_depth, air_pres, pres, x, y, trio)

  pres_grad_elem_x[t1, :], pres_grad_elem_y[t1, :] = calc_pres_grad(
      FVCOM['zeta'][t1, :], x, y, g, trio, triangles)

  # Momentum Balance x terms

  coriolis_x[t1, :] = fc * FVCOM['va'][t1, :]
  bot_stress_x[t1, :] = -tau_bot_u / (rho_0 * Hc)
  wind_stress_x[t1, :] = tau_wind_u / (rho_0 * Hc)


  momentum_full_x[t1, :] = bot_stress_x[t1, :] + wind_stress_x[t1, :] + coriolis_x[t1, :] - pres_grad_elem_x[t1, :] - u_v_momentum_x

  du_dt[t1, :] = (FVCOM['ua'][t1 + 1, :] - FVCOM['ua'][t1, :]) / sec_diff

  # Momentum Balance y terms

  coriolis_y[t1, :] = -fc * FVCOM['ua'][t1, :]
  bot_stress_y[t1, :] = -tau_bot_v / (rho_0 * Hc)
  wind_stress_y[t1, :] = tau_wind_v / (rho_0 * Hc)


  momentum_full_y[t1, :] =  bot_stress_y[t1, :] + wind_stress_y[t1, :] + coriolis_y[t1, :] - pres_grad_elem_y[t1, :] - u_v_momentum_y

  dv_dt[t1, :] = (FVCOM['va'][t1 + 1, :] - FVCOM['va'][t1, :]) / sec_diff


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



# Plot

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_axes([0.05, 0.1, 0.30, 0.8])
ax2 = fig1.add_axes([0.525, 0.1, 0.30, 0.8])
cax1 = fig1.add_axes([0.36, 0.3, 0.01, 0.4])
cax2 = fig1.add_axes([0.835, 0.3, 0.01, 0.4])

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


cs1 = ax1.tripcolor(mx, my, triangles, momentum_full_x[t1-1], vmin=-3e-6, vmax=3e-6, zorder=99)
ax1.set_title('Momentum Balance')

fig1.colorbar(cs1, cax=cax1)
cax1.set_ylabel('Acceleration (ms$^{-2}$)')

cs2 = ax2.tripcolor(mx, my, triangles, du_dt[t1-1], vmin=-3e-6, vmax=3e-6, zorder=99)
ax2.set_title('Du/Dt')

fig1.colorbar(cs2, cax=cax2)
cax2.set_ylabel('Acceleration (ms$^{-2}$)')




fig2 = plt.figure(figsize=(10, 5))
ax1 = fig2.add_axes([0.1, 0.55, 0.8, 0.35])
ax2 = fig2.add_axes([0.1, 0.05, 0.8, 0.35])

xy = np.array(((-9, 48.8), (4, 61), (10, 57.9), (10, 58.5)))  
ele_ind = np.array([fvg.closest_element(i) for i in xy]).flatten().tolist()

ax1.plot(date_list, du_dt[:, ele_ind[2]], label='Du/Dt')
ax1.plot(date_list, momentum_full_x[:, ele_ind[2]], label='Momentum')
ax1.plot(date_list, wind_stress_x[:, ele_ind[2]], '--', label='Wind Stress')
ax1.plot(date_list, bot_stress_x[:, ele_ind[2]], '--', label='Bottom Stress')
ax1.plot(date_list, coriolis_x[:, ele_ind[2]], '--', label='Coriolis')
ax1.plot(date_list, pres_grad_elem_x[:, ele_ind[2]] * -1, '--', label='Pressure Grad')

ax2.plot(date_list, dv_dt[:, ele_ind[2]], label='Du/Dt')
ax2.plot(date_list, momentum_full_y[:, ele_ind[2]], label='Momentum')
ax2.plot(date_list, wind_stress_y[:, ele_ind[2]], '--', label='Wind Stress')
ax2.plot(date_list, bot_stress_y[:, ele_ind[2]], '--', label='Bottom Stress')
ax2.plot(date_list, coriolis_y[:, ele_ind[2]], '--', label='Coriolis')
ax2.plot(date_list, pres_grad_elem_y[:, ele_ind[2]] * -1, '--', label='Pressure Grad')


ax1.legend()

fig1.savefig('./Figures/momentum_balance.png')
fig2.savefig('./Figures/momentum_time.png')
