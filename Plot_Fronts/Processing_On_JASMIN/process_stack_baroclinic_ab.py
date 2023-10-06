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


def calc_rho_s(sal, tp, depth, lon, lat, prac=False):
  pres = sw.p_from_z(-depth, lat)
  if prac:
    sa = sw.SA_from_SP(sp, pres, lon, lat)
  else:
    sa = sal
  ct = sw.CT_from_pt(sa, tp)
  rho_s = sw.rho(sa, ct, 0)
  return rho_s

def calc_rho_alpha_beta(sal, tp, depth, lon, lat, rho_0, prac=False):
  pres = sw.p_from_z(-depth, lat)
  if prac:
    sa = sw.SA_from_SP(sp, pres, lon, lat)
  else:
    sa = sal
  ct = sw.CT_from_pt(sa, tp)
  rho, alpha, beta = sw.rho_alpha_beta(sa, ct, pres)

  # Equation for alpha and beta contribution to density from Roquet et al 2015
  # rho = -rho_alpha + rho_beta
  rho_alpha = rho_0 * alpha * tp
  rho_beta = rho_0 * beta * sa

  return rho, rho_alpha, rho_beta

def baroclinic(rho, rho_0, g, f, x, y, d_depth, H):
  # Calculate baroclinic velocity

  thermal_wind_u = np.zeros(rho.shape)
  thermal_wind_v = np.zeros(rho.shape)

  for d in range(rho.shape[0]-1, -1, -1): # loop over depth upwards
    
    a = tri.LinearTriInterpolator(trio, rho[d, :])
    drho_dx, drho_dy = a.gradient(x, y)

    tw_u = drho_dy * (g / (rho_0 * f))
    tw_v = -drho_dx * (g / (rho_0 * f))

    # integrate upwards

    if d == rho.shape[0]-1:
      thermal_wind_u[d, :] = 0 + tw_u
      thermal_wind_v[d, :] = 0 + tw_v
    else:
      thermal_wind_u[d, :] = thermal_wind_u[d +1, :] + tw_u # m/s
      thermal_wind_v[d, :] = thermal_wind_v[d +1, :] + tw_v # m/s

  bc_u = np.sum(thermal_wind_u * -d_depth, axis=0) / H
  bc_v = np.sum(thermal_wind_v * -d_depth, axis=0) / H
  return bc_u, bc_v


mjas = '/gws/nopw/j04/ssw_rs/Model_Output_V3.02/Daily/'
fn = []
for yr in range(1993, 2020):
  fn.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V3.02*dy*RE.nc')))
#fn = sorted(glob.glob(mjas + '*/SSWRS*V1.1*dy*RE.nc'))

print(len(fn))
out_dir = '../Processed_Data/'


fin = '../Input_Files/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'

# Extract only the first 24 time steps.
dims = {'time':':10'}
# List of the variables to extract.
vars = ('lon', 'lat', 'latc', 'lonc', 'nv', 'zeta', 'siglay', 'siglev', 'h', 'Itime', 'Itime2')
FVCOM = readFVCOM(fn[-1], vars, dims=dims)

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

if 0:
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

print('Loaded grid')

trio = tri.Triangulation(x, y, triangles=np.asarray(triangles))

# constants

f = sw.f(lat)
fc = sw.f(latc)
g = sw.grav(56, 0)
rho_0 = 1025

vars = ('zeta', 'temp', 'salinity', 'Times', 'h')

var = 3 # pick which variable to calculate rho, alpha or beta

baroclin_u  = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
baroclin_v = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
#baroclin_u_a  = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
#baroclin_v_a = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
#baroclin_u_b  = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
#baroclin_v_b = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999

#barotrop_u  = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
#barotrop_v = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999

date_list = np.zeros((len(fn) * 10), dtype=object) - 999
c = 0
loc = np.nonzero((lon > -4.1) & (lon < -4) & (lat > 58.7) & (lat < 58.8))[0][0]

for i in range(len(fn)):
  print(i / len(fn) *100, '%')
  FVCOM = readFVCOM(fn[i], vars, dims=dims)

  for t1 in range(FVCOM['zeta'].shape[0]):

    H = h_mod + FVCOM['zeta'][t1, :]
    depth_mod = -H * siglay_mod # should be negative
    depthlev_mod = -H * siglev_mod # should be negative
    d_depth = depthlev_mod[:-1, :] - depthlev_mod[1:, :]

    #a = tri.LinearTriInterpolator(trio, FVCOM['zeta'][t1, :])
    #dn_dx, dn_dy = a.gradient(x, y)
    #barotrop_u[c, :] = dn_dy * (-g / f)
    #barotrop_v[c, :] = dn_dx * (g / f)

    rho, rho_alpha, rho_beta = calc_rho_alpha_beta(FVCOM['salinity'][t1, :, :], 
                FVCOM['temp'][t1, :, :], 
                depth_mod, lon, lat, rho_0, prac=False)

    if var == 1:
      b_parts = baroclinic(rho, rho_0, g, f, x, y, d_depth, H)
    elif var == 2:
      b_parts = baroclinic(rho_alpha, rho_0, g, f, x, y, d_depth, H)
    elif var == 3:
      b_parts = baroclinic(rho_beta, rho_0, g, f, x, y, d_depth, H)

    baroclin_u[c, :] = b_parts[0]
    baroclin_v[c, :] = b_parts[1]

    date_list[c] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-7].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S')
    c = c + 1

date_list = date_list[0:c]
baroclin_u = baroclin_u[0:c, :]
baroclin_v = baroclin_v[0:c, :]

#barotrop_u = barotrop_u[0:c, :]
#barotrop_v = barotrop_v[0:c, :]

# monthly means

yr = np.array([d.year for d in date_list])
mnt = np.array([d.month for d in date_list])
yri = np.unique(yr)
mnti =np.unique(mnt)

u_mn = np.ma.zeros((len(yri) * len(mnti), lon.shape[0]))
v_mn = np.ma.zeros((len(yri) * len(mnti), lon.shape[0]))

ut_mn = np.ma.zeros((len(yri) * len(mnti), lon.shape[0]))
vt_mn = np.ma.zeros((len(yri) * len(mnti), lon.shape[0]))

date_mn = np.zeros((len(yri) * len(mnti)), dtype=object)
count = 0
for i in range(len(yri)):
  for j in range(len(mnti)):
    ind = (yr == yri[i]) & (mnt == mnti[j])
    u_mn[count] = np.ma.mean(baroclin_u[ind, :], axis=0)
    v_mn[count] = np.ma.mean(baroclin_v[ind, :], axis=0)

    #ut_mn[count] = np.ma.mean(barotrop_u[ind, :], axis=0)
    #vt_mn[count] = np.ma.mean(barotrop_v[ind, :], axis=0)
    date_mn[count] = dt.datetime(yri[i], mnti[j], 1)
    count += 1

if var == 1:
  np.savez(out_dir + 'stack_baroclinic_r_vel_mn.npz', baroclin_u=u_mn, baroclin_v=v_mn, date_list=date_mn, lon=lon, lat=lat, tri=triangles)

elif var == 2:
  np.savez(out_dir + 'stack_baroclinic_a_vel_mn.npz', baroclin_u_a=u_mn, baroclin_v_a=v_mn, date_list=date_mn, lon=lon, lat=lat, tri=triangles)

elif var == 3:
  np.savez(out_dir + 'stack_baroclinic_b_vel_mn.npz', baroclin_u_b=u_mn, baroclin_v_b=v_mn, date_list=date_mn, lon=lon, lat=lat, tri=triangles)


