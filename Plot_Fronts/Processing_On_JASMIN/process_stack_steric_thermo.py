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

def steric_ab(sal, tp, depth, lat, prac=False):
  # Calculate the steric height based velocity for thermal and haline 
  # from Levitus et al. 2005

  pres = sw.p_from_z(-depth, lat)
  if prac:
    sa = sw.SA_from_SP(sp, pres, lon, lat)
  else:
    sa = sal
  ct = sw.CT_from_pt(sa, tp)
  g = sw.grav(56, 0)

  sub = np.ma.zeros(sa.shape)
  # 0 = steric height intergrated to surface
  srf_st = 0

  steric_thermo = (sw.geo_strf_dyn_height(sub + 35, ct, pres, 0) / g)[srf_st, :]
  steric_haline = (sw.geo_strf_dyn_height(sa, sub + 0, pres, 0) / g)[srf_st, :]

  return steric_thermo, steric_haline

def barotropic(trio, ssh, g, f, x, y):
    a = tri.LinearTriInterpolator(trio, ssh)
    dn_dx, dn_dy = a.gradient(x, y)
    barotrop_u = dn_dy * (-g / f)
    barotrop_v = dn_dx * (g / f)
    return barotrop_u, barotrop_v

mjas = '/gws/nopw/j04/ssw_rs/Model_Output_V3.02/Daily/'
fn = []
for yr in range(1993, 2020):
  fn.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V3.02*dy*RE.nc')))

print(len(fn))
out_dir = '../Processed_Data/'


fin = '../Input_Files/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'

# Extract only the first 24 time steps.
dims = {'time':':10'}
# List of the variables to extract.
vars = ('lon', 'lat', 'latc', 'lonc', 'nv', 'zeta', 'temp', 'salinity', 'ua', 'va', 'siglay', 'siglev', 'h', 'Itime', 'Itime2')
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
print(lon.shape[0])
years = np.arange(1993, 2020)
barosteric_u_a_mn = np.ma.zeros((len(years) * 12, lon.shape[0])) - 999
barosteric_v_a_mn = np.ma.zeros((len(years) * 12, lon.shape[0])) - 999

date_list = np.zeros((len(fn) * 10), dtype=object) - 999
date_mn = np.zeros((len(years) * 12), dtype=object) - 999

c = 0
mc = 0
dc = 0
loc = np.nonzero((lon > -4.1) & (lon < -4) & (lat > 58.7) & (lat < 58.8))[0][0]

for i in range(len(fn)):
  print(i / len(fn) *100, '%')
  FVCOM = readFVCOM(fn[i], vars, dims=dims)

  for t1 in range(FVCOM['zeta'].shape[0]):
    date_list[c] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-7].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S')

    if c == 0:
      mn1 = date_list[c].month
      barosteric_u_a = np.ma.zeros((32, lon.shape[0])) - 999
      barosteric_v_a = np.ma.zeros((32, lon.shape[0])) - 999


    if mn1 != date_list[c].month:
      # new month so calculate mean and reset
      date_mn[mc] = dt.datetime(date_list[c - 1].year, date_list[c - 1].month, 1)
      barosteric_u_a_mn[mc, :] = np.ma.mean(barosteric_u_a[:dc, :], axis=0)
      barosteric_v_a_mn[mc, :] = np.ma.mean(barosteric_v_a[:dc, :], axis=0)

      barosteric_u_a = np.ma.zeros((32, lon.shape[0])) - 999
      barosteric_v_a = np.ma.zeros((32, lon.shape[0])) - 999

      mc = mc + 1
      dc = 0

    mn1 = date_list[c].month

    H = h_mod + FVCOM['zeta'][t1, :]
    depth_mod = -H * siglay_mod # should be negative
    depthlev_mod = -H * siglev_mod # should be negative
    d_depth = depthlev_mod[:-1, :] - depthlev_mod[1:, :]

    steric_thermo, _ = steric_ab(
                FVCOM['salinity'][t1, :, :], 
                FVCOM['temp'][t1, :, :], 
                depth_mod, lat)

    barosteric_u_a[dc, :], barosteric_v_a[dc, :] = barotropic(trio, steric_thermo, g, f, x, y)
    dc = dc + 1
    c = c + 1

date_list = date_list[0:c]
date_mn[mc] = dt.datetime(date_list[c - 1].year, date_list[c - 1].month, 1)
barosteric_u_a_mn[mc, :] = np.ma.mean(barosteric_u_a[:dc - 1, :], axis=0)
barosteric_v_a_mn[mc, :] = np.ma.mean(barosteric_v_a[:dc - 1, :], axis=0)

print(date_mn[-2:])


np.savez(out_dir + 'stack_barosteric_thermo_vel_mn.npz', barosteric_u_a=barosteric_u_a_mn, barosteric_v_a=barosteric_v_a_mn, date_list=date_mn, lon=lon, lat=lat, tri=triangles)


