#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime as dt
import netCDF4 as nc
import glob
import matplotlib.tri as tri
from PyFVCOM.read import ncread as readFVCOM
from PyFVCOM import physics
from PyFVCOM.read import MFileReader
import pyproj
import PyFVCOM as fvcom
from shapely.geometry import LineString
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import cascaded_union
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
import gsw as sw



def calc_rho(sp, tp, depth, lon, lat):
  # density 
  pres = sw.p_from_z(depth, lat)
  sa = sw.SA_from_SP(sp, pres, lon, lat)
  ct = sw.CT_from_pt(sa, tp)
  rho = sw.rho(sa, ct, 0)
  print('This should be False:', np.ma.is_masked(pres))
  return rho

in_dir = '../Processed_Data/'
#mjas = '/gws/nopw/j04/ssw_rs/Model_Output_V3.02/Daily/'
mjas = '/gws/nopw/j04/ssw_rs/Model_Output_V3.02/Hourly/'
#fn = sorted(glob.glob(mjas + '*/SSWRS*V1.1*dy*' + 'RE.nc'))
fn = []
#for yr in range(1993, 2020):
for yr in range(1999, 2000):
  fn.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V3.02*hr*RE.nc')))



fin = '../Input_Files/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'


data = np.load(in_dir + 'coast_distance.npz', allow_pickle=True)
dist = data['dist_c']
lon = data['lon']
lat = data['lat']
data.close()

dims = {'time':':'}
# List of the variables to extract.
vars = ('nv', 'h', 'latc', 'lonc', 'siglay', 'siglev', 'h_center')
FVCOM = readFVCOM(fn[-1], vars, dims=dims)
nv = FVCOM['nv']
latc = FVCOM['latc']
lonc = FVCOM['lonc']
siglay_mod = FVCOM['siglay'][:]
siglev_mod = FVCOM['siglev'][:]
h_mod = FVCOM['h'][:] # h_mod is positive
hc_mod = FVCOM['h_center'][:]

I = np.where(lonc > 180)
lonc[I] = lonc[I] - 360 
triangles = nv.transpose() -1
siglev_mod.mask = np.zeros(siglev_mod.shape, dtype=bool)
siglev_mod[siglev_mod < -1] = -1

st_date = dt.datetime(1992, 12, 30, 0, 0, 0)
en_date = dt.datetime(1993, 1, 31, 23, 0, 0)

fvg = fvcom.preproc.Model(st_date, en_date, grid=fgrd, 
                        native_coordinates='spherical', zone='30N')
#ll_coast = np.zeros((fvg.grid.lon[fvg.grid.coastline].shape[0], 2))
#ll_coast[:, 0] = np.squeeze(fvg.grid.lon[fvg.grid.coastline])
#ll_coast[:, 1] = np.squeeze(fvg.grid.lat[fvg.grid.coastline])

x = fvg.grid.x
y = fvg.grid.y
xc = fvg.grid.xc
yc = fvg.grid.yc
triangles = fvg.grid.triangles
trio = tri.Triangulation(x, y, triangles=np.asarray(triangles))

dist_e = fvcom.grid.nodes2elems(dist, triangles)

varlist = ['temp', 'salinity']
dims = {'siglay': [0]}
fvcom_files = MFileReader(fn[0], variables=varlist, dims=dims)
fvcom_files.grid.lon[fvcom_files.grid.lon > 180] = (fvcom_files.grid.lon[
    fvcom_files.grid.lon > 180] - 360)


# constants 

f = sw.f(np.mean(lat))
g = sw.grav(np.mean(lat), 0)
rho_0 = 1025 * 1000 # convert to g/m^3

h_0 = 0 # coast wall depth at y = 0
r = 0.0005 # m/s bottom friction coefficient, taken from paper
ku = 1e-5 # m2/s vertical mixing coefficient, from nml
sig = ((2 * ku) / f) ** 0.5 # vertical scale of bottom Ekman layer


for i in range(10):#len(fn)):
  print(i / len(fn) *100, '%')
  FVCOM = readFVCOM(fn[i], vars, dims=dims)
  s = # bottom slope
  u_max = # along shelf jet velocity
  phi = # dynamic pressure (pressure / rho_0)

  for t1 in range(FVCOM['u'].shape[0]):
    y_f = ((f * rho_0) / (s * g)) * (u_max / (drho_dy)) * (1 + ((sig * dphi_dx) / (r * u_max))) - (h_0 / s)
