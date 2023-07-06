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
import networkx
import sys
import pyproj
from scipy import stats


mjas = '/gws/nopw/j04/ssw_rs/Model_Output_V3.02/Daily/'
fn = []
for yr in range(1993, 2020):
  fn.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V3.02*dy*RE.nc')))
#fn = sorted(glob.glob(mjas + '*/SSWRS*V1.1*dy*RE.nc'))

print(len(fn))
out_dir = '../Processed_Data/'

load_grid = 0
if load_grid:
  fin = '../Input_Files/'
  fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'

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
  lonc = fvg.grid.lonc
  latc = fvg.grid.latc
  hc = fvg.grid.h_center
  triangles = fvg.grid.triangles

else:
  data = np.load(out_dir + 'grid_info.npz')
  x = data['x']
  y = data['y']
  xc = data['xc']
  yc = data['yc']
  lon = data['lon']
  lat = data['lat']
  lonc = data['lonc']
  latc = data['latc']
  hc = data['hc']
  triangles = data['triangles']
  data.close()


dims = {'time':':'}
vars = ('siglay_center', 'siglev_center', 'siglay', 'siglev', 'nbve', 'nbsn', 'nbe', 'ntsn', 'ntve', 'art1', 'art2', 'a1u', 'a2u', 'aw0', 'awx', 'awy', 'h')

FVCOM = readFVCOM(fn[-1], vars, dims=dims)
siglay_mod_c = FVCOM['siglay_center'][:]
siglev_mod_c = FVCOM['siglev_center'][:]
siglay_mod = FVCOM['siglay'][:]
siglev_mod = FVCOM['siglev'][:]
h = FVCOM['h'][:]

siglev_mod_c.mask = np.zeros(siglev_mod_c.shape, dtype=bool)
siglev_mod_c[siglev_mod_c < -1] = -1
siglev_mod.mask = np.zeros(siglev_mod.shape, dtype=bool)
siglev_mod[siglev_mod < -1] = -1


# nv is nodes around elem
nbve = FVCOM['nbve'].transpose() -1 # elems around node
nbsn = FVCOM['nbsn'].transpose() -1 # nodes around node
nbe = FVCOM['nbe'].transpose() -1 # elems around elem
ntsn = FVCOM['ntsn'] # number nodes around node
ntve = FVCOM['ntve'] # number of elems around node

art1 = FVCOM['art1'] # area of node-base control volume (node)
art2 = FVCOM['art2'] # area of elements around node (node)
print(len(nbe), len(lonc))

# Check how FVCOM calculates distance using art1
# cell_area.F shows FVCOM used a function ARC to calcualate area in spherical
# probably using haversine
# page 21 and 47 in manual

def arc_dist(lon1, lat1, lon2, lat2):
  # distance calculation used by FVCOM (manual page 47)
  R = 6371000 # Earth's mean radius in m  
  xl1 = R * np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lon1)) 
  yl1 = R * np.cos(np.deg2rad(lat1)) * np.sin(np.deg2rad(lon1))
  zl1 = R * np.sin(np.deg2rad(lat1))
  xl2 = R * np.cos(np.deg2rad(lat2)) * np.cos(np.deg2rad(lon2)) 
  yl2 = R * np.cos(np.deg2rad(lat2)) * np.sin(np.deg2rad(lon2))
  zl2 = R * np.sin(np.deg2rad(lat2))
  dist = ((xl2 - xl1) ** 2 + (yl2 - yl1) ** 2 + (zl2 - zl1) ** 2) ** 0.5
  return dist


# Get EOF mask

x_min = -10.05
x_max = 2.05

y_min = 53.98
y_max = 61.02

x_mid1 = -2.87
x_mid2 = -6.35
x_mid3 = -0.25
y_mid1 = 54.25
x_cor = -5.9
y_cor = 59.9


sel_ind = np.invert((lon >= x_min) & (lon <= x_max) 
                  & (lat >= y_min) & (lat <= y_max))

art1 = np.tile(art1, (siglay_mod.shape[0], 1))

vars = ('zeta', 'salinity', 'u', 'v', 'Times', 'h')
dims = {'time':':'}

fw_volume = np.ma.zeros((len(fn) * 10)) - 999
delta_fw_vol = np.ma.zeros((len(fn) * 10)) - 999
date_list = np.zeros((len(fn) * 10), dtype=object) - 999
c = 0
s_ref = 36

now = dt.datetime.now()

for i in range(len(fn)):
  print(i / len(fn) *100, '%')
  FVCOM = readFVCOM(fn[i], vars, dims=dims)
  for t1 in range(FVCOM['u'].shape[0]):

    #print(dt.datetime.now() - now)
    #now = dt.datetime.now()
    zeta = FVCOM['zeta'][t1, :]
    sal = FVCOM['salinity'][t1, :, :] # time, depth, node

    # calculate depth change

    H = h + zeta
    depth_mod = -H * siglay_mod # should be negative (siglay is negative)
    depthlev_mod = -H * siglev_mod 
    d_depth = (depthlev_mod[1:, :] - depthlev_mod[:-1, :])

    # Multipy each by their depth and surface area

    # units m3
    fw_volume[c] = np.ma.sum(((s_ref - sal) / s_ref) * d_depth * art1)

    # units m3/s
    if c != 0:
      delta_fw_vol[c] = (fw_volume[c] - fw_volume[c - 1]) / (24 * 60 * 60)
    else:
      delta_fw_vol[c] = 0

    date_list[c] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-7].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S')
    c = c + 1

  print(date_list[c-1])
  print('fw_m', np.ma.mean(delta_fw_vol[:c-1]))


date_list = date_list[0:c]
fw_volume = fw_volume[0:c]
delta_fw_vol = delta_fw_vol[0:c]



np.savez(out_dir + 'fw_volume.npz', date_list=date_list, fw_volume=fw_volume, delta_fw_vol=delta_fw_vol)


