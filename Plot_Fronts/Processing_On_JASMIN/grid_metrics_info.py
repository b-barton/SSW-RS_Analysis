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
vars = ('siglay_center', 'siglev_center','nbve', 'nbsn', 'nbe', 'ntsn', 'ntve', 'art1', 'art2', 'a1u', 'a2u', 'aw0', 'awx', 'awy')

FVCOM = readFVCOM(fn[-1], vars, dims=dims)
siglay_mod = FVCOM['siglay_center'][:]
siglev_mod = FVCOM['siglev_center'][:]

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

a1u = FVCOM['a1u'] # momentum (triangle) control volume (four, nele)
a2u = FVCOM['a2u'] # momentum (triangle) control volume? (four, nele)
aw0 = FVCOM['aw0'] # ratio of -(x2*y3-x3*y2)/2 to area (three, nele)
awx = FVCOM['awx'] # ratio of radius to area in y-direction! (three, nele)
                    # awx[i, 0] = -(y2-y3)/2/art[i]
awy = FVCOM['awy'] # ratio of radius to area in x-direction! (three, nele)
                    # awy[i, 0] = -(x3-x2)/2/art[i]

print(a1u[:, :5])
print(a2u[:, :5])

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

art = 1# area of trinagle (elem)
ai1 = awx[:, 0] * art * -2 # y2 - y3
ai2 = awx[:, 1] * art * -2 # y3 - y1
ai3 = awx[:, 2] * art * -2 # y1 - y2
bi1 = awy[:, 0] * art * -2 # x3 - x2
bi2 = awy[:, 1] * art * -2 # x1 - x3
bi3 = awy[:, 2] * art * -2 # x2 - x1

art = np.zeros((len(xc)))
my_art2 = np.zeros((len(x)))
for i in range(len(x)):
  x_p = np.append(lon[nbsn[i]], lon[nbsn[i]][0])
  y_p = np.append(lat[nbsn[i]], lat[nbsn[i]][0])
  x_cen = lon[i]
  y_cen = lat[i]
  t_area = np.zeros((len(x_p) -1))
  for j in range(len(x_p) -1):
    # get area of triangle and add all
    #dxa = x_p[j] - x_p[j + 1]
    #dxb = x_p[j] - x_cen
    #dxc = x_p[j + 1] - x_cen
    #dya = y_p[j] - y_p[j + 1]
    #dyb = y_p[j] - y_cen
    #dyc = y_p[j + 1] - y_cen
    #da = (dxa ** 2 + dya ** 2) ** 0.5
    #db = (dxb ** 2 + dyb ** 2) ** 0.5
    #dc = (dxc ** 2 + dyc ** 2) ** 0.5
    #da = arc_dist(x_p[j], y_p[j], x_p[j + 1], y_p[j + 1])
    #db = arc_dist(x_p[j], y_p[j], x_cen, y_cen)
    #dc = arc_dist(x_p[j + 1], y_p[j + 1], x_cen, y_cen)
    da = fvcom.grid.haversine_distance((x_p[j], y_p[j]), (x_p[j + 1], y_p[j + 1])) * 1000
    db = fvcom.grid.haversine_distance((x_p[j], y_p[j]), (x_cen, y_cen)) * 1000
    dc = fvcom.grid.haversine_distance((x_p[j + 1], y_p[j + 1]), (x_cen, y_cen)) * 1000
    p = (da + db + dc) / 2
    t_area[j] = (p * da * db * dc) ** 0.5
  my_art2[i] = np.sum(t_area)


#awx = fvcom.grid.elems2nodes(ai1, triangles)
cs1 = plt.tripcolor(lon, lat, triangles, art2 - my_art2)
plt.colorbar(cs1)
plt.savefig('./Figures/grid_metrics.png')


