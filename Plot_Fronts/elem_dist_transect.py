#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime as dt
import netCDF4 as nc
import glob
import matplotlib.tri as tri
from PyFVCOM.read import ncread as readFVCOM
import pyproj
import PyFVCOM as fvcom
from shapely.geometry import LineString
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import cascaded_union
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection


def extract_poly_coords(geom):
    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        interior_coords = []
        for i in geom.interiors:
            interior_coords += i.coords[:]
    elif geom.type == 'MultiPolygon':
        exterior_coords = []
        interior_coords = []
        for part in geom:
            epc = extract_poly_coords(part)  # Recursive call
            exterior_coords += epc['exterior_coords']
            interior_coords += epc['interior_coords']
    else:
        raise ValueError('Unhandled geometry type: ' + repr(geom.type))
    return {'exterior_coords': exterior_coords,
            'interior_coords': interior_coords}



in_dir = '/scratch/benbar/Processed_Data/'
mjas = '/scratch/benbar/JASMIN/Model_Output/Daily/'
#fn = sorted(glob.glob(mjas + '*/SSWRS*V1.1*dy*' + 'RE.nc'))
fn = []
for yr in range(1993, 2019):
  fn.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V1.1*dy*RE.nc')))


out_dir = '/scratch/benbar/Processed_Data_V3.02/'

fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'


data = np.load(in_dir + 'coast_distance.npz', allow_pickle=True)
dist = data['dist_c']
lon = data['lon']
lat = data['lat']
data.close()

dims = {'time':':10'}
# List of the variables to extract.
vars = ('nv', 'h')
FVCOM = readFVCOM(fn[-1], vars, dims=dims)
nv = FVCOM['nv']
h = FVCOM['h']
triangles = nv.transpose() -1

st_date = dt.datetime(1992, 12, 30, 0, 0, 0)
en_date = dt.datetime(1993, 1, 31, 23, 0, 0)

fvg = fvcom.preproc.Model(st_date, en_date, grid=fgrd, 
                        native_coordinates='spherical', zone='30N')
#ll_coast = np.zeros((fvg.grid.lon[fvg.grid.coastline].shape[0], 2))
#ll_coast[:, 0] = np.squeeze(fvg.grid.lon[fvg.grid.coastline])
#ll_coast[:, 1] = np.squeeze(fvg.grid.lat[fvg.grid.coastline])

lonc = fvg.grid.lonc
latc = fvg.grid.latc

x = fvg.grid.x
y = fvg.grid.y
xc = fvg.grid.xc
yc = fvg.grid.yc
triangles = fvg.grid.triangles

dist_e = fvcom.grid.nodes2elems(dist, triangles)


# Define transects as lines then expand the line into a polygon

l1_x = np.array([-5.7, -5.86, -6.18, -6.63, -7.13, -7.6, -9.8])
l1_y = np.array([56.6, 56.4, 56.29, 56.12, 55.93, 55.98, 56.3])

l2_x = np.array([-4.9, -5.4])
l2_y = np.array([58.4, 60.4])

l3_x = np.array([-3.3, 0.5])
l3_y = np.array([58.5, 58.7])

l4_x = np.array([-2.0, 1.5])
l4_y = np.array([57.5, 57.0])

l5_x = np.array([-3.15, -2.23])
l5_y = np.array([59.1, 59.28])

l6_x = np.array([-2.23, 0])
l6_y = np.array([59.28, 59.28])

# convert to utm polygons

utm_zone = 30
wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
utm = pyproj.Proj(proj='utm', zone=utm_zone, datum='WGS84')

def line_to_poly(l1_x, l1_y, buff=7500):
  x1, y1  = pyproj.transform(wgs84, utm, l1_x, l1_y)
  line1_utm =LineString((np.array([x1, y1]).T).tolist())
  poly1_utm = line1_utm.buffer(buff)

  p1_utm = extract_poly_coords(poly1_utm)['exterior_coords']
  p1_x = np.array(p1_utm)[:, 0]
  p1_y = np.array(p1_utm)[:, 1]

  px1, py1  = pyproj.transform(utm, wgs84, p1_x, p1_y)
  return px1, py1, poly1_utm

px1, py1, poly1_utm = line_to_poly(l1_x, l1_y)
px2, py2, poly2_utm = line_to_poly(l2_x, l2_y)
px3, py3, poly3_utm = line_to_poly(l3_x, l3_y)
px4, py4, poly4_utm = line_to_poly(l4_x, l4_y)
px5, py5, poly5_utm = line_to_poly(l5_x, l5_y)
px6, py6, poly6_utm = line_to_poly(l6_x, l6_y)

node_p = []
for i in range(len(x)):
  node_p.append(Point(x[i], y[i]))

ele_p = []
for i in range(len(xc)):
  ele_p.append(Point(xc[i], yc[i]))

l1_n = np.zeros((len(x)), dtype=bool)
l2_n = np.zeros((len(x)), dtype=bool)
l3_n = np.zeros((len(x)), dtype=bool)
l4_n = np.zeros((len(x)), dtype=bool)
l5_n = np.zeros((len(x)), dtype=bool)
l6_n = np.zeros((len(x)), dtype=bool)
for j in range(len(x)):
  l1_n[j] = poly1_utm.contains(node_p[j])
  l2_n[j] = poly2_utm.contains(node_p[j])
  l3_n[j] = poly3_utm.contains(node_p[j])
  l4_n[j] = poly4_utm.contains(node_p[j])
  l5_n[j] = poly5_utm.contains(node_p[j])
  l6_n[j] = poly6_utm.contains(node_p[j])

l_e = np.zeros((4, len(xc)), dtype=bool)
for t in range(4):
  for j in range(len(xc)):
    if t == 0:
      l_e[t, j] = poly1_utm.contains(ele_p[j])
    elif t == 1:
      l_e[t, j] = poly2_utm.contains(ele_p[j])
    elif t == 2:
      l_e[t, j] = poly3_utm.contains(ele_p[j])
    elif t == 3:
      l_e[t, j] = poly4_utm.contains(ele_p[j])



#plt.plot(lon, lat, '.', ms=1, zorder=100)
plt.tripcolor(lon, lat, triangles, dist, zorder=99)
if 1:
  plt.plot(l1_x, l1_y, '-', zorder=101)
  plt.plot(px1, py1, '-', zorder=100)
  plt.plot(l2_x, l2_y, '-', zorder=101)
  plt.plot(px2, py2, '-', zorder=100)
  plt.plot(l3_x, l3_y, '-', zorder=101)
  plt.plot(px3, py3, '-', zorder=100)
  plt.plot(l4_x, l4_y, '-', zorder=101)
  plt.plot(px4, py4, '-', zorder=100)
  plt.plot(l5_x, l5_y, '-', zorder=101)
  plt.plot(px5, py5, '-', zorder=100)
  plt.plot(l6_x, l6_y, '-', zorder=101)
  plt.plot(px6, py6, '-', zorder=100)
plt.xlim([-10, 1])
plt.ylim([54, 61]) 



# Loop over data

vars = ('temp', 'salinity', 'ua', 'va', 'Times')
ds = 4 # was 0.1
#dmax = np.max(dist)
dmax = 200
nbin = int(dmax // ds)
m_dist = np.arange(0 + (ds / 2), dmax + (ds / 2), ds)

elem_list = np.zeros((4, nbin), dtype=int)

for t in range(4):
  dist_et = np.ma.masked_where(np.invert(l_e[t, :]), dist_e)
  c = 0
  for b in range(nbin):
    dst = ds * b
    den = ds * (b + 1)
    tmp_e = (dist_et >= dst) & (dist_et < den)
    tmp_e = tmp_e.filled(False)

    ele = np.nonzero(tmp_e)[0]
    #print(b, ele)
    if len(ele) == 0:
      elem_list[t, b] = elem_list[t, b - 1]
    else:
      elem_list[t, b] = ele[0]
  print(lonc[elem_list[t, :]])
  plt.plot(lonc[elem_list[t, :]], latc[elem_list[t, :]], '-w', lw=2, zorder=102)

plt.savefig('./Figures/transect_map.png', dpi=150)

lont = np.zeros(elem_list.shape)
latt = np.zeros(elem_list.shape)
for t in range(elem_list.shape[0]):
  lont[t, :] = lonc[elem_list[t, :]]
  latt[t, :] = latc[elem_list[t, :]]

np.savez(out_dir + 'elem_transect.npz', lont=lont, latt=latt, m_dist=m_dist, elem_list=elem_list)

print(elem_list.shape)
with open('./elem_transect.csv', 'w') as f:
  for i in range(elem_list.shape[0]):
    for j in range(elem_list.shape[1]):
      f.write(str(elem_list[i, j] + 1) + ', ') # fvcom is 1 indexed


