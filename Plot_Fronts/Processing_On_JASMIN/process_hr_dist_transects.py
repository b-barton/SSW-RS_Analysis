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



in_dir = '../Processed_Data/'
mjas = '/gws/nopw/j04/ssw_rs/Model_Output_V3.02/Hourly/'
#fn = sorted(glob.glob(mjas + '*/SSWRS*V1.1*hr*' + 'RE.nc'))
fn = []
for yr in range(2000, 2001):
  fn.extend(sorted(glob.glob(mjas + '*/SSWRS*V3.02*hr*' + str(yr) + '*RE.nc')))

#fn = fn[:180]

fin = '../Input_Files/'
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

l5_x = np.array([-3.15, -2.23, 0])
l5_y = np.array([59.1, 59.28, 59.28])

l6_x = np.array([-2.23, 0])
l6_y = np.array([59.28, 59.28])

# convert to utm polygons

utm_zone = 30
wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
utm = pyproj.Proj(proj='utm', zone=utm_zone, datum='WGS84')

def line_to_poly(l1_x, l1_y, buff=7000):
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

l1_e = np.zeros((len(xc)), dtype=bool)
l2_e = np.zeros((len(xc)), dtype=bool)
l3_e = np.zeros((len(xc)), dtype=bool)
l4_e = np.zeros((len(xc)), dtype=bool)
l5_e = np.zeros((len(xc)), dtype=bool)
l6_e = np.zeros((len(xc)), dtype=bool)
for j in range(len(xc)):
  l1_e[j] = poly1_utm.contains(ele_p[j])
  l2_e[j] = poly2_utm.contains(ele_p[j])
  l3_e[j] = poly3_utm.contains(ele_p[j])
  l4_e[j] = poly4_utm.contains(ele_p[j])
  l5_e[j] = poly5_utm.contains(ele_p[j])
  l6_e[j] = poly6_utm.contains(ele_p[j])


#plt.plot(lon, lat, '.', ms=1, zorder=100)
plt.tripcolor(lon, lat, triangles, dist, zorder=99)
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

#plt.savefig('./Figures/transect_map.png')


# Loop over data

vars = ('temp', 'salinity', 'ua', 'va', 'Times')
ds = 0.1
#dmax = np.max(dist)
dmax = 200
nbin = int(dmax // ds)
m_dist = np.arange(0 + (ds / 2), dmax - (ds / 2), ds)
c = 0

hrl = 24
date_list = np.zeros((len(fn) * hrl), dtype=object) - 999
sal_tran = np.ma.zeros((6, 2, len(fn) * hrl, nbin)) - 999
temp_tran = np.ma.zeros((6, 2, len(fn) * hrl, nbin)) - 999
ua_tran = np.ma.zeros((6, len(fn) * hrl, nbin)) - 999
va_tran = np.ma.zeros((6, len(fn) * hrl, nbin)) - 999


for i in range(len(fn)):
  print(i / len(fn) *100, '%')
  FVCOM = readFVCOM(fn[i], vars, dims=dims)
  for t1 in range(FVCOM['ua'].shape[0]):
    sal = FVCOM['salinity'][t1, :, :]
    temp = FVCOM['temp'][t1, :, :]
    ua = FVCOM['ua'][t1, :]
    va = FVCOM['va'][t1, :]
    date_list[c] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-7].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S')

    # Mask non transect areas

    sal_m1 = np.ma.masked_where(np.invert(l1_n), sal[0, :])
    temp_m1 = np.ma.masked_where(np.invert(l1_n), temp[0, :])
    sal_m2 = np.ma.masked_where(np.invert(l2_n), sal[0, :])
    temp_m2 = np.ma.masked_where(np.invert(l2_n), temp[0, :])
    sal_m3 = np.ma.masked_where(np.invert(l3_n), sal[0, :])
    temp_m3 = np.ma.masked_where(np.invert(l3_n), temp[0, :])
    sal_m4 = np.ma.masked_where(np.invert(l4_n), sal[0, :])
    temp_m4 = np.ma.masked_where(np.invert(l4_n), temp[0, :])
    sal_m5 = np.ma.masked_where(np.invert(l5_n), sal[0, :])
    temp_m5 = np.ma.masked_where(np.invert(l5_n), temp[0, :])
    sal_m6 = np.ma.masked_where(np.invert(l6_n), sal[0, :])
    temp_m6 = np.ma.masked_where(np.invert(l6_n), temp[0, :])

    sal_b1 = np.ma.masked_where(np.invert(l1_n), sal[-1, :])
    temp_b1 = np.ma.masked_where(np.invert(l1_n), temp[-1, :])
    sal_b2 = np.ma.masked_where(np.invert(l2_n), sal[-1, :])
    temp_b2 = np.ma.masked_where(np.invert(l2_n), temp[-1, :])
    sal_b3 = np.ma.masked_where(np.invert(l3_n), sal[-1, :])
    temp_b3 = np.ma.masked_where(np.invert(l3_n), temp[-1, :])
    sal_b4 = np.ma.masked_where(np.invert(l4_n), sal[-1, :])
    temp_b4 = np.ma.masked_where(np.invert(l4_n), temp[-1, :])
    sal_b5 = np.ma.masked_where(np.invert(l5_n), sal[-1, :])
    temp_b5 = np.ma.masked_where(np.invert(l5_n), temp[-1, :])
    sal_b6 = np.ma.masked_where(np.invert(l6_n), sal[-1, :])
    temp_b6 = np.ma.masked_where(np.invert(l6_n), temp[-1, :])

    ua_m1 = np.ma.masked_where(np.invert(l1_e), ua)
    va_m1 = np.ma.masked_where(np.invert(l1_e), va)
    ua_m2 = np.ma.masked_where(np.invert(l2_e), ua)
    va_m2 = np.ma.masked_where(np.invert(l2_e), va)
    ua_m3 = np.ma.masked_where(np.invert(l3_e), ua)
    va_m3 = np.ma.masked_where(np.invert(l3_e), va)
    ua_m4 = np.ma.masked_where(np.invert(l4_e), ua)
    va_m4 = np.ma.masked_where(np.invert(l4_e), va)
    ua_m5 = np.ma.masked_where(np.invert(l5_e), ua)
    va_m5 = np.ma.masked_where(np.invert(l5_e), va)
    ua_m6 = np.ma.masked_where(np.invert(l6_e), ua)
    va_m6 = np.ma.masked_where(np.invert(l6_e), va)


    # Divide into distance bins

    for b in range(nbin):
      dst = ds * b
      den = ds * (b + 1)
      tmp = (dist >= dst) & (dist < den)
      tmp_e = (dist_e >= dst) & (dist_e < den)
      sal_tran[0, 0, c, b] = np.ma.mean(sal_m1[tmp])
      temp_tran[0, 0, c, b] = np.ma.mean(temp_m1[tmp])
      sal_tran[1, 0, c, b] = np.ma.mean(sal_m2[tmp])
      temp_tran[1, 0, c, b] = np.ma.mean(temp_m2[tmp])
      sal_tran[2, 0, c, b] = np.ma.mean(sal_m3[tmp])
      temp_tran[2, 0, c, b] = np.ma.mean(temp_m3[tmp])
      sal_tran[3, 0, c, b] = np.ma.mean(sal_m4[tmp])
      temp_tran[3, 0, c, b] = np.ma.mean(temp_m4[tmp])
      sal_tran[4, 0, c, b] = np.ma.mean(sal_m5[tmp])
      temp_tran[4, 0, c, b] = np.ma.mean(temp_m5[tmp])
      sal_tran[5, 0, c, b] = np.ma.mean(sal_m6[tmp])
      temp_tran[5, 0, c, b] = np.ma.mean(temp_m6[tmp])

      sal_tran[0, 1, c, b] = np.ma.mean(sal_b1[tmp])
      temp_tran[0, 1, c, b] = np.ma.mean(temp_b1[tmp])
      sal_tran[1, 1, c, b] = np.ma.mean(sal_b2[tmp])
      temp_tran[1, 1, c, b] = np.ma.mean(temp_b2[tmp])
      sal_tran[2, 1, c, b] = np.ma.mean(sal_b3[tmp])
      temp_tran[2, 1, c, b] = np.ma.mean(temp_b3[tmp])
      sal_tran[3, 1, c, b] = np.ma.mean(sal_b4[tmp])
      temp_tran[3, 1, c, b] = np.ma.mean(temp_b4[tmp])
      sal_tran[4, 1, c, b] = np.ma.mean(sal_b5[tmp])
      temp_tran[4, 1, c, b] = np.ma.mean(temp_b5[tmp])
      sal_tran[5, 1, c, b] = np.ma.mean(sal_b6[tmp])
      temp_tran[5, 1, c, b] = np.ma.mean(temp_b6[tmp])

      ua_tran[0, c, b] = np.ma.mean(ua_m1[tmp_e])
      va_tran[0, c, b] = np.ma.mean(va_m1[tmp_e])
      ua_tran[1, c, b] = np.ma.mean(ua_m2[tmp_e])
      va_tran[1, c, b] = np.ma.mean(va_m2[tmp_e])
      ua_tran[2, c, b] = np.ma.mean(ua_m3[tmp_e])
      va_tran[2, c, b] = np.ma.mean(va_m3[tmp_e])
      ua_tran[3, c, b] = np.ma.mean(ua_m4[tmp_e])
      va_tran[3, c, b] = np.ma.mean(va_m4[tmp_e])
      ua_tran[4, c, b] = np.ma.mean(ua_m5[tmp_e])
      va_tran[4, c, b] = np.ma.mean(va_m5[tmp_e])
      ua_tran[5, c, b] = np.ma.mean(ua_m6[tmp_e])
      va_tran[5, c, b] = np.ma.mean(va_m6[tmp_e])

    for t in range(sal_tran.shape[0]):
      vsel = ua_tran.mask[t, c, :] == 0
      ivsel = np.nonzero(vsel)[0]
      ua_tran[t, c, :] = np.interp(
                              m_dist, m_dist[vsel], ua_tran[t, c, ivsel], right=-999, left=-999)
      va_tran[t, c, :] = np.interp(
                              m_dist, m_dist[vsel], va_tran[t, c, ivsel], right=-999, left=-999)

      for u in range(sal_tran.shape[1]):
        sel = sal_tran.mask[t, u, c, :] == 0
        isel = np.nonzero(sel)[0]
        sal_tran[t, u, c, :] = np.interp(
                              m_dist, m_dist[sel], sal_tran[t, u, c, isel], right=-999, left=-999)
        temp_tran[t, u, c, :] = np.interp(
                              m_dist, m_dist[sel], temp_tran[t, u, c, isel], right=-999, left=-999)

    c = c + 1

date_list = date_list[0:c]
sal_tran = sal_tran[:, :, 0:c, :]
temp_tran = temp_tran[:, :, 0:c, :]
ua_tran = ua_tran[:, 0:c, :]
va_tran = va_tran[:, 0:c, :]

np.savez(in_dir + 'transect_hr.npz', sal_tran=sal_tran, temp_tran=temp_tran, ua_tran=ua_tran, va_tran=va_tran, date_list=date_list, lon=lon, lat=lat, m_dist=m_dist)


