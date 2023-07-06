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

def calc_rho(sp, tp, depth, lon, lat):
  # density 
  pres = sw.p_from_z(depth, lat)
  sa = sw.SA_from_SP(sp, pres, lon, lat)
  ct = sw.CT_from_pt(sa, tp)
  rho = sw.rho(sa, ct, 0)
  print('This should be False:', np.ma.is_masked(pres))
  return rho

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

def calc_coriolis(fc, vel_u, vel_v):
  cx = np.mean(fc * vel_v, axis=0)
  cy = np.mean(-fc * vel_u, axis=0)
  return cx, cy 

def calc_bot_stress(u_bot, v_bot, Hc, rho_0=1025 * 1000, Cd_bot=0.0015):
  tau_bot_u = -rho_0 * Cd_bot * (((u_bot ** 2) + (v_bot ** 2)) ** 0.5) * u_bot
  tau_bot_v = -rho_0 * Cd_bot * (((u_bot ** 2) + (v_bot ** 2)) ** 0.5) * v_bot

  b_x = -tau_bot_u / (rho_0 * Hc)
  b_y = -tau_bot_v / (rho_0 * Hc)
  return b_x, b_y


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

f = sw.f(lat)
fc = sw.f(latc)
g = sw.grav(56, 0)
rho_0 = 1025 * 1000 # convert to g/m^3


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

vars = ('temp', 'salinity', 'u', 'v', 'zeta', 'Times')
ds = 0.1
#dmax = np.max(dist)
dmax = 200
nbin = int(dmax // ds)
m_dist = np.arange(0 + (ds / 2), dmax - (ds / 2), ds)
c = 0

date_list = np.zeros((len(fn) * 24), dtype=object) - 999
cor_x_tran = np.ma.zeros((6, len(fn) * 24, nbin)) - 999
cor_y_tran = np.ma.zeros((6, len(fn) * 24, nbin)) - 999
bsts_x_tran = np.ma.zeros((6, len(fn) * 24, nbin)) - 999
bsts_y_tran = np.ma.zeros((6, len(fn) * 24, nbin)) - 999
pgf_x_tran = np.ma.zeros((6, len(fn) * 24, nbin)) - 999
pgf_y_tran = np.ma.zeros((6, len(fn) * 24, nbin)) - 999


for i in range(10):#len(fn)):
  print(i / len(fn) *100, '%')
  FVCOM = readFVCOM(fn[i], vars, dims=dims)

  for t1 in range(FVCOM['u'].shape[0]):
    #sal = FVCOM['salinity'][t1, :, :]
    #temp = FVCOM['temp'][t1, :, :]
    zeta = FVCOM['zeta'][t1, :]
    u = FVCOM['u'][t1, :, :]
    v = FVCOM['v'][t1, :, :]
    date_list[c] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-7].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S')

    # Calculate 2-d fields: coriolis and bottom stress

    cor_x, cor_y = calc_coriolis(fc, u, v)
    bot_stress_x, bot_stress_y = calc_bot_stress(u[-1, :], v[-1, :], hc_mod)

    # Mask non transect areas

    cor_x_m1 = np.ma.masked_where(np.invert(l1_e), cor_x)
    cor_y_m1 = np.ma.masked_where(np.invert(l1_e), cor_y)
    bsts_x_m1 = np.ma.masked_where(np.invert(l1_e), bot_stress_x)
    bsts_y_m1 = np.ma.masked_where(np.invert(l1_e), bot_stress_y)

    cor_x_m2 = np.ma.masked_where(np.invert(l2_e), cor_x)
    cor_y_m2 = np.ma.masked_where(np.invert(l2_e), cor_y)
    bsts_x_m2 = np.ma.masked_where(np.invert(l2_e), bot_stress_x)
    bsts_y_m2 = np.ma.masked_where(np.invert(l2_e), bot_stress_y)

    cor_x_m3 = np.ma.masked_where(np.invert(l3_e), cor_x)
    cor_y_m3 = np.ma.masked_where(np.invert(l3_e), cor_y)
    bsts_x_m3 = np.ma.masked_where(np.invert(l3_e), bot_stress_x)
    bsts_y_m3 = np.ma.masked_where(np.invert(l3_e), bot_stress_y)

    cor_x_m4 = np.ma.masked_where(np.invert(l4_e), cor_x)
    cor_y_m4 = np.ma.masked_where(np.invert(l4_e), cor_y)
    bsts_x_m4 = np.ma.masked_where(np.invert(l4_e), bot_stress_x)
    bsts_y_m4 = np.ma.masked_where(np.invert(l4_e), bot_stress_y)

    cor_x_m5 = np.ma.masked_where(np.invert(l5_e), cor_x)
    cor_y_m5 = np.ma.masked_where(np.invert(l5_e), cor_y)
    bsts_x_m5 = np.ma.masked_where(np.invert(l5_e), bot_stress_x)
    bsts_y_m5 = np.ma.masked_where(np.invert(l5_e), bot_stress_y)

    cor_x_m6 = np.ma.masked_where(np.invert(l6_e), cor_x)
    cor_y_m6 = np.ma.masked_where(np.invert(l6_e), cor_y)
    bsts_x_m6 = np.ma.masked_where(np.invert(l6_e), bot_stress_x)
    bsts_y_m6 = np.ma.masked_where(np.invert(l6_e), bot_stress_y)

    zeta_m1 = np.ma.masked_where(np.invert(l1_n), zeta)
    zeta_m2 = np.ma.masked_where(np.invert(l2_n), zeta)
    zeta_m3 = np.ma.masked_where(np.invert(l3_n), zeta)
    zeta_m4 = np.ma.masked_where(np.invert(l4_n), zeta)
    zeta_m5 = np.ma.masked_where(np.invert(l5_n), zeta)
    zeta_m6 = np.ma.masked_where(np.invert(l6_n), zeta)

    # Calculate PGF on masked zeta for speed

    pgf_x_m1, pgf_y_m1 = calc_pres_grad(
        zeta_m1, x, y, g, trio, triangles, fvcom_files)
    pgf_x_m2, pgf_y_m2 = calc_pres_grad(
        zeta_m2, x, y, g, trio, triangles, fvcom_files)
    pgf_x_m3, pgf_y_m3 = calc_pres_grad(
        zeta_m3, x, y, g, trio, triangles, fvcom_files)
    pgf_x_m4, pgf_y_m4 = calc_pres_grad(
        zeta_m4, x, y, g, trio, triangles, fvcom_files)
    pgf_x_m5, pgf_y_m5 = calc_pres_grad(
        zeta_m5, x, y, g, trio, triangles, fvcom_files)
    pgf_x_m6, pgf_y_m6 = calc_pres_grad(
        zeta_m6, x, y, g, trio, triangles, fvcom_files)

    # Divide into distance bins

    for b in range(nbin):
      dst = ds * b
      den = ds * (b + 1)
      tmp = (dist >= dst) & (dist < den)
      tmp_e = (dist_e >= dst) & (dist_e < den)

      cor_x_tran[0, c, b] = np.ma.mean(cor_x_m1[tmp_e])
      cor_y_tran[0, c, b] = np.ma.mean(cor_y_m1[tmp_e])
      cor_x_tran[1, c, b] = np.ma.mean(cor_x_m2[tmp_e])
      cor_y_tran[1, c, b] = np.ma.mean(cor_y_m2[tmp_e])
      cor_x_tran[2, c, b] = np.ma.mean(cor_x_m3[tmp_e])
      cor_y_tran[2, c, b] = np.ma.mean(cor_y_m3[tmp_e])
      cor_x_tran[3, c, b] = np.ma.mean(cor_x_m4[tmp_e])
      cor_y_tran[3, c, b] = np.ma.mean(cor_y_m4[tmp_e])
      cor_x_tran[4, c, b] = np.ma.mean(cor_x_m5[tmp_e])
      cor_y_tran[4, c, b] = np.ma.mean(cor_y_m5[tmp_e])
      cor_x_tran[5, c, b] = np.ma.mean(cor_x_m6[tmp_e])
      cor_y_tran[5, c, b] = np.ma.mean(cor_y_m6[tmp_e])

      bsts_x_tran[0, c, b] = np.ma.mean(bsts_x_m1[tmp_e])
      bsts_y_tran[0, c, b] = np.ma.mean(bsts_y_m1[tmp_e])
      bsts_x_tran[1, c, b] = np.ma.mean(bsts_x_m2[tmp_e])
      bsts_y_tran[1, c, b] = np.ma.mean(bsts_y_m2[tmp_e])
      bsts_x_tran[2, c, b] = np.ma.mean(bsts_x_m3[tmp_e])
      bsts_y_tran[2, c, b] = np.ma.mean(bsts_y_m3[tmp_e])
      bsts_x_tran[3, c, b] = np.ma.mean(bsts_x_m4[tmp_e])
      bsts_y_tran[3, c, b] = np.ma.mean(bsts_y_m4[tmp_e])
      bsts_x_tran[4, c, b] = np.ma.mean(bsts_x_m5[tmp_e])
      bsts_y_tran[4, c, b] = np.ma.mean(bsts_y_m5[tmp_e])
      bsts_x_tran[5, c, b] = np.ma.mean(bsts_x_m6[tmp_e])
      bsts_y_tran[5, c, b] = np.ma.mean(bsts_y_m6[tmp_e])

      pgf_x_tran[0, c, b] = np.ma.mean(pgf_x_m1[tmp_e])
      pgf_y_tran[0, c, b] = np.ma.mean(pgf_y_m1[tmp_e])
      pgf_x_tran[1, c, b] = np.ma.mean(pgf_x_m2[tmp_e])
      pgf_y_tran[1, c, b] = np.ma.mean(pgf_y_m2[tmp_e])
      pgf_x_tran[2, c, b] = np.ma.mean(pgf_x_m3[tmp_e])
      pgf_y_tran[2, c, b] = np.ma.mean(pgf_y_m3[tmp_e])
      pgf_x_tran[3, c, b] = np.ma.mean(pgf_x_m4[tmp_e])
      pgf_y_tran[3, c, b] = np.ma.mean(pgf_y_m4[tmp_e])
      pgf_x_tran[4, c, b] = np.ma.mean(pgf_x_m5[tmp_e])
      pgf_y_tran[4, c, b] = np.ma.mean(pgf_y_m5[tmp_e])
      pgf_x_tran[5, c, b] = np.ma.mean(pgf_x_m6[tmp_e])
      pgf_y_tran[5, c, b] = np.ma.mean(pgf_y_m6[tmp_e])


    for t in range(cor_x_tran.shape[0]):
      vsel = cor_x_tran.mask[t, c, :] == 0
      ivsel = np.nonzero(vsel)[0]
      cor_x_tran[t, c, :] = np.interp(
                              m_dist, m_dist[vsel], cor_x_tran[t, c, ivsel])
      cor_y_tran[t, c, :] = np.interp(
                              m_dist, m_dist[vsel], cor_y_tran[t, c, ivsel])
      bsts_x_tran[t, c, :] = np.interp(
                              m_dist, m_dist[vsel], bsts_x_tran[t, c, ivsel])
      bsts_y_tran[t, c, :] = np.interp(
                              m_dist, m_dist[vsel], bsts_y_tran[t, c, ivsel])
      pgf_x_tran[t, c, :] = np.interp(
                              m_dist, m_dist[vsel], pgf_x_tran[t, c, ivsel])
      pgf_y_tran[t, c, :] = np.interp(
                              m_dist, m_dist[vsel], pgf_y_tran[t, c, ivsel])
    c = c + 1

date_list = date_list[0:c]
cor_x_tran = cor_x_tran[:, 0:c, :]
cor_y_tran = cor_y_tran[:, 0:c, :]
bsts_x_tran = bsts_x_tran[:, 0:c, :]
bsts_y_tran = bsts_y_tran[:, 0:c, :]
pgf_x_tran = pgf_x_tran[:, 0:c, :]
pgf_y_tran = pgf_y_tran[:, 0:c, :]

# Get x and y values of the transects

x_m1 = np.ma.masked_where(np.invert(l1_e), xc)
y_m1 = np.ma.masked_where(np.invert(l1_e), yc)
x_m2 = np.ma.masked_where(np.invert(l2_e), xc)
y_m2 = np.ma.masked_where(np.invert(l2_e), yc)
x_m3 = np.ma.masked_where(np.invert(l3_e), xc)
y_m3 = np.ma.masked_where(np.invert(l3_e), yc)
x_m4 = np.ma.masked_where(np.invert(l4_e), xc)
y_m4 = np.ma.masked_where(np.invert(l4_e), yc)
x_m5 = np.ma.masked_where(np.invert(l5_e), xc)
y_m5 = np.ma.masked_where(np.invert(l5_e), yc)
x_m6 = np.ma.masked_where(np.invert(l6_e), xc)
y_m6 = np.ma.masked_where(np.invert(l6_e), yc)

x_tran = np.ma.zeros((6, nbin)) - 999
y_tran = np.ma.zeros((6, nbin)) - 999

for b in range(nbin):
  dst = ds * b
  den = ds * (b + 1)
  tmp = (dist >= dst) & (dist < den)
  tmp_e = (dist_e >= dst) & (dist_e < den)

  x_tran[0, b] = np.ma.mean(x_m1[tmp_e])
  y_tran[0, b] = np.ma.mean(y_m1[tmp_e])
  x_tran[1, b] = np.ma.mean(x_m2[tmp_e])
  y_tran[1, b] = np.ma.mean(y_m2[tmp_e])
  x_tran[2, b] = np.ma.mean(x_m3[tmp_e])
  y_tran[2, b] = np.ma.mean(y_m3[tmp_e])
  x_tran[3, b] = np.ma.mean(x_m4[tmp_e])
  y_tran[3, b] = np.ma.mean(y_m4[tmp_e])
  x_tran[4, b] = np.ma.mean(x_m5[tmp_e])
  y_tran[4, b] = np.ma.mean(y_m5[tmp_e])
  x_tran[5, b] = np.ma.mean(x_m6[tmp_e])
  y_tran[5, b] = np.ma.mean(y_m6[tmp_e])

for t in range(x_tran.shape[0]):
  vsel = x_tran.mask[t, :] == 0
  ivsel = np.nonzero(vsel)[0]
  x_tran[t, :] = np.interp(m_dist, m_dist[vsel], x_tran[t, ivsel])
  y_tran[t, :] = np.interp(m_dist, m_dist[vsel], y_tran[t, ivsel])

# Get angle of each point on the transect

dx = x_tran[:, 1:] - x_tran[:, :-1]
dy = y_tran[:, 1:] - y_tran[:, :-1]
dx = np.append(dx, dx[:, -1:], axis=1)
dy = np.append(dy, dy[:, -1:], axis=1)

angle = np.arctan2(dx, dy) # from North

# Align components to be along and across the transect

def align(x_var, y_var, angle):
  across_var = np.zeros_like(x_var)
  along_var = np.zeros_like(x_var)

  for i in range(x_var.shape[1]): 
    var_mag = ((x_var[:, i, :] ** 2) + (y_var[:, i, :] ** 2)) ** 0.5
    var_dir = np.arctan2(x_var[:, i, :], y_var[:, i, :])
    new_dir = var_dir - angle # subtract so North lies along transect
    across_var[:, i, :] = var_mag * np.sin(new_dir) # x direction
    along_var[:, i, :] = var_mag * np.cos(new_dir) # y direction
  return across_var, along_var

cor_across, cor_along = align(cor_x_tran, cor_y_tran, angle)
bsts_across, bsts_along = align(bsts_x_tran, bsts_y_tran, angle)
pgf_across, pgf_along = align(pgf_x_tran, pgf_y_tran, angle)

cor_across, cor_along = cor_x_tran, cor_y_tran
bsts_across, bsts_along = bsts_x_tran, bsts_y_tran
pgf_across, pgf_along = pgf_x_tran, pgf_y_tran

# Save

np.savez(in_dir + 'forces_transect.npz', cor_across=cor_across, cor_along=cor_along, bsts_across=bsts_across, bsts_along=bsts_along, pgf_across=pgf_across, pgf_along=pgf_along, date_list=date_list, lon=lon, lat=lat, m_dist=m_dist)


