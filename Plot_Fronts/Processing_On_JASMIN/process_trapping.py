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
from scipy.signal import argrelextrema

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

in_dir = '../Processed_Data/'
mjas = '/gws/nopw/j04/ssw_rs/Model_Output_V3.02/Daily/'
#mjas = '/gws/nopw/j04/ssw_rs/Model_Output_V3.02/Hourly/'
#fn = sorted(glob.glob(mjas + '*/SSWRS*V1.1*dy*' + 'RE.nc'))
fn = []
#for yr in range(1993, 2020):
for yr in range(1999, 2000):
  fn.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V3.02*dy*RE.nc')))



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


if 0:

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

  #varlist = ['temp', 'salinity']
  #dims = {'siglay': [0]}
  #fvcom_files = MFileReader(fn[0], variables=varlist, dims=dims)
  #fvcom_files.grid.lon[fvcom_files.grid.lon > 180] = (fvcom_files.grid.lon[
  #    fvcom_files.grid.lon > 180] - 360)

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

  l_n = np.zeros((6, len(x)), dtype=bool)
  for j in range(len(x)):
    l_n[0, j] = poly1_utm.contains(node_p[j])
    l_n[1, j] = poly2_utm.contains(node_p[j])
    l_n[2, j] = poly3_utm.contains(node_p[j])
    l_n[3, j] = poly4_utm.contains(node_p[j])
    l_n[4, j] = poly5_utm.contains(node_p[j])
    l_n[5, j] = poly6_utm.contains(node_p[j])

  l_e = np.zeros((6, len(xc)), dtype=bool)
  for j in range(len(xc)):
    l_e[0, j] = poly1_utm.contains(ele_p[j])
    l_e[1, j] = poly2_utm.contains(ele_p[j])
    l_e[2, j] = poly3_utm.contains(ele_p[j])
    l_e[3, j] = poly4_utm.contains(ele_p[j])
    l_e[4, j] = poly5_utm.contains(ele_p[j])
    l_e[5, j] = poly6_utm.contains(ele_p[j])

  np.savez(in_dir + 'transect_mask.npz', l_e=l_e, l_n=l_n, x=x, y=y)

else:
  data = np.load(in_dir + 'transect_mask.npz')
  l_e = data['l_e']
  l_n = data['l_n']
  x = data['x']
  y = data['y']
  data.close()

trio = tri.Triangulation(x, y, triangles=np.asarray(triangles))
dist_e = fvcom.grid.nodes2elems(dist, triangles)

ds = 0.1
#dmax = np.max(dist)
dmax = 200
nbin = int(dmax // ds)
m_dist = np.arange(0 + (ds / 2), dmax - (ds / 2), ds)

# Get x and y values of the transects

node_list = np.zeros((6, nbin), dtype=int)

dist_nt = np.ma.zeros((6, dist.shape[0]))
for t in range(6):
  dist_nt[t, :] = np.ma.masked_where(np.invert(l_n[t]), dist)
  c = 0
  for b in range(nbin):
    dst = ds * b
    den = ds * (b + 1)
    tmp_n = (dist_nt[t, :] >= dst) & (dist_nt[t, :] < den)
    tmp_n = tmp_n.filled(False)

    node = np.nonzero(tmp_n)[0]
    #print(b, ele)
    if len(node) == 0:
      node_list[t, b] = node_list[t, b - 1]
    else:
      node_list[t, b] = node[0]

x_tran = x[node_list]
y_tran = y[node_list]

dx = x_tran[:, -1] - x_tran[:, 0]
dy = y_tran[:, -1] - y_tran[:, 0]

angle = np.arctan2(dx, dy) # from North

# Align components to be along and across the transect

def align(x_var, y_var, angle):
  across_var = np.zeros_like(x_var)
  along_var = np.zeros_like(x_var)

  for i in range(x_var.shape[0]): 
    var_mag = ((x_var[i, :] ** 2) + (y_var[i, :] ** 2)) ** 0.5
    var_dir = np.arctan2(x_var[i, :], y_var[i, :])
    new_dir = var_dir - angle[i] # subtract so North lies along transect
    across_var[i, :] = var_mag * np.sin(new_dir) # x direction
    along_var[i, :] = var_mag * np.cos(new_dir) # y direction
  return across_var, along_var


# constants 

f = sw.f(np.mean(lat))
g = sw.grav(np.mean(lat), 0)
rho_0 = 1025 # convert to g/m^3

r = 0.0005 # m/s bottom friction coefficient, taken from paper
ku = 1e-5 # m2/s vertical mixing coefficient, from nml
sig = ((2 * ku) / f) ** 0.5 # vertical scale of bottom Ekman layer
print(sig)
sig = 4.47 # from paper
end_t = 1000 # index of end of transect
start_t = 100
h_0 = h_mod[node_list[:4, 0]] # coast wall depth at y = 0
s = ((h_mod[node_list[:4, end_t]] - h_mod[node_list[:4, 0]]) 
    / ((m_dist[end_t] - m_dist[0]) * 1000)) # bottom slope in m

#y_f[:, c] = ((f * rho_0) / (s * g)) * (uv_max / (drho_dy)) 
#        * (1 + ((sig * dphi_dx) / (r * uv_max)))
#        - (h_0 / s)
y_f = (((1e-4 * rho_0) / (0.001 * 9.81)) * (0.275 / -0.00005) 
        * (1 + ((sig * -0.018) / (r * 0.275)))
        - (25 / 0.001)) / 1000 # m
print(y_f/1000)

# Loop over data

vars = ('temp', 'salinity', 'u', 'v', 'zeta', 'Times')

c = 0

date_list = np.zeros((len(fn) * 24), dtype=object) - 999
y_f = np.ma.zeros((4, len(fn) * 24)) - 999


for i in range(10):#len(fn)):
  print(i / len(fn) *100, '%')
  FVCOM = readFVCOM(fn[i], vars, dims=dims)

  for t1 in range(FVCOM['u'].shape[0]):

    date_list[c] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-7].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S')

    H = (h_mod + FVCOM['zeta'][t1, :]) # positive
    depth_mod = -H * siglay_mod # H should be negative (siglay is negative)

    # depth of rho
    surf = 0
    mid = 4
    bot = -1
    rho = calc_rho(FVCOM['salinity'][t1, surf, :], FVCOM['temp'][t1, surf, :], 
              -depth_mod[surf, :], lon, lat) 
    a = tri.LinearTriInterpolator(trio, rho)
    drho_dx1, drho_dy1 = a.gradient(x, y)

    pres = sw.p_from_z(-H, lat) * 10000 # convert to Pa
    phi = pres / rho_0 # dynamic pressure (pressure / rho_0)
    a = tri.LinearTriInterpolator(trio, phi)
    dphi_dx1, dphi_dy1 = a.gradient(x, y)

    u = fvcom.grid.elems2nodes(FVCOM['u'][t1, surf, :], triangles) 
    v = fvcom.grid.elems2nodes(FVCOM['v'][t1, surf, :], triangles) 
    u_bot = fvcom.grid.elems2nodes(FVCOM['u'][t1, bot, :], triangles) 
    v_bot = fvcom.grid.elems2nodes(FVCOM['v'][t1, bot, :], triangles) 

    drho_dx_tran = np.ma.zeros((4, nbin))
    drho_dy_tran = np.ma.zeros((4, nbin))
    dphi_dx_tran = np.ma.zeros((4, nbin))
    dphi_dy_tran = np.ma.zeros((4, nbin))
    u_tran = np.ma.zeros((4, nbin))
    v_tran = np.ma.zeros((4, nbin))
    ub_tran = np.ma.zeros((4, nbin))
    vb_tran = np.ma.zeros((4, nbin))

    for t in range(4):
      drho_dx_m = np.ma.masked_where(np.invert(l_n[t]), drho_dx1)
      drho_dy_m = np.ma.masked_where(np.invert(l_n[t]), drho_dy1)
      dphi_dx_m = np.ma.masked_where(np.invert(l_n[t]), dphi_dx1)
      dphi_dy_m = np.ma.masked_where(np.invert(l_n[t]), dphi_dy1)
      u_m = np.ma.masked_where(np.invert(l_n[t]), u)
      v_m = np.ma.masked_where(np.invert(l_n[t]), v)
      ub_m = np.ma.masked_where(np.invert(l_n[t]), u_bot)
      vb_m = np.ma.masked_where(np.invert(l_n[t]), v_bot)

      for b in range(nbin):
        dst = ds * b
        den = ds * (b + 1)
        tmp = (dist >= dst) & (dist < den)

        drho_dx_tran[t, b] = np.ma.mean(drho_dx_m[tmp])
        drho_dy_tran[t, b] = np.ma.mean(drho_dy_m[tmp])
        dphi_dx_tran[t, b] = np.ma.mean(dphi_dx_m[tmp])
        dphi_dy_tran[t, b] = np.ma.mean(dphi_dy_m[tmp])
        u_tran[t, b] = np.ma.mean(u_m[tmp])
        v_tran[t, b] = np.ma.mean(v_m[tmp])
        ub_tran[t, b] = np.ma.mean(ub_m[tmp])
        vb_tran[t, b] = np.ma.mean(vb_m[tmp])

      vsel = drho_dx_tran.mask[t, :] == 0
      ivsel = np.nonzero(vsel)[0]
      drho_dx_tran[t, :] = np.interp(
                              m_dist, m_dist[vsel], drho_dx_tran[t, ivsel])
      drho_dy_tran[t, :] = np.interp(
                              m_dist, m_dist[vsel], drho_dy_tran[t, ivsel])
      dphi_dx_tran[t, :] = np.interp(
                              m_dist, m_dist[vsel], dphi_dx_tran[t, ivsel])
      dphi_dy_tran[t, :] = np.interp(
                              m_dist, m_dist[vsel], dphi_dy_tran[t, ivsel])
      u_tran[t, :] = np.interp(
                              m_dist, m_dist[vsel], u_tran[t, ivsel])
      v_tran[t, :] = np.interp(
                              m_dist, m_dist[vsel], v_tran[t, ivsel])
      ub_tran[t, :] = np.interp(
                              m_dist, m_dist[vsel], ub_tran[t, ivsel])
      vb_tran[t, :] = np.interp(
                              m_dist, m_dist[vsel], vb_tran[t, ivsel])

    # along and across transect, not shelf

    drho_across, drho_along = align(drho_dx_tran, drho_dy_tran, angle)
    dphi_across, dphi_along = align(dphi_dx_tran, dphi_dy_tran, angle)
    uv_across, uv_along = align(u_tran, v_tran, angle)
    uvb_across, uvb_along = align(ub_tran, vb_tran, angle)

    run = 50
    uvb_run = uvb_along * 1
    uvs_run = uv_along * 1
    for r in range(uvb_along.shape[1] - run):
      uvb_run[:, r + (run//2)] = np.ma.mean(uvb_along[:, r:r + run], axis=1)
      uvs_run[:, r + (run//2)] = np.ma.mean(uv_along[:, r:r + run], axis=1)
    uvb_run = uvb_run[:, start_t:end_t]
    uvs_run = uvs_run[:, start_t:end_t]

    # find area with positive density change
    # only look 10 to 100 km from coast in front region

    run = 50
    drho_run = drho_across * 1
    for r in range(drho_across.shape[1] - run):
      drho_run[:, r + (run//2)] = np.ma.mean(drho_across[:, r:r + run], axis=1)
    drho_run = drho_run[:, start_t:end_t]

    m_dist_sel = m_dist[start_t:end_t]

    uv_max = np.zeros((4))
    drho_dy = np.zeros((4))
    dphi_dx = np.zeros((4))
    u_bot_min = np.zeros((4))

    for t in range(4):
      ind_min = argrelextrema(np.abs(drho_run[t, :]), np.less, order=50)[0]
      ind_min = np.concatenate((np.array([0]), ind_min, 
          np.array([drho_run.shape[1] - 1])))

      ind_mask = np.ma.zeros((ind_min.shape[0] - 1), dtype=bool)
      for p in range(len(ind_min) - 1):
        ind_mask[p] = np.ma.mean(
            drho_run[t, ind_min[p]:ind_min[p + 1]]) > 0

      if t == 1:
        plt.plot(m_dist_sel, drho_run[t, :], 'b')
        plt.plot(m_dist_sel[ind_min], drho_run[t, ind_min], 'ko')
        for p in range(len(ind_min) - 1):
          if ind_mask[p]:
            plt.plot(m_dist_sel[ind_min[p]:ind_min[p + 1]], 
                drho_run[t, :][ind_min[p]:ind_min[p + 1]], 'r')

        plt.ylabel('Density Gradient (kg/m$^{4}$)')
        plt.xlabel('Distance (km)')

        plt.savefig('./Figures/rho_along.png')
        plt.clf()

      #print(m_dist_sel[ind_min])

      # positive density gradient front indices

      im1a = ind_min[:-1][ind_mask]
      im2a = ind_min[1:][ind_mask]
      imd = im2a - im1a
      im1 = (im1a + (imd * 0.2)).astype(int)
      im2 = (im1a + (imd * 0.8)).astype(int)
      im3 = (im1a + (imd * 0.5)).astype(int)
      if len(im1) == 0:
        continue

      print(m_dist_sel[im3[0]])

      if 1:
        uv_max[t] = np.ma.max(uv_across[t, start_t:end_t][im1[0]:im2[0]]) # along shelf jet velocity
        drho_dy[t] = np.ma.max(drho_across[t, start_t:end_t][im1[0]:im2[0]])
        dphi_dx[t] = np.ma.mean(dphi_along[t, start_t:end_t][im1[0]:im2[0]])

      else:
        uv_max[t] = np.ma.max(uv_across[t, start_t:end_t]) # along shelf jet velocity
        drho_dy[t] = np.ma.mean(drho_across[t, start_t:end_t])
        dphi_dx[t] = np.ma.mean(dphi_along[t, start_t:end_t])


      # find local minima in across shelf bottom velocity
      ind_min = argrelextrema(np.abs(uvb_run[t, :]), np.less, order=50)[0]
      ind_min = np.concatenate((np.array([0]), ind_min, 
          np.array([uvb_run.shape[1] - 1])))

      # only use velocity minimum after positive velocity
      ind_mask = np.ma.zeros((ind_min.shape[0] - 1), dtype=bool)
      for p in range(len(ind_min) - 1):
        ind_mask[p] = np.ma.mean(
            uvb_run[t, ind_min[p]:ind_min[p + 1]]) > 0
      iu = ind_min[1:][ind_mask]
      print(m_dist_sel[iu])

      ind_dist = np.argmin(np.abs(m_dist_sel[ind_min] - m_dist_sel[im1[0]]))
      u_bot_min[t] = m_dist_sel[ind_min[ind_dist]]

      if t == 1:
        plt.plot(m_dist_sel, uvb_run[t, :], 'b')
        #plt.plot(m_dist_sel, uvs_run[t, :], 'g')
        plt.plot(m_dist_sel[im3[0]], uvb_run[t, im3[0]], 'ro')
        plt.plot(m_dist_sel[ind_min], uvb_run[t, ind_min], 'ko')
        plt.plot(m_dist_sel[ind_min[ind_dist]], uvb_run[t, ind_min[ind_dist]], 'g.')
        plt.ylabel('Cross Shelf Velocity (m/s)')
        plt.xlabel('Distance (km)')
        plt.savefig('./Figures/u_bot_min.png')
        plt.clf()

    print('drho_dy', drho_dy)
    print('dphi_dx', dphi_dx)
    print('uv_max', uv_max)
    y_f[:, c] = (((f * rho_0) / (s * g)) * (uv_max / drho_dy) 
                * (1 + ((sig * dphi_dx) / (r * uv_max))) 
                - (h_0 / s)) / 1000 # km
    print('y_f:', y_f[:, c])

    print('bot', u_bot_min)

    c = c + 1

y_f = np.ma.masked_where((y_f > 100) | (y_f < 10))
y_f = y_f.filled(-999)

np.savez(in_dir + 'trapped_front.npz', y_f=y_f)

