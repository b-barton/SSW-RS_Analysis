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

if 1:
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

  # Scotland clockwise
  tracks = [None] * 6
  tracks[0] = np.array([[x_mid1, y_min], [x_mid2, y_min]])
  tracks[1] = np.array([[x_min, y_mid1], [x_min, y_cor]])
  tracks[2] = np.array([[x_min, y_cor], [x_cor, y_max]])
  tracks[3] = np.array([[x_cor, y_max], [x_max, y_max]])
  tracks[4] = np.array([[x_max, y_max], [x_max, y_min]])
  tracks[5] = np.array([[x_max, y_min], [x_mid3, y_min]])

else:
  # simple box
  x_min = -9
  x_max = -8.75

  y_min = 57.5
  y_max = 58

  tracks = [None] * 4
  tracks[0] = np.array([[x_min, y_min], [x_min, y_max]])
  tracks[1] = np.array([[x_min, y_max], [x_max, y_max]])
  tracks[2] = np.array([[x_max, y_max], [x_max, y_min]])
  tracks[3] = np.array([[x_max, y_min], [x_min, y_min]])

sel_ind = np.invert((lon >= x_min) & (lon <= x_max) 
                  & (lat >= y_min) & (lat <= y_max))


plt.tripcolor(lon, lat, triangles, sel_ind)
for i in range(len(tracks)):
  plt.plot(tracks[i][:, 0], tracks[i][:, 1], '-')
plt.xlim([x_min-0.5, x_max + 0.5])
plt.ylim([y_min-0.5, y_max + 0.5])


def element_sample(xc, yc, positions, nbe):
    """
    Find the shortest path between the sets of positions using the unstructured grid triangulation.
    Returns element indices and a distance along the line (in metres).
    Parameters
    ----------
    xc, yc : np.ndarray
        Position arrays for the unstructured grid element centres (decimal degrees).
    positions : np.ndarray
        Coordinate pairs of the sample line coordinates np.array([[x1, y1], ..., [xn, yn]] in decimal degrees.
    nbe : np.array
        Elements around each element
    Returns
    -------
    indices : np.ndarray
        List of indices for the elements used in the transect.
    distance : np.ndarray, optional
        The distance along the line in metres described by the elements in indices.
    Notes
    -----
    This is lifted and adjusted for use with PyFVCOM from PySeidon.utilities.shortest_element_path.
    """
    grid = np.array((xc, yc)).T

    # Create a set for edges that are indices of the points.
    edges = []
    for elem_c, vertices in enumerate(nbe):
        # For each edge of the triangle, sort the vertices (sorting avoids duplicated edges being added to the set)
        # and add to the edges set.
        for vertex in vertices:
          if vertex != -1:
            edge = sorted([elem_c, vertex])
            a = grid[edge[0]]
            b = grid[edge[1]]
            weight = (np.hypot(a[0] - b[0], a[1] - b[1]))
            edges.append((edge[0], edge[1], {'weight': weight}))

    # Make a graph based on the Delaunay triangulation edges.
    graph = networkx.Graph(edges)

    # List of elements forming the shortest path.
    elements = []
    for position in zip(positions[:-1], positions[1:]):
        # We need grid indices for networkx.shortest_path rather than positions, so for the current pair of positions,
        # find the closest element IDs.
        source = np.argmin(np.hypot(xc - position[0][0], yc - position[0][1]))
        target = np.argmin(np.hypot(xc - position[1][0], yc - position[1][1]))
        elements += networkx.shortest_path(graph, source=source, target=target, weight='weight')

    # Calculate the distance along the transect in kilometres (use the fast-but-less-accurate Haversine function rather
    # than the slow-but-more-accurate Vincenty distance function).
    distance = np.array([fvcom.grid.haversine_distance((xc[i], yc[i]), (xc[i + 1], yc[i + 1])) for i in elements[:-1]]) * 1000 # convert km to m

    return np.asarray(elements), distance

def make_proj(x_origin, y_origin):
  aeqd = pyproj.Proj(proj='aeqd', datum='WGS84', 
      lon_0=x_origin, lat_0=y_origin, units='m')
  return aeqd

def get_angle(lon_tran, lat_tran, wgs84):
  angle = np.zeros((len(lon_tran) - 1))
  for i in range(len(lon_tran) -1):
    lon_mean = np.mean(lon_tran[i:i + 2])
    lat_mean = np.mean(lat_tran[i:i + 2])
    aeqd = make_proj(lon_mean, lat_mean)
    xt, yt = pyproj.transform(
        wgs84, aeqd, lon_tran[i:i + 2], lat_tran[i:i + 2])
    dx = xt[1] - xt[0]
    dy = yt[1] - yt[0]
    angle[i] = np.arctan2(dx, dy)
  return angle

wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')

# calculate on nodes instead of between elements

ind = [None] * len(tracks)
dist = [None] * len(tracks)
lon_l = [None] * len(tracks)
lat_l = [None] * len(tracks)
lon_all = np.array([])
lat_all = np.array([])
elem_sub = [None] * len(tracks)
node_sub = [None] * len(tracks)
tri_sub = [None] * len(tracks)
trio_n = [None] * len(tracks)
trio_c = [None] * len(tracks)
elem_pair = [None] * len(tracks)
node_pair = [None] * len(tracks)
angle = [None] * len(tracks)

for tr in range(len(tracks)):
  #ind, dist = fvg.grid.horizontal_transect_nodes(positions)
  # Extract node IDs along a line defined by 
  # `positions' [[x1, y1], [x2, y2], ..., [xn, yn]]
  ind[tr], dist[tr] = element_sample(lon, lat, tracks[tr], nbsn)
  dist[tr] = np.tile(dist[tr], (siglay_mod.shape[0], 1))

  # use number of nodes to make regular section
  lon_full = np.linspace(tracks[tr][0, 0], tracks[tr][1, 0], len(ind[tr]))
  lat_full = np.linspace(tracks[tr][0, 1], tracks[tr][1, 1], len(ind[tr]))
  lon_l[tr] = (lon_full[1:] + lon_full[:-1]) / 2
  lat_l[tr] = (lat_full[1:] + lat_full[:-1]) / 2
  lon_all = np.append(lon_all, lon_l[tr])
  lat_all = np.append(lat_all, lat_l[tr])

  #dist[tr] = np.array([fvcom.grid.haversine_distance((lon_full[i], lat_full[i]), (lon_full[i + 1], lat_full[i + 1])) for i in range(len(lon_full[:-1]))]) * 1000 # convert km to m
  dist[tr] = np.array([arc_dist(lon_full[i], lat_full[i], 
      lon_full[i + 1], lat_full[i + 1]) for i in range(len(lon_full[:-1]))])
  dist[tr] = np.tile(dist[tr], (siglay_mod.shape[0], 1))

  dxl = tracks[tr][1, 0] - tracks[tr][0, 0]
  dyl = tracks[tr][1, 1] - tracks[tr][0, 1]
  xt = x[ind[tr]]
  yt = y[ind[tr]]
  dx = xt[1:] - xt[:-1]
  dy = yt[1:] - yt[:-1]

  extra = 0.5
  elem_sub[tr] = ((lonc >= np.min(tracks[tr][:, 0]) - extra) 
                  & (lonc <= np.max(tracks[tr][:, 0]) + extra) 
                  & (latc >= np.min(tracks[tr][:, 1]) - extra) 
                  & (latc <= np.max(tracks[tr][:, 1]) + extra))
  tri_sub[tr] = triangles[elem_sub[tr], :]
  node_sub[tr] = np.unique(tri_sub[tr])
  trio_n[tr] = tri.Triangulation(lon[node_sub[tr]], lat[node_sub[tr]])
  trio_c[tr] = tri.Triangulation(lonc[elem_sub[tr]], latc[elem_sub[tr]])

  print(np.sum(elem_sub[tr]), tri_sub[tr].shape)

  angle1 = np.arctan2(dx, dy) # from North
  angle[tr] = get_angle(lon_full, lat_full, wgs84)
  print(stats.circmean(angle[tr] % (2 * np.pi)) * 180/np.pi, np.arctan2(dxl, dyl) * 180/np.pi, stats.circmean(angle1 % (2 * np.pi)) * 180/np.pi)

  print(ind[tr].shape, dist[tr].shape)
  plt.scatter(lon_l[tr], lat_l[tr], c=dist[tr][0, :], vmin=0, vmax=10000)
  plt.plot(lon_l[tr][0], lat_l[tr][0], 'ro', ms=5)

plt.savefig('./Figures/tracks_node.png' )
print('Plotted')



def align(x_var, y_var, angle):
  # Align components to be along and across the transect
  # angle is clockwise from North

  across_var = np.zeros_like(x_var)
  along_var = np.zeros_like(x_var)

  #for i in range(x_var.shape[1]): 
  var_mag = ((x_var[:] ** 2) + (y_var[:] ** 2)) ** 0.5
  var_dir = np.arctan2(x_var[:], y_var[:])
  new_dir = var_dir - angle[:] # subtract so North lies along transect
  across_var[:] = var_mag * np.sin(new_dir) # x direction
  along_var[:] = var_mag * np.cos(new_dir) # y direction
  return across_var, along_var

vars = ('zeta', 'salinity', 'u', 'v', 'Times', 'h')
dims = {'time':':'}

across_trans = np.ma.zeros((len(fn) * 10, len(tracks))) - 999
along_trans = np.ma.zeros((len(fn) * 10, len(tracks))) - 999
across_fw = np.ma.zeros((len(fn) * 10, len(tracks))) - 999
along_fw = np.ma.zeros((len(fn) * 10, len(tracks))) - 999
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
    u = FVCOM['u'][t1, :, :] # time, depth, elem
    v = FVCOM['v'][t1, :, :] # time, depth, elem
    zeta = FVCOM['zeta'][t1, :]
    sal = FVCOM['salinity'][t1, :, :] # time, depth, node

    # calculate depth along tracks

    zeta_c = fvcom.grid.nodes2elems(zeta, triangles)

    Hc = hc + zeta_c
    depth_mod_c = -Hc * siglay_mod_c # should be negative (siglay is negative)
    depthlev_mod_c = -Hc * siglev_mod_c 
    d_depth_c = (depthlev_mod_c[1:, :] - depthlev_mod_c[:-1, :])

    H = h + zeta
    depth_mod = -H * siglay_mod # should be negative (siglay is negative)
    depthlev_mod = -H * siglev_mod 
    d_depth = (depthlev_mod[1:, :] - depthlev_mod[:-1, :])

    # Multipy each by their depth then for final fw divide by total depth
    u_t = u * d_depth_c
    v_t = v * d_depth_c
    s_t = sal * d_depth

    for tr in range(len(tracks)):
      # convert to lines

      #u_n = fvcom.grid.elems2nodes(u, triangles)
      #v_n = fvcom.grid.elems2nodes(v, triangles)

      line_u = np.zeros((d_depth.shape[0], len(lon_l[tr])))
      line_v = np.zeros((d_depth.shape[0], len(lon_l[tr])))
      line_s = np.zeros((d_depth.shape[0], len(lon_l[tr])))
      for d in range(d_depth.shape[0]):
        interpolator_u = tri.CubicTriInterpolator(
            trio_c[tr], u_t[d, elem_sub[tr]], kind='geom')
        interpolator_v = tri.CubicTriInterpolator(
            trio_c[tr], v_t[d, elem_sub[tr]], kind='geom')
        interpolator_s = tri.CubicTriInterpolator(
            trio_n[tr], s_t[d, node_sub[tr]], kind='geom')

        line_u1 = interpolator_u(lon_l[tr], lat_l[tr])
        line_v1 = interpolator_v(lon_l[tr], lat_l[tr])
        line_s1 = interpolator_s(lon_l[tr], lat_l[tr])

        interpolator_u = tri.LinearTriInterpolator(
            trio_c[tr], u_t[d, elem_sub[tr]])
        interpolator_v = tri.LinearTriInterpolator(
            trio_c[tr], v_t[d, elem_sub[tr]])
        interpolator_s = tri.LinearTriInterpolator(
            trio_n[tr], s_t[d, node_sub[tr]])

        line_u2 = interpolator_u(lon_l[tr], lat_l[tr])
        line_v2 = interpolator_v(lon_l[tr], lat_l[tr])
        line_s2 = interpolator_s(lon_l[tr], lat_l[tr])

        # average cubic and linear to improve accuracy
        line_u[d, :] = (line_u1 + line_u2) / 2
        line_v[d, :] = (line_v1 + line_v2) / 2
        line_s[d, :] = (line_s1 + line_s2) / 2


      interpolator_hc = tri.CubicTriInterpolator(
            trio_c[tr], Hc[elem_sub[tr]], kind='geom')
      interpolator_h = tri.CubicTriInterpolator(
            trio_n[tr], H[node_sub[tr]], kind='geom')
      line_h = interpolator_hc(lon_l[tr], lat_l[tr])
      line_hc = interpolator_h(lon_l[tr], lat_l[tr])
      line_h = (line_hc + line_h) / 2

      u_m = line_u * dist[tr] # m3/s
      v_m = line_v * dist[tr]

      # calculate transport across tracks

      across_vel, along_vel = align(u_m, v_m, angle[tr])
      across_vf = (across_vel * ((s_ref - line_s) / s_ref)) / line_h

      across_trans[c, tr] = np.ma.sum(across_vel)
      across_fw[c, tr] = np.ma.sum(across_vf)

      # units m3/s

      if tr == -1:
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2)
        ax3 = plt.subplot(3, 1, 3)
        ax1.plot(np.sum(u.T, axis=0))
        ax1.plot(np.sum(v.T, axis=0))
        ax2.plot(np.sum(across_vel, axis=0))
        ax2.plot(np.sum(along_vel, axis=0))
        ax3.plot(np.sum(across_vel * ((s_ref - sal_gt) / s_ref), axis=0))
        ax3.plot(np.sum(along_vel * ((s_ref - sal_gt) / s_ref), axis=0))
        #ax3.plot(np.mean((35 - sal_gt) / 35, axis=0))
        #ax3.plot(np.mean((35 - sal_gt) / 35, axis=0))
        plt.savefig('./Figures/fw_test1.png')
        sys.exit()

    print('angle', across_trans[c, :])
    print(np.sum(across_trans[c, :]))
    print('angle', across_fw[c, :])
    print(np.sum(across_fw[c, :]))
    print()

    date_list[c] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-7].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S')
    c = c + 1

  if 0 & (i == 18): # 6 months
    ax = [None] * 6
    ax[0] = plt.subplot(3, 2, 1)
    ax[1] = plt.subplot(3, 2, 2)
    ax[2] = plt.subplot(3, 2, 3)
    ax[3] = plt.subplot(3, 2, 4)
    ax[4] = plt.subplot(3, 2, 5)
    ax[5] = plt.subplot(3, 2, 6)
    for tr in range(len(tracks)):
      ax[tr].plot(date_list[:c-1], across_fw[:c-1, tr])
      #ax[tr].plot(date_list[:i], across_trans[:i, tr])
      #ax[tr].plot(date_list[:i], along_trans[:i, tr])
    plt.savefig('./Figures/fw_test.png')
    sys.exit()

  print(date_list[c-1])
  print('angle_m', np.ma.mean(np.ma.sum(across_trans[:c-1, :], axis=1)))
  print('fw_m',np.ma.mean(np.ma.sum(across_fw[:c-1, :], axis=1)))


date_list = date_list[0:c]
across_trans = across_trans[0:c, :]
across_fw = across_fw[0:c, :]



np.savez(out_dir + 'fw_budget_node.npz', date_list=date_list, across_trans=across_trans, across_fw=across_fw)


