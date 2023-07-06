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

if 0:
  x_min = -10.05
  x_max = 2.05

  y_min = 53.98
  y_max = 61.02

  x_mid1 = -2.8
  x_mid2 = -6.5
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

ind_c = [None] * len(tracks)
dist_c = [None] * len(tracks)
elem_pair = [None] * len(tracks)
node_pair = [None] * len(tracks)
angle = [None] * len(tracks)

for tr in range(len(tracks)):
  #ind, dist = fvg.grid.horizontal_transect_nodes(positions)
  # Extract node IDs along a line defined by 
  # `positions' [[x1, y1], [x2, y2], ..., [xn, yn]]
  ind_c[tr], dist_c[tr] = element_sample(lonc, latc, tracks[tr], nbe)
  #dist_c[tr] = np.cumsum([0] + [fvcom.grid.haversine_distance(
  #            (lonc[i], latc[i]), 
  #            (lonc[i + 1], latc[i + 1])) for i in ind_c[tr][:-1]])
  dist_c[tr] = np.tile(dist_c[tr], (siglay_mod.shape[0], 1))
  dxl = tracks[tr][1, 0] - tracks[tr][0, 0]
  dyl = tracks[tr][1, 1] - tracks[tr][0, 1]
  xt = xc[ind_c[tr]]
  yt = yc[ind_c[tr]]
  dx = xt[1:] - xt[:-1]
  dy = yt[1:] - yt[:-1]
  #dist_tmp = (dx ** 2 + dy ** 2) ** 0.5
  #dist_c[tr] = np.tile(dist_tmp, (siglay_mod.shape[0], 1))
  angle1 = np.arctan2(dx, dy) # from North
  angle[tr] = get_angle(lonc[ind_c[tr]], latc[ind_c[tr]], wgs84)
  print(stats.circmean(angle[tr] % (2 * np.pi)) * 180/np.pi, np.arctan2(dxl, dyl) * 180/np.pi, stats.circmean(angle1 % (2 * np.pi)) * 180/np.pi)

  # get node pairs attached to element pairs

  elem_pair[tr] = np.array([ind_c[tr][:-1], ind_c[tr][1:]])
  node_pair[tr] = np.zeros(elem_pair[tr].shape, dtype=int)
  for i in range(elem_pair[tr].shape[1]):
    n1 = triangles[elem_pair[tr][0, i]]
    n2 = triangles[elem_pair[tr][1, i]]
    index = np.where(np.in1d(n1, n2))[0]
    node_pair[tr][:, i] = n1[index]
  print(elem_pair[tr].shape, node_pair[tr].shape)

  print(ind_c[tr].shape, dist_c[tr].shape)
  plt.scatter(lonc[ind_c[tr][1:]], latc[ind_c[tr][1:]], c=dist_c[tr][0, :], vmin=0, vmax=10000)
  plt.plot(lonc[ind_c[tr][0]], latc[ind_c[tr][0]], 'ro', ms=5)

plt.savefig('./Figures/tracks.png' )
print('Plotted')



def align(x_var, y_var, angle):
  # Align components to be along and across the transect
  # angle is clockwise from North

  across_var = np.zeros_like(x_var)
  along_var = np.zeros_like(x_var)

  for i in range(x_var.shape[1]): 
    var_mag = ((x_var[:, i] ** 2) + (y_var[:, i] ** 2)) ** 0.5
    var_dir = np.arctan2(x_var[:, i], y_var[:, i])
    new_dir = var_dir - angle[i] # subtract so North lies along transect
    across_var[:, i] = var_mag * np.sin(new_dir) # x direction
    along_var[:, i] = var_mag * np.cos(new_dir) # y direction
  return across_var, along_var

vars = ('zeta', 'salinity', 'u', 'v', 'Times', 'h')
dims = {'time':':'}

test = np.ma.zeros((len(fn) * 10, len(tracks))) - 999
across_trans = np.ma.zeros((len(fn) * 10, len(tracks))) - 999
along_trans = np.ma.zeros((len(fn) * 10, len(tracks))) - 999
across_fw = np.ma.zeros((len(fn) * 10, len(tracks))) - 999
along_fw = np.ma.zeros((len(fn) * 10, len(tracks))) - 999
date_list = np.zeros((len(fn) * 10), dtype=object) - 999
c = 0
s_ref = 36

for i in range(len(fn)):
  print(i / len(fn) *100, '%')
  FVCOM = readFVCOM(fn[i], vars, dims=dims)
  for t1 in range(FVCOM['u'].shape[0]):

    zeta = FVCOM['zeta'][t1, :]
    sal = FVCOM['salinity'][t1, :, :] # time, depth, node

    # convert to elements

    zeta_c = fvcom.grid.nodes2elems(zeta, triangles)
    sal_c = fvcom.grid.nodes2elems(sal, triangles)

    # calculate depth along tracks

    Hc = hc + zeta_c
    depth_mod = -Hc * siglay_mod # should be negative (siglay is negative)
    depthlev_mod = -Hc * siglev_mod # should be negative (siglev is negative)
    d_depth = (depthlev_mod[1:, :] - depthlev_mod[:-1, :])

    for tr in range(len(tracks)):
      #depth_track = d_depth[:, ind_c[tr]]
      depth_track = (d_depth[:, elem_pair[tr][0, :]]
           + d_depth[:, elem_pair[tr][1, :]]) / 2

      # calculate transport across tracks

      u = (FVCOM['u'][t1, :, elem_pair[tr][0, :]]
           + FVCOM['u'][t1, :, elem_pair[tr][1, :]]) / 2
      v = (FVCOM['v'][t1, :, elem_pair[tr][0, :]]
           + FVCOM['v'][t1, :, elem_pair[tr][1, :]]) / 2

      # try constant
      #u = np.zeros(u.shape) - 0.5
      #v = np.zeros(v.shape) + 0.5

      sal_gt = (sal[:, node_pair[tr][0, :]]
           + sal[:, node_pair[tr][1, :]]) / 2

      across_vel, along_vel = align(u.T, v.T, angle[tr])

      transp = across_vel * dist_c[tr] * depth_track # m/s * m * m

      across_fw[c, tr] = np.ma.sum(transp * ((s_ref - sal_gt) / s_ref))
      #along_fw[c, tr] = np.ma.sum(along_vel * ((s_ref - sal_gt) / s_ref))

      across_trans[c, tr] = np.ma.sum(transp)
      #along_trans[c, tr] = np.ma.sum(along_vel)

      # units m3/s
      if tr == 0:
        test[c, tr] = np.sum(v.T * dist_c[tr] * depth_track)
      elif tr == 1:
        test[c, tr] = np.sum(u.T * dist_c[tr] * depth_track)
      elif tr == 2:
        test[c, tr] = np.sum(((v.T * dist_c[tr] * depth_track * -1) + u.T * dist_c[tr] * depth_track) / 2)
      elif tr == 3:
        test[c, tr] = np.sum(v.T * dist_c[tr] * depth_track * -1)
      elif tr == 4:
        test[c, tr] = np.sum(u.T * dist_c[tr] * depth_track * -1)
      elif tr == 5:
        test[c, tr] = np.sum(v.T * dist_c[tr] * depth_track)

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

    print('uv', test[c, :])
    print('angle', across_trans[c, :])
    print(np.sum(test[c, :]), np.sum(across_trans[c, :]))
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
      ax[tr].plot(date_list[:c], across_fw[:c, tr])
      #ax[tr].plot(date_list[:i], across_trans[:i, tr])
      #ax[tr].plot(date_list[:i], along_trans[:i, tr])
    plt.savefig('./Figures/fw_test.png')
    print(np.ma.mean(across_fw[:c, :]))
    print(np.ma.mean(across_trans[:c, :]))
    sys.exit()

  #print(np.ma.mean(across_fw[i * 10:(i + 1) * 10, :]))
  #print(np.ma.mean(across_trans[i * 10:(i + 1) * 10, :]))
  print('uv_m', np.ma.mean(np.ma.sum(test[:c, :], axis=1)))
  print('angle_m', np.ma.mean(np.ma.sum(across_trans[:c, :], axis=1)))
  print('fw_m',np.ma.mean(np.ma.sum(across_fw[:c, :], axis=1)))


date_list = date_list[0:c]
across_trans = across_trans[0:c, :]
across_fw = across_fw[0:c, :]



np.savez(out_dir + 'fw_budget.npz', date_list=date_list, across_trans=across_trans, across_fw=across_fw)


