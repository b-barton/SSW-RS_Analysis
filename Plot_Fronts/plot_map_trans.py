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
from PyFVCOM.grid import elems2nodes
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors
import matplotlib.patheffects as PathEffects

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



in_dir = '/scratch/benbar/Processed_Data_V3.02/'
#mjas = '/scratch/benbar/JASMIN/Model_Output/Daily/'
#fn = sorted(glob.glob(mjas + '*/SSWRS*V1.1*dy*' + 'RE.nc'))

#fname = '/scratch/Scottish_waters_FVCOM/SSW_RS/'
#folder1 = 'SSW_Reanalysis_v1.1_1995_02_18'
datafile = (in_dir 
      + '/SSWRS_V3.02_NOC_FVCOM_NWEuropeanShelf_01dy_19930101-1200_RE.nc')

#fn = []
#for yr in range(2000, 2005):
#  fn.extend(sorted(glob.glob(mjas + '*/SSWRS*V1.1*dy*' + str(yr) + '*RE.nc')))



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
FVCOM = readFVCOM(datafile, vars, dims=dims)
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


# Load rivers and boundaries

if 1:

  fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
  fbound = fin + 'Nesting/SSW_Hindcast_nest_forcing_1995.nc'
  friver = fin + 'SSW_Hindcast_riv_xe3.nml'

  first_river = '../../Check_Rivers/Climatology_swona_run_river.nml'

  t1 = 0

  # Extract only the first 24 time steps.
  dims = {'time':[0, -1]}
  #dims = {'time':range(9)}
  # List of the variables to extract.
  vars = ['lon', 'lat', 'nv', 'zeta', 'h', 'Times']
  FVCOM = readFVCOM(datafile, vars, dims=dims)

  nc_fid = nc.Dataset(fbound)
  b_lon = nc_fid.variables['lon'][:]
  b_lat = nc_fid.variables['lat'][:]
  nc_fid.close()
  b_lon[b_lon > 180] = b_lon[b_lon > 180] -360

  # Read namelist to get river nodes
  with open(friver, 'r') as fid:
    lines = fid.readlines()

  r_node = np.zeros(len(lines)//7, dtype=int)
  st = 0
  for j in range(0, len(lines), 7):
    r_node[j//7] = int(lines[j + 3].strip().strip(',').split('=')[1])

  rlon = np.zeros((len(r_node)))
  rlat = np.zeros((len(r_node)))

  for i in range(len(r_node)):
    ind = np.nonzero(r_node[i] == fvg.grid.nodes)[0]
    rlon[i] = fvg.grid.lon[ind]
    rlat[i] = fvg.grid.lat[ind] 

  # Read first namelist to get old river nodes
  with open(first_river, 'r') as fid:
    lines = fid.readlines()

  r1_node = np.zeros(577, dtype=int)
  st = 0
  c = 0
  for j in range(0, len(lines)):
    if 'RIVER_GRID_LOCATION' in lines[j]:
      r1_node[c] = int(lines[j].strip().strip(',').split('=')[1])
      c = c + 1

  r1lon = np.zeros((len(r1_node)))
  r1lat = np.zeros((len(r1_node)))

  for i in range(len(r1_node)):
    ind = np.nonzero(r1_node[i] == fvg.grid.nodes)[0]
    r1lon[i] = fvg.grid.lon[ind]
    r1lat[i] = fvg.grid.lat[ind] 


data = np.load(in_dir + 'stack_uv_mn1.npz', allow_pickle=True)
date = data['date_list'][:-12]
print(data['date_list'][0], data['date_list'][-1])
ua = data['ua'][:-12, :]
va = data['va'][:-12, :] # time, space
lonc = data['lonc']
latc = data['latc']
triangles = data['tri']

data.close()

data = np.load(in_dir + 'stack_uv_mn2.npz', allow_pickle=True)
date2 = data['date_list']
ua2 = data['ua']
va2 = data['va'] # time, space

data.close()

print(date2[0], date2[-1])
date = np.append(date, date2)
ua = np.append(ua, ua2, axis=0)
va = np.append(va, va2, axis=0) # time, space
ua2 = None
va2 = None

u_mean = np.ma.mean(ua, axis=0)
v_mean = np.ma.mean(va, axis=0)

mag_r = (u_mean ** 2 + v_mean ** 2) ** 0.5
dir_r = np.ma.arctan2(v_mean, u_mean) # y, x
ua = None
va = None

mag_r = np.ma.masked_where(mag_r == -999, mag_r)
dir_r = np.ma.masked_where(dir_r == -999, dir_r)
mag_n = elems2nodes(mag_r, triangles)
u_f = elems2nodes(mag_r * np.ma.cos(dir_r), triangles)
v_f = elems2nodes(mag_r * np.ma.sin(dir_r), triangles)


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

# get nodes for 
elem_p = []
for i in range(len(xc)):
  elem_p.append(Point(xc[i], yc[i]))

line_mask = np.zeros((len(xc)), dtype=bool)
for j in range(len(xc)):
  line_mask[j] = line_mask[j] | poly1_utm.contains(elem_p[j])
  line_mask[j] = line_mask[j] | poly2_utm.contains(elem_p[j])
  line_mask[j] = line_mask[j] | poly3_utm.contains(elem_p[j])
  line_mask[j] = line_mask[j] | poly4_utm.contains(elem_p[j])
  #line_mask[j] = line_mask[j] | poly5_utm.contains(node_p[j])
  line_mask[j] = line_mask[j] | poly6_utm.contains(elem_p[j])


# Use transects as lines to define surrounding area polygons

l0_x = np.array([-10, -2, -3, -4, -4.7])
l0_y = np.array([54, 54, 55.5, 55.5, 57.2])

l1b_x = np.array([-4.7])
l1b_y = np.array([57.2])

l2b_x = np.array([-1, 0])
l2b_y = np.array([60.9, 60.9])

l3b_x = np.array([-4.7, -4.9])
l3b_y = np.array([57.2, 58.4])

l4b_x = np.array([-4.7, -4, -3, -2, 2])
l4b_y = np.array([57.2, 55.5, 55.5, 54, 54])

# Append lines to make polygons

a1_x = np.concatenate((l0_x, l1_x, l0_x[0:1]))
a1_y = np.concatenate((l0_y, l1_y, l0_y[0:1]))

a2_x = np.concatenate((l1_x, l2_x[::-1], l1b_x, l1_x[0:1]))
a2_y = np.concatenate((l1_y, l2_y[::-1], l1b_y, l1_y[0:1]))

a3_x = np.concatenate((l2_x, l2b_x, l3_x[::-1], l2_x[0:1]))
a3_y = np.concatenate((l2_y, l2b_y, l3_y[::-1], l2_y[0:1]))

a4_x = np.concatenate((l3_x, l4_x[::-1], l3b_x, l3_x[0:1]))
a4_y = np.concatenate((l3_y, l4_y[::-1], l3b_y, l3_y[0:1]))

a5_x = np.concatenate((l4_x[::-1], l4b_x, l4_x[-1:]))
a5_y = np.concatenate((l4_y[::-1], l4b_y, l4_y[-1:]))

# EOF box

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

# Plot


def plot_streamlines(ax, x, y, lon, lat, m, triangles, u, v):
    lon_g, lat_g, xg, yg = m.makegrid(100, 100, returnxy=True)
    print(np.min(lon_g), np.max(lon_g), np.min(lat_g), np.max(lat_g))
    trio = tri.Triangulation(lon, lat, triangles=np.asarray(triangles))
    interpolator_u = tri.LinearTriInterpolator(trio, u)
    interpolator_v = tri.LinearTriInterpolator(trio, v)

    grid_u = interpolator_u(lon_g, lat_g)
    grid_v = interpolator_v(lon_g, lat_g)
    speed = (grid_u ** 2 + grid_v ** 2) ** 0.5
    lw = 8 * speed / speed.max()

    ax.streamplot(xg, yg, grid_u, grid_v, density=(1, 2), color='k', linewidth=lw, zorder=102)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

my_cmap = truncate_colormap(plt.get_cmap('GnBu'), 0.2, 1)

if 1:
  fig1 = plt.figure(figsize=(12, 6))  # size in inches
  ax1 = fig1.add_axes([0.05, 0.1, 0.35, 0.8])
  cax1 = fig1.add_axes([0.41, 0.3, 0.02, 0.4])

  ax2 = fig1.add_axes([0.55, 0.1, 0.35, 0.8])
  cax2 = fig1.add_axes([0.91, 0.3, 0.02, 0.4])

  # model setup

  extra = 0.5
  extents = np.array((lon.min() - extra,
                    lon.max() + extra,
                    lat.min() - extra,
                    lat.max() + extra))

  m = Basemap(llcrnrlon=extents[:2].min(),
        llcrnrlat=extents[-2:].min(),
        urcrnrlon=extents[:2].max(),
        urcrnrlat=extents[-2:].max(),
        rsphere=(6378137.00, 6356752.3142),
        resolution='h',
        projection='merc',
        lat_0=extents[-2:].mean(),
        lon_0=extents[:2].mean(),
        lat_ts=extents[-2:].mean(),
        ax=ax1)  
  parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), 5)  
  meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), 5) 
  #m.drawmapboundary()
  #m.drawcoastlines(zorder=100)
  #m.fillcontinents(color='0.6', zorder=100)
  m.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)
  m.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)
  x1, y1 = m(lon, lat)
  x2, y2 = m(b_lon, b_lat)
  x3, y3 = m(rlon, rlat)
  x4, y4 = m(r1lon, r1lat)

  CS1 = ax1.tripcolor(x1, y1, triangles, FVCOM['h'], cmap=my_cmap, vmin=0, vmax=300)

  ax1.plot(x4, y4, '.k', ms=6, zorder=99)
  ax1.plot(x4, y4, '.b', ms=4, label='SSM G2G Rivers', zorder=101)
  ax1.plot(x3, y3, '.k', ms=6, zorder=99)
  ax1.plot(x3, y3, '.g', ms=4, label='E-Hype Rivers', zorder=100)

  ax1.plot(x2[:492], y2[:492], '.k', ms=6, zorder=99)
  ax1.plot(x2[:492], y2[:492], '.', ms=4, color='tab:cyan', label='Atlantic Boundary', zorder=101)
  ax1.plot(x2[492:], y2[492:], '.k', ms=6, zorder=99)
  ax1.plot(x2[492:], y2[492:], '.', ms=4, color='tab:olive', label='Baltic Boundary', zorder=101)

  for i in range(len(tracks)):
    m_tracks_x, m_tracks_y  = m(tracks[i][:, 0], tracks[i][:, 1])
    ax1.plot(m_tracks_x, m_tracks_y, '-k')


  cb1 = fig1.colorbar(CS1, cax1, extend='max')
  cax1.set_ylabel('Depth (m)')
  l = ax1.legend(loc='lower right')
  l.set_zorder(110)

# transects
else:
  fig1 = plt.figure(figsize=(6, 7))  # size in inches
  ax2 = fig1.add_axes([0.1, 0.1, 0.75, 0.8])
  cax2 = fig1.add_axes([0.86, 0.3, 0.02, 0.4])

extents = np.array((-10, 2, 54, 61))

m1 = Basemap(llcrnrlon=extents[:2].min(),
        llcrnrlat=extents[-2:].min(),
        urcrnrlon=extents[:2].max(),
        urcrnrlat=extents[-2:].max(),
        rsphere=(6378137.00, 6356752.3142),
        resolution='h',
        projection='merc',
        lat_0=extents[-2:].mean(),
        lon_0=extents[:2].mean(),
        lat_ts=extents[-2:].mean(),
        ax=ax2)  

parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), 5)  
meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), 5) 
m1.drawmapboundary()
#m1.drawcoastlines(zorder=100)
#m.fillcontinents(color='0.6', zorder=100)
m1.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=10, linewidth=0)
m1.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=10, linewidth=0)

x1, y1 = m1(lon, lat)
cs1 = ax2.tripcolor(x1, y1, triangles, FVCOM['h'], cmap=my_cmap, vmin=0, vmax=300, zorder=98)


mx, my = m1(lon, lat)
triangles = nv.transpose() -1

plot_streamlines(ax2, x, y, lon, lat, m1, triangles, u_f, v_f)

cs1 = ax2.tripcolor(mx, my, triangles, dist, mask=np.invert(line_mask), zorder=103)

m1_x, m1_y = m1(px1, py1)
m2_x, m2_y = m1(px2, py2)
m3_x, m3_y = m1(px3, py3)
m4_x, m4_y = m1(px4, py4)
m5_x, m5_y = m1(px5, py5)
m6_x, m6_y = m1(px6, py6)

#m1_x, m1_y = m1(l1_x, l1_y)
#m2_x, m2_y = m1(l2_x, l2_y)
#m3_x, m3_y = m1(l3_x, l3_y)
#m4_x, m4_y = m1(l4_x, l4_y)
#m5_x, m5_y = m1(l5_x, l5_y)
#m6_x, m6_y = m1(l6_x, l6_y)

#ax1.plot(l1_x, l1_y, '-', zorder=101)
ax2.plot(m1_x, m1_y, '-k', zorder=103)
#ax1.plot(m1(l2_x, l2_y), '-', zorder=101)
ax2.plot(m2_x, m2_y, '-k', zorder=103)
#ax1.plot(m1(l3_x, l3_y), '-', zorder=101)
ax2.plot(m3_x, m3_y, '-k', zorder=103)
#ax1.plot(m1(l4_x, l4_y), '-', zorder=101)
ax2.plot(m4_x, m4_y, '-k', zorder=103)
#ax1.plot(m1(l5_x, l5_y), '-', zorder=101)
#ax1.plot(m5_x, m5_y, '-k', zorder=101)
#ax1.plot(m1(l6_x, l6_y), '-', zorder=101)
ax2.plot(m6_x, m6_y, '-k', zorder=103)

ax2.plot(m1_x, m1_y, '-w', lw=3, zorder=102)
ax2.plot(m2_x, m2_y, '-w', lw=3, zorder=102)
ax2.plot(m3_x, m3_y, '-w', lw=3, zorder=102)
ax2.plot(m4_x, m4_y, '-w', lw=3, zorder=102)
#ax2.plot(m5_x, m5_y, '-k', lw=2, zorder=101)
ax2.plot(m6_x, m6_y, '-w', lw=3, zorder=102)


m1_x, m1_y = m1(a1_x, a1_y)
m2_x, m2_y = m1(a2_x, a2_y)
m3_x, m3_y = m1(a3_x, a3_y)
m4_x, m4_y = m1(a4_x, a4_y)
m5_x, m5_y = m1(a5_x, a5_y)

#ax1.plot(m1_x, m1_y, '-', color='tab:orange', zorder=100)
#ax1.plot(m2_x, m2_y, '-', color='tab:orange', zorder=100)
#ax1.plot(m3_x, m3_y, '-', color='tab:orange', zorder=100)
#ax1.plot(m4_x, m4_y, '-', color='tab:orange', zorder=100)
#ax1.plot(m5_x, m5_y, '-', color='tab:orange', zorder=100)



ax2.annotate('t1', m1(-9, 55.8), xycoords='data', fontsize=10, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('t2', m1(-5.2, 60), xycoords='data', fontsize=10, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('t3', m1(0, 58.8), xycoords='data', fontsize=10, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('t4', m1(0, 56.8), xycoords='data', fontsize=10, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('JONSIS', m1(-1, 59.5), xycoords='data', fontsize=10, bbox=dict(boxstyle="round", fc="w"), zorder=105)

fig1.colorbar(cs1, cax=cax2)
cax2.set_ylabel('Distance Offshore (km)')

pe = [PathEffects.withStroke(linewidth=2.5, foreground="w", alpha=1)]
pe1 = [PathEffects.withStroke(linewidth=2.5, foreground="k", alpha=1)]

ax2.annotate('Slope Current', xy=(0.22, 0.75), 
          xycoords='axes fraction', rotation = 40,
          size=10, va='center', ha='center', color='w', 
          path_effects=pe1, zorder=110)
ax2.annotate('Atlantic\nInflow Current', xy=(0.15, 0.55), 
          xycoords='axes fraction', rotation = 60,
          size=10, va='center', ha='center', 
          path_effects=pe, zorder=110)
ax2.annotate('Scottish\nCoastal\nCurrent', xy=(0.38, 0.63), xytext=(0.42, 0.48),
          xycoords='axes fraction',
          size=10, va='center', ha='center', zorder=110, 
          arrowprops=dict(facecolor='w', edgecolor='k', arrowstyle='fancy', 
          connectionstyle='arc3,rad=-0.3'))
ax2.annotate('East\nShetland\nCurrent', xy=(0.85, 0.9), 
          xycoords='axes fraction', rotation = 60,
          size=10, va='center', ha='center', color='k', 
          path_effects=pe, zorder=110)
ax2.annotate('Fair Isle\nCurrent', xy=(0.6, 0.8), 
          xycoords='axes fraction', rotation = -20,
          size=10, va='center', ha='center', color='k', 
          path_effects=pe, zorder=110)
ax2.annotate('Dooley\nCurrent', xy=(0.9, 0.52), xytext=(0.58, 0.44), 
          xycoords='axes fraction',
          size=10, va='center', ha='center', zorder=110, 
          arrowprops=dict(facecolor='w', edgecolor='k', arrowstyle='fancy', 
          connectionstyle='arc3,rad=-0.3'))


ax2.annotate('Orkney', xy=(0.58, 0.70), 
          xycoords='axes fraction', rotation = 60,
          size=8, va='center', ha='center', color='k', 
          path_effects=pe, zorder=110)
ax2.annotate('Shetland', xy=(0.72, 0.9), 
          xycoords='axes fraction', rotation = 60,
          size=8, va='center', ha='center', color='k', 
          path_effects=pe, zorder=110)
ax2.annotate('Outer Hebrides', xy=(0.25, 0.52), 
          xycoords='axes fraction', rotation = 60,
          size=8, va='center', ha='center', 
          path_effects=pe, zorder=110)


#ax2.annotate('', xy=(0.4, 0.8), xytext=(0.3, 0.7), 
#          xycoords='axes fraction',
#          size=12, va='center', ha='center', zorder=110, 
#          arrowprops=dict(facecolor='w', edgecolor='k', arrowstyle='fancy', 
#          connectionstyle='arc3,rad=-0.3'))


ax1.annotate('(a)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)


fig1.savefig('./Figures/transect_map.png')


