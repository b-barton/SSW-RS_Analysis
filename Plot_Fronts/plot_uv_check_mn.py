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
import scipy.stats as stats
import scipy.signal as sig
import df_regress
from plot_uv_bivariate_mn import calc_bivar_uv


out_dir = '/scratch/benbar/Processed_Data_V3.02/'
fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'


data = np.load(out_dir + 'stack_baroclinic_r_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bcr_ua = data['baroclin_u'][:, :]
bcr_va = data['baroclin_v'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_baroclinic_a_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bca_ua = data['baroclin_u_a'][:, :]
bca_va = data['baroclin_v_a'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_baroclinic_b_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bcb_ua = data['baroclin_u_b'][:, :]
bcb_va = data['baroclin_v_b'][:, :] # time, space
data.close()


data = np.load(out_dir + 'stack_barosteric_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bsr_ua = data['barosteric_u'][:, :]
bsr_va = data['barosteric_v'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_barosteric_thermo_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bsa_ua = data['barosteric_u_a'][:, :]
bsa_va = data['barosteric_v_a'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_barosteric_haline_vel_mn.npz', allow_pickle=True)
date_bc = data['date_list'][:]
print(data['date_list'][0], data['date_list'][-1])
bsb_ua = data['barosteric_u_b'][:, :]
bsb_va = data['barosteric_v_b'][:, :] # time, space
data.close()

data = np.load(out_dir + 'stack_uv_mn1.npz', allow_pickle=True)
date_uv = data['date_list'][:-12]
lonc = data['lonc']
latc = data['latc']
triangles = data['tri']
data.close()

data = np.load(out_dir + 'sswrs_xy.npz')
x = data['x']
y = data['y']
xc = data['xc']
yc = data['yc']
data.close()

data = np.load(out_dir + 'coast_distance.npz', allow_pickle=True)
lon = data['lon']
lat = data['lat']
data.close()

datafile = (out_dir 
      + '/SSWRS_V3.02_NOC_FVCOM_NWEuropeanShelf_01dy_19930101-1200_RE.nc')
dims = {'time':':10'}
vars = ('nv', 'h')
FVCOM = readFVCOM(datafile, vars, dims=dims)
h = FVCOM['h']


if 0:
  bcr_ua_c = fvcom.grid.nodes2elems(bcr_ua, triangles)
  bcr_va_c = fvcom.grid.nodes2elems(bcr_va, triangles)
  bca_ua_c = fvcom.grid.nodes2elems(bca_ua, triangles)
  bca_va_c = fvcom.grid.nodes2elems(bca_va, triangles)
  bcb_ua_c = fvcom.grid.nodes2elems(bcb_ua, triangles)
  bcb_va_c = fvcom.grid.nodes2elems(bcb_va, triangles)

  bsr_ua_c = fvcom.grid.nodes2elems(bsr_ua, triangles)
  bsr_va_c = fvcom.grid.nodes2elems(bsr_va, triangles)
  bsa_ua_c = fvcom.grid.nodes2elems(bsa_ua, triangles)
  bsa_va_c = fvcom.grid.nodes2elems(bsa_va, triangles)
  bsb_ua_c = fvcom.grid.nodes2elems(bsb_ua, triangles)
  bsb_va_c = fvcom.grid.nodes2elems(bsb_va, triangles)

# scale up the thermo and halosteric heights

bsa_ua = bsa_ua * np.ma.mean((bsr_ua / (bsa_ua + bsb_ua)))
bsa_va = bsa_va * np.ma.mean((bsr_va / (bsa_va + bsb_va)))
bsb_ua = bsb_ua * np.ma.mean((bsr_ua / (bsa_ua + bsb_ua)))
bsb_va = bsb_va * np.ma.mean((bsr_va / (bsa_va + bsb_va)))

bca_ua = bca_ua * np.ma.mean((bcr_ua / (bca_ua + bcb_ua)))
bca_va = bca_va * np.ma.mean((bcr_va / (bca_va + bcb_va)))
bcb_ua = bcb_ua * np.ma.mean((bcr_ua / (bca_ua + bcb_ua)))
bcb_va = bcb_va * np.ma.mean((bcr_va / (bca_va + bcb_va)))

# mask bad data at 120 m

h = np.tile(h, (bsr_ua.shape[0], 1))
print(h.shape, bsr_ua.shape)
print(np.mean(h))
h1 = 115
h2 = 222
bsr_ua[(h >= h1) & (h < h2)] = 0
bsr_va[(h >= h1) & (h < h2)] = 0
bsa_ua[(h >= h1) & (h < h2)] = 0
bsa_va[(h >= h1) & (h < h2)] = 0
bsb_ua[(h >= h1) & (h < h2)] = 0
bsb_va[(h >= h1) & (h < h2)] = 0


bct_ua = bca_ua + bcb_ua
bct_va = bca_va + bcb_va

bst_ua = bsa_ua + bsb_ua
bst_va = bsa_va + bsb_va

loc = np.nonzero((lonc > -4.1) & (lonc < -4) & (latc > 58.7) & (latc < 58.8))[0][0]
print(loc)

fig, axs = plt.subplots(4)
if 0:
  axs[0].plot(date_ek, ek_ua[:, loc])
  axs[0].plot(date_bc, bc_ua_c[:, loc])
  axs[0].plot(date_uv, ua[:, loc])
  axs[1].plot(date_ek, ek_va[:, loc])
  axs[1].plot(date_bc, bc_va_c[:, loc])
  axs[1].plot(date_uv, va[:, loc])
  axs[2].plot(date_uv, ls_ua[:, loc])
  axs[2].plot(date_bt, -bt_ua[:, loc])
  axs[3].plot(date_uv, ls_va[:, loc])
  axs[3].plot(date_bt, -bt_va[:, loc])

fig.savefig('./Figures/uv_check_part.png')


def plot_streamlines(ax, fig, cax, x, y, lon, lat, triangles, u, v, lw):
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
          ax=ax)  

  parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), 5)  
  meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), 5) 
  m1.drawmapboundary()
  #m1.drawcoastlines(zorder=100)
  #m.fillcontinents(color='0.6', zorder=100)
  m1.drawparallels(parallels, labels=[1, 0, 0, 0],
                  fontsize=10, linewidth=0)
  m1.drawmeridians(meridians, labels=[0, 0, 0, 1],
                  fontsize=10, linewidth=0)

  mx, my = m1(lon, lat)

  u_m = np.ma.mean(u, axis=0)
  v_m = np.ma.mean(v, axis=0)

  mag = (u_m ** 2 + v_m ** 2) ** 0.5


  cs1 = ax.tripcolor(mx, my, triangles, mag, vmin=0, vmax=0.04, cmap=plt.get_cmap('Reds'), zorder=99)
  fig.colorbar(cs1, cax=cax)
  cax.set_ylabel('vel')

  lon_g, lat_g, xg, yg = m1.makegrid(100, 100, returnxy=True)
  print(np.min(lon_g), np.max(lon_g), np.min(lat_g), np.max(lat_g))
  trio = tri.Triangulation(lon, lat, triangles=np.asarray(triangles))
  interpolator_u = tri.LinearTriInterpolator(trio, u_m)
  interpolator_v = tri.LinearTriInterpolator(trio, v_m)

  grid_u = interpolator_u(lon_g, lat_g)
  grid_v = interpolator_v(lon_g, lat_g)
  grid_lw = (grid_u ** 2 + grid_v**2) ** 0.5

  ax.streamplot(xg, yg, grid_u, grid_v, density=(2, 4), color='k', linewidth=grid_lw, arrowsize=1.5, zorder=102)



fig1 = plt.figure(figsize=(10, 14))  # size in inches
ax1 = fig1.add_axes([0.05, 0.76, 0.35, 0.2])
ax2 = fig1.add_axes([0.525, 0.76, 0.35, 0.2])
ax3 = fig1.add_axes([0.05, 0.52, 0.35, 0.2])
ax4 = fig1.add_axes([0.525, 0.52, 0.35, 0.2])
ax5 = fig1.add_axes([0.05, 0.28, 0.35, 0.2])
ax6 = fig1.add_axes([0.525, 0.28, 0.35, 0.2])
ax7 = fig1.add_axes([0.05, 0.04, 0.35, 0.2])
ax8 = fig1.add_axes([0.525, 0.04, 0.35, 0.2])
cax1 = fig1.add_axes([0.88, 0.14, 0.01, 0.48])

plot_streamlines(ax1, fig1, cax1, x, y, lon, lat, triangles, bcr_ua, bcr_va, 1)
plot_streamlines(ax2, fig1, cax1, x, y, lon, lat, triangles, bct_ua, bct_va, 1)
plot_streamlines(ax3, fig1, cax1, x, y, lon, lat, triangles, bca_ua, bca_va, 1)
plot_streamlines(ax4, fig1, cax1, x, y, lon, lat, triangles, bcb_ua, bcb_va, 1)

plot_streamlines(ax5, fig1, cax1, x, y, lon, lat, triangles, bsr_ua, bsr_va, 1)
plot_streamlines(ax6, fig1, cax1, x, y, lon, lat, triangles, bst_ua, bst_va, 1)
plot_streamlines(ax7, fig1, cax1, x, y, lon, lat, triangles, bsa_ua, bsa_va, 1)
plot_streamlines(ax8, fig1, cax1, x, y, lon, lat, triangles, bsb_ua, bsb_va, 1)


fig1.savefig('./Figures/vel_check_mn.png')

