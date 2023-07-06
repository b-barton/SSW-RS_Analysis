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


mjas = '/gws/nopw/j04/ssw_rs/Model_Output_V3.02/Daily/'
fn = []
part1 = 0
if part1:
  for yr in range(1993, 2006):
    fn.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V3.02*dy*RE.nc')))
else:
  for yr in range(2006, 2020):
    fn.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V3.02*dy*RE.nc')))
#fn = sorted(glob.glob(mjas + '*/SSWRS*V1.1*dy*RE.nc'))
print(fn[0])

print(len(fn))
out_dir = '../Processed_Data/'


fin = '../Input_Files/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'

# Extract only the first 24 time steps.
dims = {'time':':10'}
# List of the variables to extract.
vars = ('lon', 'lat', 'latc', 'lonc', 'nv', 'zeta', 'temp', 'salinity', 'ua', 'va', 'siglay', 'h', 'Itime', 'Itime2')
FVCOM = readFVCOM(fn[-1], vars, dims=dims)

# Create the triangulation table array (with Python indexing
# [zero-based])
triangles = FVCOM['nv'].transpose() - 1
# Find the domain extents.
I=np.where(FVCOM['lon'] > 180) # MICDOM: for Scottish shelf domain
FVCOM['lon'][I]=FVCOM['lon'][I]-360 # MICDOM: for Scottish shelf domain
I=np.where(FVCOM['lonc'] > 180) # MICDOM: for Scottish shelf domain
FVCOM['lonc'][I]=FVCOM['lonc'][I]-360 # MICDOM: for Scottish shelf domain
extents = np.array((FVCOM['lon'].min(),
                  FVCOM['lon'].max(),
                   FVCOM['lat'].min(),
                   FVCOM['lat'].max()))

lon = FVCOM['lon']
lat = FVCOM['lat']
lonc = FVCOM['lonc']
latc = FVCOM['latc']
siglay_mod = FVCOM['siglay'][:]
h_mod = FVCOM['h'][:]
depth_mod = -h_mod * siglay_mod # layer, node

st_date = dt.datetime(1992, 12, 30, 0, 0, 0)
en_date = dt.datetime(1993, 1, 31, 23, 0, 0)

fvg = fvcom.preproc.Model(st_date, en_date, grid=fgrd, 
                    native_coordinates='spherical', zone='30N')
x = fvg.grid.x
y = fvg.grid.y
xc = fvg.grid.xc
yc = fvg.grid.yc

trio = tri.Triangulation(x, y, triangles=np.asarray(triangles))

vars = ('u', 'v', 'Times', 'h')
bot = -1

ub = np.ma.zeros((len(fn) * 10, lonc.shape[0])) - 999
vb = np.ma.zeros((len(fn) * 10, lonc.shape[0])) - 999
date_list = np.zeros((len(fn) * 10), dtype=object) - 999
c = 0

for i in range(len(fn)):
  print(i / len(fn) *100, '%')
  FVCOM = readFVCOM(fn[i], vars, dims=dims)
  for t1 in range(FVCOM['u'].shape[0]):

    ub[c, :] = FVCOM['u'][t1, bot, :]
    vb[c, :] = FVCOM['v'][t1, bot, :]

    date_list[c] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-7].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S')
    c = c + 1

date_list = date_list[0:c]
ub = ub[0:c, :]
vb = vb[0:c, :]

# monthly means

yr = np.array([d.year for d in date_list])
mnt = np.array([d.month for d in date_list])
yri = np.unique(yr)
mnti =np.unique(mnt)

ub_mn = np.ma.zeros((len(yri) * len(mnti), ub.shape[1]))
vb_mn = np.ma.zeros((len(yri) * len(mnti), vb.shape[1]))
date_mn = np.zeros((len(yri) * len(mnti)), dtype=object)
count = 0
for i in range(len(yri)):
  for j in range(len(mnti)):
    ind = (yr == yri[i]) & (mnt == mnti[j])
    ub_mn[count] = np.ma.mean(ub[ind, :], axis=0)
    vb_mn[count] = np.ma.mean(vb[ind, :], axis=0)
    date_mn[count] = dt.datetime(yri[i], mnti[j], 1)
    count += 1

if part1:
  np.savez(out_dir + 'stack_bot_uv_mn1.npz', ub=ub_mn, vb=vb_mn, date_list=date_mn, lonc=lonc, latc=latc, tri=triangles)
else:
  np.savez(out_dir + 'stack_bot_uv_mn2.npz', ub=ub_mn, vb=vb_mn, date_list=date_mn, lonc=lonc, latc=latc, tri=triangles)


