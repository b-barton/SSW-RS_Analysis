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
for yr in range(1993, 2020):
  fn.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V3.02*dy*RE.nc')))
#fn = sorted(glob.glob(mjas + '*/SSWRS*V1.1*dy*RE.nc'))

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

data = np.load(out_dir + 'sswrs_xy.npz')
x = data['x']
y = data['y']
xc = data['xc']
yc = data['yc']
data.close()

trio = tri.Triangulation(x, y, triangles=np.asarray(triangles))

vars = ('zeta', 'temp', 'salinity', 'ua', 'va', 'Times', 'h')

sal = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
temp = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
date_list = np.zeros((len(fn) * 10), dtype=object) - 999
c = 0

for i in range(len(fn)):
  print(i / len(fn) *100, '%')
  FVCOM = readFVCOM(fn[i], vars, dims=dims)
  for t1 in range(FVCOM['ua'].shape[0]):

    temp[c, :] = FVCOM['temp'][t1, -1, :]
    sal[c, :] = FVCOM['salinity'][t1, -1, :]

    date_list[c] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-7].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S')
    c = c + 1

date_list = date_list[0:c]
temp = temp[0:c, :]
sal = sal[0:c, :]

# monthly means

yr = np.array([d.year for d in date_list])
mnt = np.array([d.month for d in date_list])
yri = np.unique(yr)
mnti =np.unique(mnt)

t_mn = np.ma.zeros((len(yri) * len(mnti), temp.shape[1]))
s_mn = np.ma.zeros((len(yri) * len(mnti), sal.shape[1]))
date_mn = np.zeros((len(yri) * len(mnti)), dtype=object)
count = 0
for i in range(len(yri)):
  for j in range(len(mnti)):
    ind = (yr == yri[i]) & (mnt == mnti[j])
    t_mn[count] = np.ma.mean(temp[ind, :], axis=0)
    s_mn[count] = np.ma.mean(sal[ind, :], axis=0)
    date_mn[count] = dt.datetime(yri[i], mnti[j], 1)
    count += 1


np.savez(out_dir + 'stack_ts_bot_mn.npz', temp=t_mn, sal=s_mn, date_list=date_mn, lon=lon, lat=lat, tri=triangles)


