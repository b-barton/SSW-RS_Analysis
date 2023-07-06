#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
from PyFVCOM.read import ncread as readFVCOM
from PyFVCOM import physics
from PyFVCOM.read import MFileReader
import datetime as dt
import netCDF4 as nc
import glob
import matplotlib.tri as tri
import PyFVCOM as fvcom
import gsw as sw
import xarray as xr
import sys



def calc_ep(forcing_file, yr):
  # wind stress

  with nc.Dataset(forcing_file, 'r') as dataset:
    Itime = dataset.variables['Itime'][:]
    Itime2 = dataset.variables['Itime2'][:]

    ref = dt.datetime(1858, 11, 17)
    air_date = np.zeros((len(Itime)), dtype=object)
    for i in range(len(air_date)):
      air_date[i] = (ref + dt.timedelta(days=int(Itime[i])) 
          + dt.timedelta(seconds=int(Itime2[i]/1000)))
    d_yr = np.array([d.year for d in air_date])
    d_ind = d_yr == yr

    precip1 = dataset.variables['precip'][d_ind, :] # time, node
    evap1 = dataset.variables['evap'][d_ind, :] # m/s
    air_date = air_date[d_ind]

  # daily accumulations

  n_day = (precip1.shape[0] // 24)
  print(n_day, len(air_date))
  precip = np.zeros((n_day, evap1.shape[1]))
  evap = np.zeros((n_day, evap1.shape[1]))
  ep_date = np.zeros((n_day), dtype=object)
  for i in range(n_day):
    print(yr, i)
    precip[i, :] = np.sum(precip1[i * 24:(i + 1) * 24, :] * 60 * 60, axis=0)
    evap[i, :] = np.sum(evap1[i * 24:(i + 1) * 24, :] * 60 * 60, axis=0)
    ep_date[i] = air_date[i * 24]

  # units m per day 

  return evap, precip, ep_date


forcing_dir = '/dssgfs01/scratch/benbar/Forcing/'
out_dir = '/dssgfs01/scratch/benbar/Processed_Data/'
data_dir = '/dssgfs01/scratch/benbar/SSW_RS/SSW_RS_v1.2_2018_12_16/' 
fn_data = data_dir + 'SSWRS_V1.2_NOC_FVCOM_NWEuropeanShelf_01dy_20191231-1200_RE.nc'

fn = sorted(glob.glob(forcing_dir + '*.nc'))

#evap, precip, ep_date = calc_ep(fn[0], 1999)

with nc.Dataset(fn[0], 'r') as dataset:
  lon = dataset.variables['x'][:]
  lat = dataset.variables['y'][:]
  nv = dataset.variables['nv'][:]

triangles = np.array(nv).T -1

with nc.Dataset(fn_data, 'r') as dataset:
  art1 = dataset.variables['art1'][:]

# Get EOF mask
x_min = -10.05
x_max = 2.05

y_min = 53.98
y_max = 61.02

sel_ind = np.invert((lon >= x_min) & (lon <= x_max) 
                  & (lat >= y_min) & (lat <= y_max))

area = np.ma.sum(art1[sel_ind])
print(area)


c = 0
for yr in range(1993, 2020):
  print(yr)
  yr_str = str(yr)
  evap1, precip1, ep_date1 = calc_ep(forcing_dir 
          + 'SSW_Hindcast_metforcing_ERA5_' + yr_str +'.nc', yr)

  if c == 0:
    evap = evap1 * 1
    precip = precip1 * 1
    date_list = ep_date1
  else:
    evap = np.append(evap, evap1, axis=0)
    precip = np.append(precip, precip1, axis=0)
    date_list = np.append(date_list, ep_date1, axis=0)
  c = c + 1

# Integrate over area

evap_t = np.ma.zeros((evap.shape[0]))
precip_t = np.ma.zeros((precip.shape[0]))

for i in range(len(date_list)):
  evap_tmp = np.ma.masked_where(sel_ind, evap[i, :])
  precip_tmp = np.ma.masked_where(sel_ind, precip[i, :])
  evap_t[i] = np.ma.sum(evap_tmp * art1) # m3 / day
  precip_t[i] = np.ma.sum(precip_tmp * art1)

# integrated units change to m3/day

evap_t = evap_t.filled(1e20)
precip_t = precip_t.filled(1e20)


np.savez(out_dir + 'ep.npz', evap=evap_t, precip=precip_t, date_list=date_list)


