#!/usr/bin/env python3

""" Plot a surface from an FVCOM model output.

"""

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
import haversine

if __name__ == '__main__':

  def calc_rho(sp, tp, depth, lon, lat):
    pres = sw.p_from_z(depth * -1, lat)
    sa = sw.SA_from_SP(sp, pres, lon, lat)
    ct = sw.CT_from_pt(sa, tp)
    rho = sw.rho(sa, ct, 0)
    return rho

  amm7_root = '/scratch/benbar/AMM7_N/'

  fnt = []
  for yr in range(1993, 2020):
    fnt.extend(sorted(glob.glob(amm7_root + 'Temperature/' + str(yr) + '/*.nc')))
  fns = []
  for yr in range(1993, 2020):
    fns.extend(sorted(glob.glob(amm7_root + 'Salinity/' + str(yr) + '/*.nc')))

  print(len(fnt))
  out_dir = '/scratch/benbar/Processed_Data_V3.02/'

  with nc.Dataset(fnt[0], 'r') as src_amm7:
    lat1 = src_amm7.variables['latitude'][:]
    lon1 = src_amm7.variables['longitude'][:]
    dep1 = src_amm7.variables['depth'][:]
    temp1 = src_amm7.variables['thetao'][0, :, :, :]

  lon_g, lat_g = np.meshgrid(lon1, lat1)
  dep_g = np.tile(dep1, (len(lon1), len(lat1), 1)).T
  dx, dy = haversine.area(lon1, lat1)

  temp1 = np.ma.masked_where(temp1 == -32768, temp1)
  dzdx, dzdy = np.gradient(temp1.mask[0, :, :] * 1)
  dzdx = dzdx / dx
  dzdy = dzdy / dy
  mask_t = ((dzdx ** 2 + dzdy ** 2) ** 0.5) > 0

  bottom_ind = np.zeros((temp1.shape[1], temp1.shape[2]), dtype=int)
  for i in range(len(lon1)):
    for j in range(len(lat1)):
      mask = temp1.mask[:, j, i]
      ind = np.nonzero(np.invert(mask))[0]
      print(ind)
      if len(ind) == 0:
        continue 

      ind = np.max(ind)
      bottom_ind[j, i] = ind

  mag_t_grad = np.ma.zeros((len(fnt), lat1.shape[0], lon1.shape[0])) - 999
  mag_s_grad = np.ma.zeros((len(fnt), lat1.shape[0], lon1.shape[0])) - 999
  mag_r_grad = np.ma.zeros((len(fnt), lat1.shape[0], lon1.shape[0])) - 999
  tb_rho = np.ma.zeros((len(fnt), lat1.shape[0], lon1.shape[0])) - 999
  #ua_app = np.ma.zeros((len(fnt), lon1.shape[0], lat1.shape[0])) - 999
  #va_app = np.ma.zeros((len(fnt), lon1.shape[0], lat1.shape[0])) - 999
  date_list = np.zeros((len(fnt)), dtype=object) - 999
  c = 0

  ref = dt.datetime(1970, 1, 1)

  for i in range(len(fnt)):
    print(i / len(fnt) *100, '%')

    with nc.Dataset(fnt[i], 'r') as src_amm7:
      time1 = ref + dt.timedelta(seconds=int(src_amm7.variables['time'][0]))
      temp1 = src_amm7.variables['thetao'][0, :, :, :] # t, d, lat, lon
      temp1 = np.ma.masked_where(temp1 == -32768, temp1)
    with nc.Dataset(fns[i], 'r') as src_amm7:
      sal1 = src_amm7.variables['so'][0, :, :, :]
      sal1 = np.ma.masked_where(sal1 == -32768, sal1)

    dzdx, dzdy = np.gradient(temp1[0, :, :])
    dzdx = dzdx / dx
    dzdy = dzdy / dy
    mag_t_grad[c, :, :] = (dzdx ** 2 + dzdy ** 2) ** 0.5

    dzdx, dzdy = np.gradient(sal1[0, :, :])
    dzdx = dzdx / dx
    dzdy = dzdy / dy
    mag_s_grad[c, :, :] = (dzdx ** 2 + dzdy ** 2) ** 0.5

    rho = calc_rho(sal1, temp1, dep_g, np.mean(lon1), np.mean(lat1))  

    dzdx, dzdy = np.gradient(rho[0, :, :])
    dzdx = dzdx / dx
    dzdy = dzdy / dy
    mag_r_grad[c, :, :] = (dzdx ** 2 + dzdy ** 2) ** 0.5

    rho_bot = np.ma.zeros(bottom_ind.shape)
    for i in range(len(lon1)):
      for j in range(len(lat1)):
        rho_bot[j, i] = rho[bottom_ind[j, i], j, i]

    tb_rho[c, :, :] = rho[0, :, :] - rho_bot

    #ua_app[c, :] = FVCOM['ua'][t1, :]
    #va_app[c, :] = FVCOM['va'][t1, :]

    date_list[c] = time1

    mag_t_grad.mask[c, :, :] = mask_t
    mag_s_grad.mask[c, :, :] = mask_t
    mag_r_grad.mask[c, :, :] = mask_t
    c = c + 1

#  tb_rho = tb_rho[0:c, :]
#  ua_app = ua_app[0:c, :]
#  va_app = va_app[0:c, :]
  mag_t_grad = mag_t_grad.filled(-999)
  mag_s_grad = mag_s_grad.filled(-999)
  mag_r_grad = mag_r_grad.filled(-999)
  tb_rho = tb_rho.filled(-999)

  np.savez(out_dir + 'amm7_n_front.npz', mag_t_grad=mag_t_grad, mag_s_grad=mag_s_grad, mag_r_grad=mag_r_grad, tb_rho=tb_rho, date_list=date_list, lon=lon1, lat=lat1)


