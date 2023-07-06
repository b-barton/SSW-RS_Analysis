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

  amm15_root = '/projectsa/NEMO/Simulations/AMM15/RIVERAMM15/'


  fnt = []
  for yr in range(2005, 2015):
    fnt.extend(sorted(glob.glob(amm15_root + str(yr) + '/*/*25hourm_*_T_cmp3.nc')))

  print(len(fnt))
  out_dir = '/scratch/benbar/Processed_Data_V3.02/'

  with nc.Dataset(fnt[0], 'r') as src_amm15:
    lat1_g = src_amm15.variables['nav_lat'][:]
    lon1_g = src_amm15.variables['nav_lon'][:]
    dep1 = src_amm15.variables['deptht'][:]
    temp1 = src_amm15.variables['votemper'][0, :, :, :] # t, d, y, x

  date_amm15_st = np.zeros((len(fnt)), dtype=object)
  date_amm15_en = np.zeros((len(fnt)), dtype=object)
  for i in range(len(fnt)):
    date_amm15_st[i] = dt.datetime.strptime(fnt[i].split('_')[-6], 
        '%Y%m%d')
    date_amm15_en[i] = dt.datetime.strptime(fnt[i].split('_')[-5], 
      '%Y%m%d')

  dep_g = np.tile(dep1, (lon1_g.shape[0], lat1_g.shape[1], 1)).T

  dx = np.zeros(lon1_g.shape) - 999
  dy = np.zeros(lon1_g.shape) - 999
  for xi in range(lon1_g.shape[1] - 1):
    for yi in range(lon1_g.shape[0] - 1):
      dy[yi, xi] = haversine.dist(lon1_g[yi, xi], lat1_g[yi, xi], 
                                          lon1_g[yi, xi], lat1_g[yi + 1, xi]) 
      dx[yi, xi] = haversine.dist(lon1_g[yi, xi], lat1_g[yi, xi], 
                                          lon1_g[yi, xi + 1], lat1_g[yi, xi]) 

  dy[-1, :] = dy[-2, :]
  dy[:, -1] = dy[:, -2]
  dx[-1, :] = dx[-2, :]
  dx[:, -1] = dx[:, -2]

  temp1 = np.ma.masked_where(temp1 == -32768, temp1)
  dzdx, dzdy = np.gradient(temp1.mask[0, :, :] * 1)
  dzdx = dzdx / dx
  dzdy = dzdy / dy
  mask_t = ((dzdx ** 2 + dzdy ** 2) ** 0.5) > 0


  mag_t_grad = np.ma.zeros((len(fnt), lat1_g.shape[0], lon1_g.shape[1])) - 999
  mag_s_grad = np.ma.zeros((len(fnt), lat1_g.shape[0], lon1_g.shape[1])) - 999
  mag_r_grad = np.ma.zeros((len(fnt), lat1_g.shape[0], lon1_g.shape[1])) - 999
  #ua_app = np.ma.zeros((len(fnt), lon1.shape[0], lat1.shape[0])) - 999
  #va_app = np.ma.zeros((len(fnt), lon1.shape[0], lat1.shape[0])) - 999
  date_list = np.zeros((len(fnt)), dtype=object) - 999
  ref1 = dt.datetime(1900, 1, 1)

  for i in range(0, len(fnt)):
    print(i / len(fnt) *100, '%')
    with nc.Dataset(fnt[i], 'r') as src_amm15:
      day_diff = len(src_amm15.variables['time_centered'])

    #day_diff = date_amm15_en[i].day - 1
    tmp_t_grad = np.ma.zeros((day_diff, lat1_g.shape[0], lon1_g.shape[1])) - 999
    tmp_s_grad = np.ma.zeros((day_diff, lat1_g.shape[0], lon1_g.shape[1])) - 999
    tmp_r_grad = np.ma.zeros((day_diff, lat1_g.shape[0], lon1_g.shape[1])) - 999
    date_tmp = np.zeros((day_diff), dtype=object) - 999

    for ti in range(day_diff):
      with nc.Dataset(fnt[i], 'r') as src_amm15:
        if np.ma.is_masked(src_amm15.variables['time_centered'][ti]):
          continue

        time1 = (ref1 + dt.timedelta(
            seconds=int(src_amm15.variables['time_centered'][ti])))
        temp1 = src_amm15.variables['votemper'][ti, 0, :, :].squeeze() # t, d, lat, lon
        sal1 = src_amm15.variables['vosaline'][ti, 0, :, :].squeeze()
        mask_sal = sal1 == 1e20
      temp1 = np.ma.masked_where(mask_sal, temp1)
      sal1 = np.ma.masked_where(mask_sal, sal1)


      dzdx, dzdy = np.gradient(temp1[:, :])
      dzdx = dzdx / dx
      dzdy = dzdy / dy
      tmp_t_grad[ti, :, :] = (dzdx ** 2 + dzdy ** 2) ** 0.5

      dzdx, dzdy = np.gradient(sal1[:, :])
      dzdx = dzdx / dx
      dzdy = dzdy / dy
      tmp_s_grad[ti, :, :] = (dzdx ** 2 + dzdy ** 2) ** 0.5

      rho = calc_rho(sal1, temp1, dep_g[0, :, :].T, np.mean(lon1_g), np.mean(lat1_g))  

      dzdx, dzdy = np.gradient(rho[:, :])
      dzdx = dzdx / dx
      dzdy = dzdy / dy
      tmp_r_grad[ti, :, :] = (dzdx ** 2 + dzdy ** 2) ** 0.5

      #ua_app[c, :] = FVCOM['ua'][t1, :]
      #va_app[c, :] = FVCOM['va'][t1, :]

      date_tmp[ti] = time1

      tmp_t_grad.mask[ti, :, :] = mask_t
      tmp_s_grad.mask[ti, :, :] = mask_t
      tmp_r_grad.mask[ti, :, :] = mask_t

    tmp_t_grad = np.ma.masked_where(tmp_t_grad == -999, tmp_t_grad)
    tmp_s_grad = np.ma.masked_where(tmp_s_grad == -999, tmp_s_grad)
    tmp_r_grad = np.ma.masked_where(tmp_r_grad == -999, tmp_r_grad)

    mag_t_grad[i, :, :] = np.ma.mean(tmp_t_grad, axis=0)
    mag_s_grad[i, :, :] = np.ma.mean(tmp_s_grad, axis=0)
    mag_r_grad[i, :, :] = np.ma.mean(tmp_r_grad, axis=0)
    date_list[i] = date_tmp[0]

#  tb_rho = tb_rho[0:c, :]
#  ua_app = ua_app[0:c, :]
#  va_app = va_app[0:c, :]
#  mag_t_grad = mag_t_grad[0:c, :, :]
#  mag_s_grad = mag_s_grad[0:c, :, :]
#  mag_r_grad = mag_r_grad[0:c, :, :]
  mag_t_grad = mag_t_grad.filled(-999)
  mag_s_grad = mag_s_grad.filled(-999)
  mag_r_grad = mag_r_grad.filled(-999)

  np.savez(out_dir + 'amm15_old_front.npz', mag_t_grad=mag_t_grad, mag_s_grad=mag_s_grad, mag_r_grad=mag_r_grad, date_list=date_list, lon=lon1_g, lat=lat1_g)


