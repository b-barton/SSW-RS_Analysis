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

if __name__ == '__main__':

  def calc_rho(sp, tp, depth, lon, lat):
    pres = sw.p_from_z(depth * -1, lat)
    sa = sw.SA_from_SP(sp, pres, lon, lat)
    ct = sw.CT_from_pt(sa, tp)
    rho = sw.rho(sa, ct, 0)
    return rho

  mjas = '/gws/nopw/j04/ssw_rs/Model_Output_V3.02/Daily/'
  fn = []
  for yr in range(1993, 2020):
    fn.extend(sorted(glob.glob(mjas + str(yr) + '/SSWRS*V3.02*dy*RE.nc')))

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

  vars = ('zeta', 'temp', 'salinity', 'ua', 'va', 'Times', 'h')

  mag_t_grad = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
  mag_s_grad = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
  mag_r_grad = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
  tb_rho = np.ma.zeros((len(fn) * 10, lon.shape[0])) - 999
#  ua_app = np.ma.zeros((len(fn) * 10, lonc.shape[0])) - 999
#  va_app = np.ma.zeros((len(fn) * 10, lonc.shape[0])) - 999
  date_list = np.zeros((len(fn) * 10), dtype=object) - 999
  c = 0

  for i in range(len(fn)):
    print(i / len(fn) *100, '%')
    FVCOM = readFVCOM(fn[i], vars, dims=dims)
    for t1 in range(FVCOM['ua'].shape[0]):

      a = tri.LinearTriInterpolator(trio, FVCOM['temp'][t1, 0, :])
      dzdx, dzdy = a.gradient(x, y)
      mag_t_grad[c, :] = (dzdx ** 2 + dzdy ** 2) ** 0.5

      a = tri.LinearTriInterpolator(trio, FVCOM['salinity'][t1, 0, :])
      dzdx, dzdy = a.gradient(x, y)
      mag_s_grad[c, :] = (dzdx ** 2 + dzdy ** 2) ** 0.5

      rho = calc_rho(FVCOM['salinity'][t1, :, :], FVCOM['temp'][t1, :, :], 
            depth_mod, lon, lat)   
      a = tri.LinearTriInterpolator(trio, rho[0, :])
      dzdx, dzdy = a.gradient(x, y)
      mag_r_grad[c, :] = (dzdx ** 2 + dzdy ** 2) ** 0.5

      tb_rho[c, :] = rho[0, :] - rho[-1, :]

      #ua_app[c, :] = FVCOM['ua'][t1, :]
      #va_app[c, :] = FVCOM['va'][t1, :]

      date_list[c] = dt.datetime.strptime(''.join(FVCOM['Times'][t1, :-7].astype(str)).replace('T', ' '), '%Y-%m-%d %H:%M:%S')
      c = c + 1

  date_list = date_list[0:c]
  mag_t_grad = mag_t_grad[0:c, :]
  mag_s_grad = mag_s_grad[0:c, :]
  mag_r_grad = mag_r_grad[0:c, :]
  tb_rho = tb_rho[0:c, :]
#  ua_app = ua_app[0:c, :]
#  va_app = va_app[0:c, :]


  np.savez(out_dir + 'front.npz', mag_t_grad=mag_t_grad, mag_s_grad=mag_s_grad, mag_r_grad=mag_r_grad, tb_rho=tb_rho, date_list=date_list, lon=lon, lat=lat, lonc=lonc, latc=latc)



