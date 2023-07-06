#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PyFVCOM.read import MFileReader
from PyFVCOM.plot import Time
from PyFVCOM.grid import unstructured_grid_depths
import PyFVCOM.plot as fvplot
import glob
import netCDF4 as nc
import datetime as dt

# NOTE mount JASMIN first

# Model folder
in_dir = '/scratch/benbar/Processed_Data'
match_file = in_dir + '/Indices/vel_index_full.npz'
mod_file = in_dir + '/Indices/velocity_model_full.npz'
obs_file = in_dir + '/Indices/vel_match_full.npz'
mjas = '/scratch/benbar/JASMIN/Model_Output/Daily/'
mfiles = sorted(glob.glob(mjas + '*/SSWRS*RE.nc'))

with nc.Dataset(mfiles[0], 'r') as nc_fid:
  lat_modg = nc_fid.variables['latc'][:]
  lon_modg = nc_fid.variables['lonc'][:]
  siglay_mod = nc_fid.variables['siglay_center'][:]
  h_mod = nc_fid.variables['h_center'][:]
  zeta_mod = nc_fid.variables['zeta'][:]

lon_modg[lon_modg > 180] = lon_modg[lon_modg > 180] - 360

depth_mod = -h_mod * siglay_mod

data = np.load(match_file, allow_pickle=True)
m_date = data['m_date']
m_hour = data['m_hour']
m_elem = data['m_elem']
m_sig = data['m_sig']

o_date = data['o_date']
o_prof = data['o_prof']
dep_obs = data['dep_obs']
data.close()

dep_obs = np.ma.masked_where(dep_obs >= 9999, dep_obs)
m_elem = m_elem.astype(int)

u_mod = np.ma.zeros(dep_obs.shape) -9999
v_mod = np.ma.zeros(dep_obs.shape) -9999
h_mod = np.ma.zeros(dep_obs.shape) -9999

data = np.load(obs_file, allow_pickle=True)
prof_obs = data['prof_obs']
data.close()

depth_mod = depth_mod - 1 # make sure drifter is in model

print(np.min(dep_obs), np.max(dep_obs))
# Fix o_prof and prof_obs
#pind = 0
#sti = 0
#for i in range(len(dep_obs) -1):
#  if (prof_obs[i] != prof_obs[i + 1]):
#    print(i / len(dep_obs) *100, '%')
#    o_prof[pind] = pind
#    prof_obs[sti:i + 1] = pind
#    sti = i
#    pind = pind + 1

#o_prof[pind] = pind
#prof_obs[sti:i + 1] = pind
print(len(o_prof), o_prof[-1])
print(len(prof_obs), prof_obs[-1])
print(o_date[0], o_date[-1])
ref = dt.datetime(1858, 11, 17)

for i in range(len(o_prof)):
  print(i / len(o_prof) *100, '%')
  ind = prof_obs == o_prof[i]

  if m_date[i] >= dt.datetime(2018, 12, 20):
    continue

  # loop back through 10 days and stop whe the file is found
  found = 0
  for d in range(10):
    ddiff = dt.timedelta(days = d)
    date_str = (m_date[i] - ddiff).strftime('%Y%m%d')
    fn = sorted(glob.glob(mjas + '*/SSWRS*' 
        + date_str + '*RE.nc'))

    if len(fn):
      found = 1
      print(date_str)
      with nc.Dataset(fn[-1], 'r') as nc_fid:
        itime = nc_fid.variables['Itime'][:]
        if len(itime) < d:
          break
        #idate = np.array([], dtype=object)
        #for ti in range(len(itime)):
        #  idate = np.append(idate, itime[ti] + ref)

        u_p = (nc_fid.variables['u'][d, :, m_elem[i]])
        v_p = (nc_fid.variables['v'][d, :, m_elem[i]])
        h_p = (nc_fid.variables['h_center'][m_elem[i]])
        break

    else:
      continue

  if found == 0:
    continue

  u_mod[ind] = np.interp(dep_obs[ind], depth_mod[:, m_elem[i]], u_p)
  v_mod[ind] = np.interp(dep_obs[ind], depth_mod[:, m_elem[i]], v_p)
  h_mod[ind] = h_p * 1

  #plt.plot(temp_p, depth_mod[:, m_node[i]])
  #plt.plot(temp_mod[ind], dep_obs[ind])
  #plt.show()

print(np.max(u_mod))
u_mod[dep_obs == -9999] = -9999
v_mod[dep_obs == -9999] = -9999
h_mod[dep_obs == -9999] = -9999


np.savez(mod_file, u_mod=u_mod, v_mod=v_mod, h_mod=h_mod)



