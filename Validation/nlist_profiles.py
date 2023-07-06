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
import properscoring as ps
from mpl_toolkits.basemap import Basemap
import haversine


#def find_index(fvcom_root, obs_root, dates):

# Model folder

mname = '/scratch/benbar/JASMIN/Model_Output_V3.02/Hourly/1995/'
mfiles = sorted(glob.glob(mname + 'SSWRS*hr*RE.nc'))
in_dir = '/scratch/benbar/Processed_Data_V3.02'

with nc.Dataset(mfiles[0], 'r') as nc_fid:
  lat_modg = nc_fid.variables['lat'][:]
  lon_modg = nc_fid.variables['lon'][:]
  siglay_mod = nc_fid.variables['siglay'][:]
  h_mod = nc_fid.variables['h'][:]
  zeta_mod = nc_fid.variables['zeta'][:]

lon_modg[lon_modg > 180] = lon_modg[lon_modg > 180] - 360

x1 = np.min(lon_modg) - 0.2
x2 = np.max(lon_modg) + 0.2
y1 = np.min(lat_modg) - 0.2
y2 = np.max(lat_modg) + 0.2

# Dates

st_date = dt.datetime(1993, 1, 1)
en_date = dt.datetime(2020, 1, 1)
dtime = (en_date - st_date).days

#print(x1, x2, y1, y2)

dims = {'siglay': [0]}
varlist = ['temp']

fvcom1 = MFileReader(mfiles[0], variables=varlist, dims=dims)
fvcom1.grid.lon[fvcom1.grid.lon > 180] = fvcom1.grid.lon[
    fvcom1.grid.lon > 180] - 360
fvcom1.grid.lonc[fvcom1.grid.lonc > 180] = fvcom1.grid.lonc[
    fvcom1.grid.lonc > 180] - 360

depth_mod = -h_mod * siglay_mod # layer, node
#depth_mod = -unstructured_grid_depths(h_mod, zeta_mod, 
#      siglay_mod, nan_invalid=True) # time, layer, node

# Observation folder

oname = '/scratch/benbar/Validation/Profiles/Delayed/'
ofiles = []

for i in range(dtime):
  date_f = st_date + dt.timedelta(days = i)
  str_date = date_f.strftime('%Y%m%d')
  ofiles.extend(glob.glob(oname + str(date_f.year) 
      + '/CO_DMQCGL01_' + str_date + '*.nc'))

# Find observations in model domain and time-frame

match_file = in_dir + '/Indices/index_full.npz'
obs_file = in_dir + '/Indices/prof_match_full.npz'


ref = dt.datetime(1950, 1, 1)

load_saved = 1
if load_saved:

  data = np.load(match_file, allow_pickle=True)
  m_date = data['m_date']
  m_hour = data['m_hour']
  m_node = data['m_node']
  m_sig = data['m_sig']

  o_date = data['o_date']
  o_prof = data['o_prof']
  dep_obs = data['dep_obs']

  file_checked = data['file_checked']
  data.close()

  file_checked = file_checked.tolist()

  data = np.load(obs_file, allow_pickle=True)
  lat_obs = data['lat_obs']
  lon_obs = data['lon_obs']
  prof_obs = data['prof_obs']
  dep_obs = data['dep_obs']
  temp_obs = data['temp_obs']
  sal_obs = data['sal_obs']
  date_obs = data['date_obs']
  data.close()

  temp_obs = np.ma.masked_where(temp_obs == -9999, temp_obs)
  sal_obs = np.ma.masked_where(sal_obs == -9999, sal_obs)

  pind = np.ma.max(prof_obs)

else:
  m_date = np.array([], dtype=object)
  m_hour = np.array([], dtype=int)
  m_node = np.array([], dtype=int)
  m_sig = np.array([], dtype=int)

  o_date = np.array([], dtype=object)
  o_prof = np.array([], dtype=int)

  file_checked = []

  lat_obs = np.array([])
  lon_obs = np.array([])
  prof_obs = np.array([])
  dep_obs = np.array([])
  temp_obs = np.ma.array([])
  sal_obs = np.ma.array([])
  date_obs = np.array([])

  pind = -1
print(len(file_checked))

if 1:
  for f in range(0, len(ofiles)):
    print(f, ofiles[f] in file_checked, len(ofiles))

    if ofiles[f] in file_checked:
      continue

    file_checked.append(ofiles[f])

    with nc.Dataset(ofiles[f], 'r') as nc_fid:
      # In QC fields 1 or 2 is good the rest are bad or no quality control
      # If adjusted value is available it should be used over not adjusted
      lat_tmp = nc_fid.variables['LATITUDE'][:]
      lon_tmp = nc_fid.variables['LONGITUDE'][:]
      dy_tmp = nc_fid.variables['JULD'][:] # jday relative to ref
      try:
        dep_tmp = nc_fid.variables['DEPH'][:]
        dep_tmp = nc_fid.variables['TEMP'][:]
      except:
        continue
        nc_fid.close()

      temp_tmp = nc_fid.variables['TEMP'][:]
      dep_adj = nc_fid.variables['DEPH_ADJUSTED'][:]
      temp_adj = nc_fid.variables['TEMP_ADJUSTED'][:]

      pos_qc = nc_fid.variables['POSITION_QC'][:]
      jday_qc = nc_fid.variables['JULD_QC'][:]
      dep_qc = nc_fid.variables['DEPH_QC'][:]
      temp_qc = nc_fid.variables['TEMP_QC'][:]
      dep_adj_qc = nc_fid.variables['DEPH_ADJUSTED_QC'][:]
      temp_adj_qc = nc_fid.variables['TEMP_ADJUSTED_QC'][:]

      try:
        sal_tmp = nc_fid.variables['PSAL'][:]
        sal_qc = nc_fid.variables['PSAL_QC'][:]
        sal_adj = nc_fid.variables['PSAL_ADJUSTED'][:]
        sal_adj_qc = nc_fid.variables['PSAL_ADJUSTED_QC'][:]
      except:
        sal_tmp = temp_tmp * 0 - 99
        sal_qc = np.ma.zeros(temp_qc.shape, dtype='b')
        sal_adj = temp_adj * 0 - 99
        sal_adj_qc = np.ma.zeros(temp_adj_qc.shape, dtype='b')


      pos_qc = pos_qc.filled(0).astype(int)
      jday_qc = jday_qc.filled(0).astype(int)
      dep_qc = dep_qc.filled(0).astype(int)
      temp_qc = temp_qc.filled(0).astype(int)
      sal_qc = sal_qc.filled(0).astype(int)
      dep_adj_qc = dep_adj_qc.filled(0).astype(int)
      temp_adj_qc = temp_adj_qc.filled(0).astype(int)
      sal_adj_qc = sal_adj_qc.filled(0).astype(int)

    nc_fid = None

    n_prof = len(lat_tmp)
    for i in range(n_prof):

      if ((pos_qc[i] < 1) | (pos_qc[i] > 2) 
            | (jday_qc[i] < 1) | (jday_qc[i] > 2)):
        continue

      if ((lon_tmp[i] > x1) & (lon_tmp[i] < x2) 
            & (lat_tmp[i] > y1) & (lat_tmp[i] < y2)):
        # rough cut out nearby profiles for speed

        if dep_adj[i].count() > 0:
          dep_tmp[i] = dep_adj[i] * 1
          dep_qc[i] = dep_adj_qc[i] * 1
        if temp_adj[i].count() > 0:
          temp_tmp[i] = temp_adj[i] * 1
          temp_qc[i] = temp_adj_qc[i] * 1
        if sal_adj[i].count() > 0:
          sal_tmp[i] = sal_adj[i] * 1
          sal_qc[i] = sal_adj_qc[i] * 1

        dep_tmp[i] = np.ma.masked_where((dep_qc[i] < 1) 
              | (dep_qc[i] > 2), dep_tmp[i])
        temp_tmp[i] = np.ma.masked_where((dep_qc[i] < 1) 
              | (dep_qc[i] > 2), temp_tmp[i])
        sal_tmp[i] = np.ma.masked_where((dep_qc[i] < 1) 
              | (dep_qc[i] > 2), sal_tmp[i])
        temp_tmp[i] = np.ma.masked_where((temp_qc[i] < 1) 
              | (temp_qc[i] > 2), temp_tmp[i])
        sal_tmp[i] = np.ma.masked_where((temp_qc[i] < 1) 
              | (temp_qc[i] > 2), sal_tmp[i])
        sal_tmp[i] = np.ma.masked_where((sal_qc[i] < 1) 
              | (sal_qc[i] > 2), sal_tmp[i])

        if (dep_tmp[i].count() == 0):
          continue
        if (temp_tmp[i].count() == 0):
          continue
        #if (sal_tmp[i].count() == 0):
        #  continue


        # Find nearest node of observation

        max_dist = 15000 # m
        xy_all = np.array([lon_tmp[i, np.newaxis], lat_tmp[i, np.newaxis]]).T
        if 0:
          node, dist = np.array([fvcom1.closest_node(ll, threshold=max_dist, 
              haversine=True, return_dists=True) for ll in xy_all]).flatten().tolist()
        else:

          # sub select grid points based on lat lon

          sub_ind = 0
          lat_size = 0
          lon_size = 0
          lat_inc = 0.01
          lon_inc = 0.02
          while np.sum(sub_ind) < 20:
            lat_size = lat_size + lat_inc
            lon_size = lon_size + lon_inc
            
            lat_min1 = lat_tmp[i] - lat_size
            lat_max1 = lat_tmp[i] + lat_size
            lon_min1 = lon_tmp[i] - lon_size
            lon_max1 = lon_tmp[i] + lon_size
            sub_ind = ((fvcom1.grid.lon > lon_min1) 
                & (fvcom1.grid.lon < lon_max1)
                & (fvcom1.grid.lat > lat_min1)
                & (fvcom1.grid.lat < lat_max1))

          # calculate distance of sub selected nodes
          sub_ind1 = np.nonzero(sub_ind)[0]
          dist_all = np.ma.zeros((len(fvcom1.grid.lon)))

          for sui in range(len(sub_ind1)):
            fvind = sub_ind1[sui]
            dist_all[fvind] = haversine.dist(lon_tmp[i], lat_tmp[i], 
                fvcom1.grid.lon[fvind], fvcom1.grid.lat[fvind])
          dist_all = np.ma.masked_where(np.invert(sub_ind), dist_all)
          node = np.ma.argmin(dist_all)
          dist = dist_all[node]
          if dist > max_dist:
            node = None

        print(node, dist, xy_all)
        #ele_ind = np.array([fvcom1.closest_element(ll, threshold=max_dist, 
        #      haversine=True) for ll in xy_all]).flatten().tolist()

        if node is not None:
          pind = pind + 1
          m_node = np.append(m_node, node)

          o_date = np.append(o_date, dt.timedelta(days = dy_tmp[i]) + ref)
          o_prof = np.append(o_prof, pind)

          lon_obs = np.append(lon_obs, lon_tmp[i])
          lat_obs = np.append(lat_obs, lat_tmp[i])
          prof_obs = np.append(prof_obs, np.zeros(len(dep_tmp[i, :])) + pind)
          dep_obs = np.append(dep_obs, dep_tmp[i, :])
          temp_obs = np.ma.append(temp_obs, temp_tmp[i, :])
          sal_obs = np.ma.append(sal_obs, sal_tmp[i, :])
          date_obs =np.append(date_obs, dt.timedelta(days = dy_tmp[i]) + ref)

    if f % 500 == 0: #load_save:
      
      if np.ma.is_masked(temp_obs):
        temp_obs = temp_obs.filled(-9999)
      if np.ma.is_masked(sal_obs):
        sal_obs = sal_obs.filled(-9999)

      np.savez(match_file, m_date=m_date, m_hour=m_hour, m_node=m_node, 
        m_sig=m_sig, 
        o_date=o_date, o_prof=o_prof, dep_obs=dep_obs, file_checked=file_checked)

      np.savez(obs_file, lat_obs=lat_obs, lon_obs=lon_obs, 
          dep_obs=dep_obs, temp_obs=temp_obs, sal_obs=sal_obs, 
          date_obs=date_obs, prof_obs=prof_obs)


# NOTE all indices identified by prof_obs == i

# Find nearest depth of observation
print(len(m_sig), len(temp_obs))
if len(m_sig) != len(temp_obs):
  print('Finding model points')

  if 0:
    for i in range(len(m_hour), len(o_date)):
      print(i / len(o_date) * 100, '%')
      idep = np.ma.zeros((len(dep_obs[prof_obs == i])), dtype=int)
      for d in range(len(dep_obs[prof_obs == i])):
        # Match obs to model
        dist1 = dep_obs[prof_obs == i][d] - depth_mod[:, int(m_node[i])]
        idep[d] = np.ma.argmin(np.ma.absolute(dist1))
        
        # Match model to obs if same result save else pick smaller distance
        dist2 = dep_obs[prof_obs == i] - depth_mod[idep[d], int(m_node[i])]
        idep2 = np.ma.argmin(np.ma.absolute(dist2))

        if d != idep2:
        # Mask non matching obs in profile
          if not np.ma.is_masked(temp_obs[prof_obs == i]):
            temp_obs[prof_obs == i].mask = np.zeros(temp_obs[prof_obs == i].shape)
          if not np.ma.is_masked(sal_obs[prof_obs == i]):
            sal_obs[prof_obs == i].mask = np.zeros(sal_obs[prof_obs == i].shape)
          temp_obs[prof_obs == i].mask[d] = True
          sal_obs[prof_obs == i].mask[d] = True

        #print(dist, dep_obs[i][d], depth_mod[int(idep[d]), int(node_ind[i])])
      m_sig = np.append(m_sig, idep)


  # Find nearest time of observation

  for i in range(len(m_hour), len(date_obs)):
    min_round = round(date_obs[i].minute / 60)
    hr_ind = date_obs[i].hour + min_round
    date_round = date_obs[i]
    if hr_ind == 24:
      date_round = date_round + dt.timedelta(hours = 1)
      hr_ind = 0
    m_date = np.append(m_date, date_round)
    m_hour = np.append(m_hour, hr_ind)


#    date_str = date_round[i].strftime('%Y%m%d')
#    fn = sorted(glob.glob(mname + 'SSW_Reanalysis_v1.1*/SSWRS*' 
#        + date_str + '*RE.nc'))[-1]
#    with nc.Dataset(fn, 'r') as nc_fid:
#      #print(hr_ind, mdiff[i], node_ind[i])
#      temp_mod.append(nc_fid.variables['temp'][hr_ind, m_sig[i], node_ind[i]])
#      sal_mod.append(nc_fid.variables['salinity'][hr_ind, m_sig[i], node_ind[i]])

  # Save the indices

  if 1:#load_save:

    if np.ma.is_masked(temp_obs):
      temp_obs = temp_obs.filled(-9999)
    if np.ma.is_masked(sal_obs):
      sal_obs = sal_obs.filled(-9999)

    np.savez(match_file, m_date=m_date, m_hour=m_hour, m_node=m_node, 
      m_sig=m_sig, 
      o_date=o_date, o_prof=o_prof, dep_obs=dep_obs, file_checked=file_checked)

    np.savez(obs_file, lat_obs=lat_obs, lon_obs=lon_obs, 
        dep_obs=dep_obs, temp_obs=temp_obs, sal_obs=sal_obs, 
        date_obs=date_obs, prof_obs=prof_obs)



