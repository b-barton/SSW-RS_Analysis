#!/usr/bin/env python3

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stats
import PyFVCOM as fvcom


out_dir = '/scratch/benbar/Processed_Data_V3.02/'
fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
fgrd = fin + 'SSW_Hindcast_riv_xe3_grd.dat'

# Process

use_svd = 1
rm_season = 1
do_temp = 0

st_date = dt.datetime(1993, 1, 1)
en_date = dt.datetime(2019, 12, 1)
#en_date = dt.datetime(2005, 12, 1)

if do_temp:
  f_end = 'temp'
else:
  f_end = 'sal'


data = np.load(out_dir + 'stack_ts_mn.npz', allow_pickle=True)
date = data['date_list']
print(date[-1])
i_st = np.nonzero(date == st_date)[0][0]
i_en = np.nonzero(date == en_date)[0][0] + 1
print(i_st, i_en)
date = date[i_st:i_en]

lat = data['lat']
lon = data['lon']
temp = data[f_end][i_st:i_en, :] # time, space
triangles = data['tri']
data.close()

st_date = date[0]
en_date = date[-1]

print(np.count_nonzero(np.isnan(temp)))

fvg = fvcom.preproc.Model(st_date, en_date, grid=fgrd, 
                    native_coordinates='spherical', zone='30N')

lonc = fvg.grid.lonc
latc = fvg.grid.latc

# Select Scotland mask rest for removal

x_min = -10.05
x_max = 2.05

y_min = 53.98
y_max = 61.02

sel_ind = np.invert((lon >= x_min) & (lon <= x_max) 
                  & (lat >= y_min) & (lat <= y_max))

temp = np.ma.array(temp)

for i in range(temp.shape[0]):
  temp[i, :] = np.ma.masked_where(sel_ind, temp[i, :])


# Remove seasonal cycle

if rm_season == 1:
  run = 12
  temp_s = np.ma.zeros((len(date) - run, temp.shape[1]))
  date_s = np.empty(len(date) - run, dtype=object)

  for i in range(len(date_s)):
    temp_s[i, :] = np.ma.mean(temp[i:i+run, :], axis=0)
    date_s[i] = date[i + int(run//2)]

  date = date_s

else:
  temp_s = temp * 1

# flatten
# in temp_flat, rows(M) = space and columns(N) = time; (M,N) 
size = np.shape(temp_s)
print (size)
if len(size) == 3:
  temp_flat = np.reshape(temp_s, (size[0], size[1] * size[2]))
else:
  temp_flat = temp_s

temp_flat = temp_flat.T


size_flat = temp_flat.shape
print(size_flat)

# remove masked grid points
# count_masked returns array with length of the space points and each element 
# representing the number of masked time points. If all the time points 
# are masked, it is land and therefore not selected in keep_ind
cmask = np.ma.count_masked(temp_flat, axis=1)
keep_ind = np.nonzero(cmask != size_flat[1])[0]
temp_sub = temp_flat[keep_ind, :]
size_sub = temp_sub.shape
print(size_sub)

# remove mean
temp_mean = np.ma.mean(temp_sub, axis=1)
temp_detrend = np.ma.empty_like(temp_sub)
for i in range(temp_sub.shape[0]):
  temp_detrend[i, :] = temp_sub[i, :] - temp_mean[i]

# remove trend
mean_trend = np.ma.mean(temp_detrend - sig.detrend(temp_detrend, axis=1, type='linear'), axis=1)
print(mean_trend[1] - mean_trend[0])
temp_detrend = sig.detrend(temp_detrend, axis=1, type='linear')
#temp_detrend = np.ma.masked_where(temp_sub.mask, temp_detrend)


# remove standard deviation
st_dev = np.ma.std(temp_detrend, axis=1)

for i in range(size_sub[0]):
  temp_detrend[i, :] = temp_detrend[i, :] / st_dev[i]


if size_sub[0] < size_sub[1]:
  raise Exception('Rows shorter than columns! i.e. more locations that times')


if use_svd:
  # SVD

  vect_space, s_val, vect_time = np.linalg.svd(temp_detrend, 
                                full_matrices=False)
  n_eig = len(s_val)
  s_val = np.diag(s_val)
  eig_val = s_val ** 2
  print (vect_space.shape, s_val.shape, vect_time.shape)

  amp = np.zeros((size_sub[1], n_eig))
  for i in range(n_eig):
    amp[:, i] = vect_time.T[:, i] * s_val.T[i, i]

else:
  # EOF analysis

  temp_covar = np.ma.dot(temp_detrend, temp_detrend.T)
  L, C = np.linalg.eig(temp_covar)
  n_eig = len(L)
  eig_val = np.diag(np.real(L))
  vect_space = np.real(C)

  amp = np.zeros((size_sub[1], n_eig))
  for i in range(size_sub[1]): # time
    amp_sum = np.zeros((size_sub[0], n_eig))
    for j in range(size_sub[0]): # space
      amp_sum[j, :] = (temp_detrend[j, i] * vect_space[j, :])
    amp[i, :] = np.ma.sum(amp_sum, axis=0)
  print (vect_space.shape, amp.shape, L.shape)

  # eigenvector[:, 0] corresponds to eigenvalue[0]
  # the rows corespond to locations.
  # the values in the eigenvector[:, 0] tell you how much the signal in that
  # area is affected by EOF1.
  # eigenvalue[0] / sum(eigenvalues) tell you how much of the total variance 
  # is accounted for by EOF1.


# Normalize
amp_norm = np.zeros_like(amp)
space_norm = np.zeros((size_flat[0], n_eig)) -1e20

for i in range(n_eig):
  amp_norm[:, i] = amp[:, i] / np.ma.std(amp[:, i])
  space_norm[keep_ind, i] = (1 / n_eig * 
                              np.dot(temp_detrend, amp_norm[:, i]))

space_norm = np.ma.masked_where(space_norm == -1e20, space_norm)
print ('S.D.', np.ma.std(amp[:, 0]), np.ma.std(amp[:, 1]), np.ma.std(amp[:, 2]))
print ('S.D.', np.ma.std(amp_norm[:, 0]), np.ma.std(amp_norm[:, 1]), np.ma.std(amp_norm[:, 2]))
print(np.ma.sum(np.ma.std(amp, axis=1) ** 2)) # should be roughly equal to number of variables



if len(size) == 3:
  eof = np.ma.zeros((size[1], size[2], n_eig))
  for i in range(n_eig):
  #  space_norm[:, i] = space_norm[:, i] / np.ma.std(space_norm[:, i])
    eof[:, :, i] = np.reshape(space_norm[:, i], (size[1], size[2]))
else:
  eof = space_norm * 1

# Percent of variability
var_per = np.zeros((n_eig))
for i in range(n_eig):
  var_per[i] = (eig_val[i, i]/np.sum(eig_val)) * 100
print('Variance explained:', var_per[:10])

x = np.arange(1, 16)
plt.plot(x, var_per[:15], 'o-')
plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
plt.ylabel("Variance")
plt.show()

# Significance. If orth at i != j greater than 0.05. i = j should be 1

orth = np.dot(vect_space.T, vect_space)
#orth = orth.filled(-1e20)
eof_save = eof.filled(-1e20)

# TODO: select top 5 eof modes for saving

if rm_season:
  np.savez(out_dir + 'eof_data_mn_' + f_end, variability=var_per, space_eof=eof_save, amp_pc=amp_norm, date=date, lat=lat, lon=lon, tri=triangles, orth=orth)
else:
  np.savez(out_dir + 'eof_data_season_mn_' + f_end, variability=var_per, space_eof=eof_save, amp_pc=amp_norm, date=date, lat=lat, lon=lon, tri=triangles, orth=orth)


