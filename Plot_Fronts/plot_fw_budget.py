#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime as dt
import netCDF4 as nc
import glob
import matplotlib.tri as tri
from PyFVCOM.read import ncread as readFVCOM
from PyFVCOM.read import FileReader 
import scipy.stats as stats

in_dir = '/scratch/benbar/Processed_Data_V3.02/'

fin = '/projectsa/SSW_RS/FVCOM/input/SSW_Reanalysis/'
river_nc_file = fin + 'SSW_Hindcast_riv_xe3_rivers.nc'
river_nml_file = fin + 'SSW_Hindcast_riv_xe3.nml'


data = np.load(in_dir + 'ep.npz', allow_pickle=True)
# daily accumulations summed over box
evap = data['evap'] # m3 / day
precip = data['precip']
date_ep = data['date_list']
data.close()

evap = np.ma.masked_where(evap == 1e20, evap)
precip = np.ma.masked_where(precip == 1e20, precip)

data = np.load(in_dir + 'fw_budget_node.npz', allow_pickle=True)
#data = np.load(in_dir + 'fw_budget.npz', allow_pickle=True)
# daily transports summed over transects
across_fw = data['across_fw'] # m3 / s
across_transp = data['across_trans'] 
date_tr = data['date_list']
data.close()

evap = evap[:len(across_fw)]
precip = precip[:len(across_fw)]
date_ep = date_ep[:len(across_fw)]

data = np.load(in_dir + 'fw_volume_35.npz', allow_pickle=True)
# fw volume change
fw_volume = data['fw_volume'] # m3
dfw_vol = data['delta_fw_vol'] # m3 / s
date_vol = data['date_list']
data.close()

dfw_vol = dfw_vol[:len(across_fw)]
date_vol = date_vol[:len(across_fw)]


ax1 = plt.subplot(2, 1, 1)

ax1.plot(date_tr, across_fw[:, 0], label='Irish Sea')
ax1.plot(date_tr, across_fw[:, 1] + across_fw[:, 2], label='West')
ax1.plot(date_tr, across_fw[:, 3], label='North')
ax1.plot(date_tr, across_fw[:, 4] + across_fw[:, 5], label='North Sea')
ax1.legend()


plt.savefig('./Figures/fw_test.png')


# Convert to m3 / day
#along_fw = along_fw * 24 * 60 * 60
#across_fw = across_fw * 24 * 60 * 60

# Convert to m3/s
evap = evap / (24 * 60 * 60)
precip = precip / (24 * 60 * 60)
across_fw = across_fw 

# Get river data

dims = {'time': slice(0, 10)}
vars = ('Times', 'time', 'Itime', 'Itime2')

fn = in_dir + 'SSWRS_V3.02_NOC_FVCOM_NWEuropeanShelf_01dy_19930101-1200_RE.nc'
fvf = FileReader(fn, list(vars), dims=dims)
fvf.add_river_flow(river_nc_file, river_nml_file)

riv_lat = fvf.river.river_lat
riv_lon = fvf.river.river_lon
print(np.shape(fvf.river.river_fluxes)) # time, location
riv_lon[riv_lon > 180] = riv_lon[riv_lon > 180] - 360

river_flux = fvf.river.raw_fluxes # units m3/s
#river_flux = river_flux * 24 * 60 * 60 # m3/day

# Monthly means

yr = np.array([d.year for d in fvf.river.time_dt])
mnt = np.array([d.month for d in fvf.river.time_dt])
yri = np.unique(yr)
mnti =np.unique(mnt)

flux_mn = np.ma.zeros((len(yri) * len(mnti), river_flux.shape[1]))
date_mn = np.zeros((len(yri) * len(mnti)), dtype=object)
count = 0
for i in range(len(yri)):
  for j in range(len(mnti)):
    ind = (yr == yri[i]) & (mnt == mnti[j])
    flux_mn[count, :] = np.ma.mean(river_flux[ind, :], axis=0)
    date_mn[count] = dt.datetime(yri[i], mnti[j], 1)
    count += 1

yr = np.array([d.year for d in date_ep])
mnt = np.array([d.month for d in date_ep])
yri = np.unique(yr)
mnti =np.unique(mnt)

e_mn = np.ma.zeros((len(yri) * len(mnti)))
p_mn = np.ma.zeros((len(yri) * len(mnti)))
dfw_vol_mn = np.ma.zeros((len(yri) * len(mnti)))
across_mn = np.ma.zeros((len(yri) * len(mnti), across_fw.shape[1]))

date_ep_mn = np.zeros((len(yri) * len(mnti)), dtype=object)
count = 0
for i in range(len(yri)):
  for j in range(len(mnti)):
    ind = (yr == yri[i]) & (mnt == mnti[j])
    e_mn[count] = np.ma.mean(evap[ind])
    p_mn[count] = np.ma.mean(precip[ind])
    dfw_vol_mn[count] = np.ma.mean(dfw_vol[ind])
    across_mn[count, :] = np.ma.mean(across_fw[ind, :], axis=0)
    date_ep_mn[count] = dt.datetime(yri[i], mnti[j], 1)
    count += 1

# running mean

run = 12
riv_run = np.zeros((len(date_mn) - run, flux_mn.shape[1]))
run_date = np.empty(len(date_mn) - run, dtype=object)
for i in range(len(run_date)):
  riv_run[i, :] = np.ma.mean(flux_mn[i:i+run, :], axis=0)
  run_date[i] = date_mn[i + run // 2]


e_run = np.zeros((len(date_ep_mn) - run))
p_run = np.zeros((len(date_ep_mn) - run))
vol_run = np.zeros((len(date_ep_mn) - run))
across_run = np.zeros((len(date_ep_mn) - run, across_mn.shape[1]))
run_ep_date = np.empty(len(date_ep_mn) - run, dtype=object)
for i in range(len(run_ep_date)):
  e_run[i] = np.ma.mean(e_mn[i:i+run])
  p_run[i] = np.ma.mean(p_mn[i:i+run])
  vol_run[i] = np.ma.mean(dfw_vol_mn[i:i+run])
  across_run[i, :] = np.ma.mean(across_mn[i:i+run, :], axis=0)
  run_ep_date[i] = date_ep_mn[i + run // 2]

st_d = np.nonzero(run_date >= run_ep_date[0])[0][0]
en_d = np.nonzero(run_date >= run_ep_date[-1])[0][0] + 1
riv_run = riv_run[st_d:en_d, :]
riv_date = run_date[st_d:en_d]

# Select area

extents = np.array((-10.05, 2.05, 53.98, 61.02))
sel_riv = (riv_lon > extents[0]) & (riv_lon < extents[1]) & (riv_lat > extents[2]) & (riv_lon < extents[3])

riv_run = riv_run[:, sel_riv]
riv_lon = riv_lon[sel_riv]
riv_lat = riv_lat[sel_riv]

riv_sum = np.ma.sum(riv_run, axis=1)

print(riv_sum.shape, e_run.shape, across_run.shape)

st_d = np.nonzero(date_mn >= date_ep_mn[0])[0][0]
en_d = np.nonzero(date_mn >= date_ep_mn[-1])[0][0] + 1
total_trans = e_mn + p_mn + np.ma.sum(flux_mn[st_d:en_d, sel_riv], axis=1)
print(-np.ma.mean(total_trans), date_ep_mn[0])
print(np.ma.mean(np.ma.sum(across_mn, axis=1)))



fig1 = plt.figure(figsize=(10, 6))  # size in inches
ax1 = fig1.add_axes([0.1, 0.69, 0.85, 0.27])
ax2 = fig1.add_axes([0.1, 0.37, 0.85, 0.27])
ax3 = fig1.add_axes([0.1, 0.05, 0.85, 0.27])

ax1.plot(run_ep_date, across_run[:, 0], label='Irish Sea') # m3/s
ax1.plot(run_ep_date, across_run[:, 1] + across_run[:, 2], label='West')
ax1.plot(run_ep_date, across_run[:, 3], label='North')
ax1.plot(run_ep_date, across_run[:, 4] + across_run[:, 5], label='North Sea')

ax1.plot(run_ep_date, np.ma.sum(across_run, axis=1), 'k', lw=2, label='Sum Flux')

ax2.plot(run_ep_date, e_run, label='Evap.') # m3/s
ax2.plot(run_ep_date, p_run, label='Precip.')
ax2.plot(run_ep_date, e_run + p_run, 'k', lw=2, label='P + E')

ax3.plot(riv_date, riv_sum, label='River') # m3/s
ax3.plot(riv_date, (np.ma.sum(across_run, axis=1)), label='Total Flux')
ax3.plot(run_ep_date, vol_run, label='Volume Change')
ax3.plot(riv_date, riv_sum + e_run + p_run + vol_run + (np.ma.sum(across_run, axis=1)), 'k', lw=2, label='Total Budget')


ax1.set_ylabel('Freshwater Transport (m$^{3}$/s)')
ax2.set_ylabel('Surface flux (m$^{3}$/s)')
ax3.set_ylabel('River flux (m$^{3}$/s)')

ax1.legend(loc='lower right')
ax2.legend(loc='lower right')
ax3.legend(loc='lower right')




fig2 = plt.figure(figsize=(10, 6))  # size in inches
ax1 = fig2.add_axes([0.11, 0.77, 0.85, 0.19])
ax2 = fig2.add_axes([0.11, 0.53, 0.85, 0.19])
ax3 = fig2.add_axes([0.11, 0.29, 0.85, 0.19])
ax4 = fig2.add_axes([0.11, 0.05, 0.85, 0.19])


fw_bound = -(riv_sum + e_run + p_run) + vol_run

ax1.plot(run_ep_date, e_run, label='Evap.') # m3/s
ax1.plot(run_ep_date, p_run, label='Precip.')
ax1.plot(run_ep_date, e_run + p_run, 'k', lw=2, label='P + E')

ax2.plot(riv_date, riv_sum, 'k', lw=2, label='River') # m3/s

ax3.plot(run_ep_date, vol_run, 'k', lw=2)

ax4.plot(riv_date, fw_bound, 'k', lw=2)

ax1.set_ylabel('Surface flux (m$^{3}$/s)')
ax2.set_ylabel('River flux (m$^{3}$/s)')
ax3.set_ylabel('Freshwater\nvolume change (m$^{3}$/s)')
ax4.set_ylabel('Freshwater transport\n(m$^{3}$/s)')

ax1.set_xlim([dt.datetime(1993, 1, 1), dt.datetime(2020, 1, 1)])
ax2.set_xlim([dt.datetime(1993, 1, 1), dt.datetime(2020, 1, 1)])
ax3.set_xlim([dt.datetime(1993, 1, 1), dt.datetime(2020, 1, 1)])
ax4.set_xlim([dt.datetime(1993, 1, 1), dt.datetime(2020, 1, 1)])

ax1.legend(loc='lower right')

print('Boundary', np.ma.mean(fw_bound), np.ma.std(fw_bound))
print('Surface', np.ma.mean(e_run + p_run), np.ma.std(e_run + p_run))
print('River', np.ma.mean(riv_sum), np.ma.std(riv_sum))
print('Vol', np.ma.mean(vol_run), np.ma.std(vol_run))

ax1.annotate('(a)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax2.annotate('(b)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax3.annotate('(c)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)
ax4.annotate('(d)', (0.05, 0.9), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w"), zorder=105)


fig1.savefig('./Figures/fw_budget.png')
fig2.savefig('./Figures/fw_budget_flux.png')


