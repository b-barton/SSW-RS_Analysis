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
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import math
import calendar



match_file = './Indices/vel_index.npz'
obs_file = './Indices/vel_match.npz'
mod_file = './Indices/velocity_model.npz'



data = np.load(obs_file, allow_pickle=True)
prof_obs = data['prof_obs'] # wrong
lat_obs = data['lat_obs']
lon_obs = data['lon_obs']
dep_obs = data['dep_obs']
u_obs = data['u_obs']
v_obs = data['v_obs']
date_obs = data['date_obs']
data.close()

u_obs = np.ma.masked_where(u_obs == -9999, u_obs)
v_obs = np.ma.masked_where(v_obs == -9999, v_obs)
dep_obs = np.ma.masked_where(dep_obs >= 9999, dep_obs)

data = np.load(match_file, allow_pickle=True)
o_date = data['o_date']
o_prof = data['o_prof']
data.close()

data = np.load(mod_file)
u_mod = data['u_mod']
v_mod = data['v_mod']
data.close()

u_mod = np.ma.masked_where(u_mod == -9999, u_mod)
v_mod = np.ma.masked_where(v_mod == -9999, v_mod)

u_mod = np.ma.masked_where(u_obs.mask, u_mod)
v_mod = np.ma.masked_where(v_obs.mask, v_mod)
u_obs = np.ma.masked_where(u_mod.mask, u_obs)
v_obs = np.ma.masked_where(v_mod.mask, v_obs)


dtmp = np.zeros((len(prof_obs)), dtype=object)

for i in range(0, len(o_prof)):
  ind = prof_obs == o_prof[i]
  dtmp[ind] = date_obs[i]

# sort data based on date

ref = dt.datetime(1990, 1, 1)
o_days = np.zeros((len(date_obs)))
ddays = np.zeros((len(dtmp)))
for i in range(len(date_obs)):
  tmp = date_obs[i] - ref
  o_days[i] = tmp.days + (tmp.seconds / (24 * 60 * 60))
for i in range(len(dtmp)):
  tmp = dtmp[i] - ref
  ddays[i] = tmp.days + (tmp.seconds / (24 * 60 * 60))

# sort based on date, profile and depth
# np.lexsort
o_sort_ind = np.argsort(o_days)
sort_ind = np.argsort(ddays)

date_obs = date_obs[o_sort_ind]
o_prof = o_prof[o_sort_ind]
lat_obs = lat_obs[o_sort_ind]
lon_obs = lon_obs[o_sort_ind]

prof_obs = prof_obs[sort_ind]
dep_obs = dep_obs[sort_ind]
u_obs = u_obs[sort_ind]
v_obs = v_mod[sort_ind]
u_mod = u_mod[sort_ind]
v_mod = v_mod[sort_ind]


mnt_o = np.array([d.month for d in date_obs])
mnt_d = np.array([d.month for d in dtmp])

ind1 = ((mnt_o <= 4) | (mnt_o >= 11))
ind2 = ((mnt_o > 4) & (mnt_o < 11))

# Speed Direction

spd_obs = (u_obs ** 2 + v_obs ** 2) ** 0.5
drc_obs = np.ma.arctan2(v_obs, u_obs) # y, x
drc_obs = drc_obs * (180 / np.pi)

spd_mod = (u_mod ** 2 + v_mod ** 2) ** 0.5
drc_mod = np.ma.arctan2(v_mod, u_mod) # y, x
drc_mod = drc_mod * (180 / np.pi)


# Polar bar

dbin_s = 11.25

drc_obs_b = drc_mod * 1
drc_mod_b = drc_mod * 1

drc_obs_b[drc_obs_b > (360 - dbin_s)] = drc_obs_b[drc_obs_b > (360 - dbin_s)] - 360
drc_mod_b[drc_mod_b > (360 - dbin_s)] = drc_mod_b[drc_mod_b > (360 - dbin_s)] - 360



# Calculate histogram for windroses

for pl_num in range(2):

  fig1 = plt.figure(figsize=(7,10))

  for month in range(1,7):
    month2 = (month * 2) - 1
    month3 = (month * 2)
    mtext = (calendar.month_name[month2] + '-' 
          +  calendar.month_name[month3] + '\n')
    if month == 1:
      parea1 = [0.05,0.68,0.3,0.3]
      parea2 = [0.42,0.73,0.02,0.2]
    elif month == 2:
      parea1 = [0.57,0.68,0.3,0.3]
      parea2 = [0.94,0.73,0.02,0.2]
    elif month == 3:
      parea1 = [0.05,0.39,0.3,0.3]
      parea2 = [0.42,0.44,0.02,0.2]
    elif month == 4:
      parea1 = [0.57,0.39,0.3,0.3]
      parea2 = [0.94,0.44,0.02,0.2]
    elif month == 5:
      parea1 = [0.05,0.10,0.3,0.3]
      parea2 = [0.42,0.15,0.02,0.2]
    elif month == 6:
      parea1 = [0.57,0.10,0.3,0.3]
      parea2 = [0.94,0.15,0.02,0.2]
    ind = (mnt_d == month2) | (mnt_d == month3)
    if pl_num == 0:
      spd_hist = spd_obs[ind]
      dir_hist = drc_obs[ind]
    else:
      spd_hist = spd_mod[ind]
      dir_hist = drc_mod[ind]
    bins1 = np.array([-1,0.1,0.2,0.4,0.6,0.8,1.0,50,100])
    bins2 = dbin_s * np.arange(-1, 33, 2)
  
    in_hist = np.vstack((spd_hist,dir_hist)).T
    H, edges = np.histogramdd(in_hist, bins=[bins1,bins2])
  
    H = H*100.0/np.sum(H)

    # Plot the rose

    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=7)

    ax1 = fig1.add_axes(parea1, polar=True)

    theta = (90.0 - 22.5 * np.arange(0, 16) - 7.5) % 360
    theta = theta * 2 * np.pi/360.0
    mywidth = 15.0 * 2 * np.pi/360.0

    radii1 = np.sum(H[0,:]) / 16
    bar_all = {}
    bar_all['0'] = ax1.bar(0, radii1, width=3 * np.pi, bottom = 0.0, 
          edgecolor='b')
    radii2 = H[1,:]

    for ind in range(1, 7):
      bar_all[str(ind)] = ax1.bar(theta, radii2, width=mywidth, bottom = radii1)
      radii1 = radii2 + radii1
      radii2 = H[ind+1,:]
  
    ax1.set_xticklabels(['E','NE','N','NW','W','SW','S','SE'])
    ax1.set_ylim([0,42.0])
    ax1.set_yticks([10,20,30,40])
    ax1.set_yticklabels(['10%','','30%',''])
    ax1.set_title(mtext,fontsize=12)

    for ind in range(0, 7):
      for bar_tmp in bar_all[str(ind)]:
        bar_tmp.set_facecolor(cm.jet(ind/7.))
        bar_tmp.set_edgecolor(cm.jet(ind/7.))
        bar_tmp.set_linewidth(0)

    sumH = np.sum(H,axis=1) 

    plt.rc('ytick',labelsize=7)

    ax2 = fig1.add_axes(parea2,frame_on=False)

    bar_all2={}
    hgt2 = 0
    for ind in range(0, 8):
      hgt1 = sumH[ind]
      bar_all[str(ind)] = ax2.bar(0.1, hgt1, width=0.8, bottom = hgt2)
      hgt2 = hgt1 + hgt2

    for ind in range(0, 8):
      for bar_tmp in bar_all[str(ind)]:
        bar_tmp.set_facecolor(cm.jet(ind / 7.))
        bar_tmp.set_edgecolor(cm.jet(ind / 7.))
        bar_tmp.set_linewidth(0)

    ax2.set_xlim([0.1, 0.9])
    ax2.set_ylim([0, 100])
    ax2.set_xticks([])
    ax2.tick_params(direction='out')
    ax2.set_yticks([0,10,20,30,40,50,60,70,80,90,100])
    ax2.set_yticklabels(['0%','','','','','50%','','','','','100%'])
    ax2.get_yaxis().tick_left()

# Plot the legend

  ax3 = fig1.add_axes([0.025,0.017,0.95,0.06],frame_on=False,aspect='equal')
  ax3.set_xticks([])
  ax3.set_yticks([])
  ax3.set_title('Current speed\n(m/s)',fontsize=10)
  
  ax3.set_xlim([0,18.5])
  ax3.set_ylim([0,1.4])
  
  labs = ['< 0','0 - 0.2','0.2 - 0.4','0.4 - 0.6','0.6 - 0.8','0.8 - 1.0','> 1.0']

  for ind in range(0,7):

    offset = (ind)*2.6+0.1
    ax3.fill(np.array([0.1,0.1,0.9,0.9,0.1])+offset,np.array([0.1,0.9,0.9,0.1,0.1]),facecolor=cm.jet(ind/7.),linewidth=0)
    ax3.text(1.05+offset,0.5,labs[ind],fontsize=9,va='center',ha='left')

  if pl_num == 0:
    fig1.savefig('./Figures/rose_obs_vel.png',dpi=100)
  if pl_num == 1:
    fig1.savefig('./Figures/rose_mod_vel.png',dpi=100)




#Plot

fig2 = plt.figure(figsize=(10,8))

ax1 = fig2.add_axes([0.1, 0.55, 0.8, 0.35])
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.35])

nspd = 0
print(len(o_prof))
for i in range(0, len(o_prof)):
  print(i / len(o_prof) *100, '%')
  ind = prof_obs == o_prof[i]

  if i == 0:
    ax1.plot(dtmp[ind], spd_obs[ind], 'k.', label='Observations')
    ax1.plot(dtmp[ind], spd_mod[ind], 'r.', label='Model')
  else:
    ax1.plot(dtmp[ind], spd_obs[ind], 'k.')
    ax1.plot(dtmp[ind], spd_mod[ind], 'r.')

  ax2.plot(dtmp[ind], drc_obs[ind], 'k.')

  ax2.plot(dtmp[ind], drc_mod[ind], 'r.')

  nspd = nspd + int(np.ma.count(spd_obs[ind]) > 0)

ax1.legend(loc='upper right')

ax1.set_ylabel('Speed (m/s)')
ax2.set_ylabel('Direction ($^{\circ}$)')

ax1.annotate('(a) {:.0f} Drifter points'.format(nspd), (0.05, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), zorder=105)




fig2.savefig('./Figures/all_vel_time.png')

