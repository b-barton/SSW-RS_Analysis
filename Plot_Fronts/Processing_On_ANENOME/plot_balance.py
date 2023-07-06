#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

fname1 = '/dssgfs01/working/benbar/SSW_RS/run/Momentum_v1.2_1998_12/balance.2d'
fname2 = '/dssgfs01/working/benbar/SSW_RS/run/Momentum_v1.2_1998_12/header_balance.txt'


with open(fname1, 'r') as f:
  lines = f.readlines()

num_cell = 200

head = 37
head = 42
head = 52
#print(lines[head-1])

#******, Adv_u, Adv_v, Adf_x, Adf_y, Baroclinic_PGF_x, Baroclinic_PGF_y, Coriolis_x, Coriolis_y, Barotopic_PGF_x, Barotropic_PGF_y, Diff_x, Diff_y, Stress_u, Stress_v, du_dt, dv_dt, Divergence_x, Divergence_y, delev_dt, I, Num_Balance

n_times = int((len(lines) - head) / 2) + 1

adv_u = np.zeros((n_times, num_cell))
adv_v = np.zeros((n_times, num_cell))
clin_pgf_x = np.zeros((n_times, num_cell))
clin_pgf_y = np.zeros((n_times, num_cell))
cor_x = np.zeros((n_times, num_cell))
cor_y = np.zeros((n_times, num_cell))
trop_pgf_x = np.zeros((n_times, num_cell))
trop_pgf_y = np.zeros((n_times, num_cell))
diff_x = np.zeros((n_times, num_cell))
diff_y = np.zeros((n_times, num_cell))
stress_u = np.zeros((n_times, num_cell))
stress_v = np.zeros((n_times, num_cell))
du_dt = np.zeros((n_times, num_cell))
dv_dt = np.zeros((n_times, num_cell))
div_x = np.zeros((n_times, num_cell))
div_y = np.zeros((n_times, num_cell))
date_list = np.zeros((n_times), dtype=object)
ref = dt.datetime(1998, 12, 31)

count = 0
for i in range(head, len(lines)):
  parts = lines[i].split()
  if (i % 2) == 0:
    num_cell = 150
    asterix = 0
  else:
    num_cell = 50
    count = count + 1
    asterix = 1

  for c in range(num_cell):
    ic = (c * 19) - asterix
    adv_u[count, c] = float(parts[1 + ic])
    adv_v[count, c] = float(parts[2 + ic])
    clin_pgf_x[count, c] = float(parts[5 + ic])
    clin_pgf_y[count, c] = float(parts[6 + ic])
    cor_x[count, c] = float(parts[7 + ic])
    cor_y[count, c] = float(parts[8 + ic])
    trop_pgf_x[count, c] = float(parts[9 + ic])
    trop_pgf_y[count, c] = float(parts[10 + ic])
    diff_x[count, c] = float(parts[11 + ic])
    diff_y[count, c] = float(parts[12 + ic])
    stress_u[count, c] = float(parts[13 + ic])
    stress_v[count, c] = float(parts[14 + ic])
    du_dt[count, c] = float(parts[15 + ic])
    dv_dt[count, c] = float(parts[16 + ic])
    div_x[count, c] = float(parts[17 + ic])
    div_y[count, c] = float(parts[18 + ic])
    date_list[count] = ref + dt.timedelta(seconds=(count) * 18)


# Plot

fig1 = plt.figure(figsize=(10, 8))
axes = fig1.subplots(nrows=3, ncols=2)

n_cell = 0
axes[0, 0].plot(date_list, adv_u[:, n_cell])
axes[0, 0].plot(date_list, adv_v[:, n_cell])
axes[0, 0].plot(date_list[::200], adv_u[::200, n_cell])
axes[0, 0].plot(date_list[::200], adv_v[::200, n_cell])

axes[0, 1].plot(date_list, cor_x[:, n_cell])
axes[0, 1].plot(date_list, cor_y[:, n_cell])
axes[0, 1].plot(date_list[::200], cor_x[::200, n_cell])
axes[0, 1].plot(date_list[::200], cor_y[::200, n_cell])

axes[1, 0].plot(date_list, trop_pgf_x[:, n_cell])
axes[1, 0].plot(date_list, trop_pgf_y[:, n_cell])
axes[1, 0].plot(date_list[::200], trop_pgf_x[::200, n_cell])
axes[1, 0].plot(date_list[::200], trop_pgf_y[::200, n_cell])

axes[1, 1].plot(date_list, clin_pgf_x[:, n_cell])
axes[1, 1].plot(date_list, clin_pgf_y[:, n_cell])
axes[1, 1].plot(date_list[::200], clin_pgf_x[::200, n_cell])
axes[1, 1].plot(date_list[::200], clin_pgf_y[::200, n_cell])

#axes[1, 1].plot(date_list, diff_x[:, n_cell])
#axes[1, 1].plot(date_list, diff_y[:, n_cell])
#axes[1, 1].plot(date_list[::200], diff_x[::200, n_cell])
#axes[1, 1].plot(date_list[::200], diff_y[::200, n_cell])

axes[2, 0].plot(date_list, stress_u[:, n_cell])
axes[2, 0].plot(date_list, stress_v[:, n_cell])
axes[2, 0].plot(date_list[::200], stress_u[::200, n_cell])
axes[2, 0].plot(date_list[::200], stress_v[::200, n_cell])

#axes[2, 0].plot(date_list, div_x[:, n_cell])
#axes[2, 0].plot(date_list, div_y[:, n_cell])

axes[2, 1].plot(date_list, du_dt[:, n_cell])
axes[2, 1].plot(date_list, dv_dt[:, n_cell])
axes[2, 1].plot(date_list[::200], du_dt[::200, n_cell])
axes[2, 1].plot(date_list[::200], dv_dt[::200, n_cell])

#axes[2, 1].plot(date_list, -trop_pgf_x + cor_x + stress_u)# + diff_x + adv_u)
#axes[2, 1].plot(date_list, -trop_pgf_y + cor_y + stress_v)# + diff_y + adv_v)


axes[0, 0].set_ylabel('Advection')
axes[0, 1].set_ylabel('Coriolis')
axes[1, 0].set_ylabel('BTrop PGF')
axes[1, 1].set_ylabel('BClin PGF')
axes[2, 0].set_ylabel('Stress (Surf + Bot)')
#axes[2, 1].set_ylabel('Divergence')
axes[2, 1].set_ylabel('dVel_dt')

axf = axes.flatten()
for ax in axf:
#  ax.set_xlim([dt.datetime(1998, 12, 30, 23, 50), 
#                      dt.datetime(1998, 12, 31, 3, 10)])

  ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))   #to get a tick every 15 minutes
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))     #optional formatting
#fig1.autofmt_xdate()

fig1.savefig('./Figures/balance.png')
