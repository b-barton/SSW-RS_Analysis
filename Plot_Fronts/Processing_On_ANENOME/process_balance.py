#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

fname1 = '/dssgfs01/working/benbar/SSW_RS/run/Momentum1_v1.2_1998_12/balance1.2d'
fname2 = '/dssgfs01/working/benbar/SSW_RS/run/Momentum2_v1.2_1998_12/balance2.2d'
fname3 = '/dssgfs01/working/benbar/SSW_RS/run/Momentum3_v1.2_1998_12/balance3.2d'
fname4 = '/dssgfs01/working/benbar/SSW_RS/run/Momentum1_v1.2_1998_12/balance4.2d'
fname5 = '/dssgfs01/working/benbar/SSW_RS/run/Momentum2_v1.2_1998_12/balance5.2d'
fname6 = '/dssgfs01/working/benbar/SSW_RS/run/Momentum3_v1.2_1998_12/balance6.2d'
fname7 = '/dssgfs01/working/benbar/SSW_RS/run/Momentum3_v1.2_1998_12/balance7.2d'

head_fname = '/dssgfs01/working/benbar/SSW_RS/run/Momentum1_v1.2_1998_12/header_balance.txt'
out_dir = '/dssgfs01/scratch/benbar/Processed_Data/' 

def read_head(fname):
  with open(fname, 'r') as f:
    lines1 = f.readlines()
  for i in range(len(lines1)):
    if '******' in lines1[i][:7]:
      head = i * 1
      break
  return lines1, head

lines1, head1 = read_head(fname1)
lines2, head2 = read_head(fname2)
lines3, head3 = read_head(fname3)
lines4, head4 = read_head(fname4)
lines5, head5 = read_head(fname5)
lines7, head7 = read_head(fname7)
lines6, head6 = read_head(fname6)
lines = [lines1, lines2, lines3, lines4, lines5, lines6]
head = [head1, head2, head3, head4, head5, head6]
#lines = [lines1, lines2, lines3]#, lines4, lines5, lines6]
#head = [head1, head2, head3]#, head4, head5, head6]

print(head)

num_cell = 200

#print(lines[head-1])

#******, Adv_u, Adv_v, Adf_x, Adf_y, Baroclinic_PGF_x, Baroclinic_PGF_y, Coriolis_x, Coriolis_y, Barotopic_PGF_x, Barotropic_PGF_y, Diff_x, Diff_y, Stress_u, Stress_v, du_dt, dv_dt, Divergence_x, Divergence_y, delev_dt, I, Num_Balance

f_len = np.array([len(lines[0]) - head[0],
              len(lines[1]) - head[1],
              len(lines[2]) - head[2],
              len(lines[3]) - head[3],
              len(lines[4]) - head[4],
              len(lines[5]) - head[5]], dtype=int)
#              len(lines[6]) - head[6]], dtype=int)
print(f_len)

n_times = int(np.sum(f_len) / 2) + 1

# Divide by 200 for hourly
hourly = 1
if hourly:
  n_times = int(n_times / 200) + 1 

print(n_times)

adv = np.zeros((2, n_times, num_cell)) # u, v
adf = np.zeros((2, n_times, num_cell)) # u, v
clin_pgf = np.zeros((2, n_times, num_cell))
cor = np.zeros((2, n_times, num_cell))
trop_pgf = np.zeros((2, n_times, num_cell))
diff = np.zeros((2, n_times, num_cell))
stress = np.zeros((2, n_times, num_cell))
du_dt = np.zeros((2, n_times, num_cell))
div = np.zeros((2, n_times, num_cell))
date_list = np.zeros((n_times), dtype=object)
#ref = dt.datetime(1998, 12, 31)

# Use the end date because some of the start seems corrupted
ref = [dt.datetime(1999, 3, 1), dt.datetime(1999, 4, 30), dt.datetime(1999, 6, 29), dt.datetime(1999, 8, 28), dt.datetime(1999, 10, 27), dt.datetime(2000, 1, 1)]# dt.datetime(1999, 11, 1, 23), dt.datetime(2000, 1, 1)]

count = -1
for f in range(len(head)):
  date_count = -1
  for i in range(head[f], len(lines[f]), 400): # 200 * 2
    date_count = date_count + 1
    count = count + 1
    for l in range(2):
      i = i + l
      parts = lines[f][i].split()
      #if (i % 2) == 1:
      if parts[0] == '******':
        num_cell = 150
        asterix = 0
      else:
        num_cell = 50
        asterix = 1

      for c in range(num_cell):
        if asterix == 1:
          ct = c + 150
        else:
          ct = c
        ic = (c * 19) - asterix
        adv[0, count, ct] = float(parts[1 + ic])
        adv[1, count, ct] = float(parts[2 + ic])
        adf[0, count, ct] = float(parts[3 + ic])
        adf[1, count, ct] = float(parts[4 + ic])
        clin_pgf[0, count, ct] = float(parts[5 + ic])
        clin_pgf[1, count, ct] = float(parts[6 + ic])
        cor[0, count, ct] = float(parts[7 + ic])
        cor[1, count, ct] = float(parts[8 + ic])
        trop_pgf[0, count, ct] = float(parts[9 + ic])
        trop_pgf[1, count, ct] = float(parts[10 + ic])
        diff[0, count, ct] = float(parts[11 + ic])
        diff[1, count, ct] = float(parts[12 + ic])
        stress[0, count, ct] = float(parts[13 + ic])
        stress[1, count, ct] = float(parts[14 + ic])
        du_dt[0, count, ct] = float(parts[15 + ic])
        du_dt[1, count, ct] = float(parts[16 + ic])
        div[0, count, ct] = float(parts[17 + ic])
        div[1, count, ct] = float(parts[18 + ic])
        date_list[count] = ref[f] - dt.timedelta(hours=int((f_len[f] / 400) - date_count))
        #date_list[count] = ref + dt.timedelta(seconds=(count) * 18)

print(date_list[-int(np.sum(f_len[-3:]) / 400)-5:-int(np.sum(f_len[-3:]) / 400) + 5])
print(date_list[-int(np.sum(f_len[-2:]) / 400)-5:-int(np.sum(f_len[-2:]) / 400) + 5])
print(date_list[-int(f_len[-1] / 400)-5:-int(f_len[-1] / 400) + 5])
print(count)

np.savez(out_dir + 'momentum_transects.npz', adv=adv, adf=adf, clin_pgf=clin_pgf, cor=cor, trop_pgf=trop_pgf, diff=diff, stress=stress, du_dt=du_dt, div=div, date_list=date_list)

