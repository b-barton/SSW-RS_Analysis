#!/usr/bin/env python3


import numpy as np
import datetime as dt

out_dir = '../Processed_Data/'

data = np.load(out_dir + 'stack_baroclinic_vel_mn.npz', allow_pickle=True)
u = data['barotrop_u']
v = data['barotrop_v']
date_list = data['date_list']
lon = data['lon']
lat = data['lat']
tri = data['tri']

np.savez(out_dir + 'stack_barotropic_vel_mn.npz', barotrop_u=u, barotrop_v=v, date_list=date_list, lon=lon, lat=lat, tri=tri)


