#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:33:16 2026

@author: fedora
"""



import os
import sys
import json
import numpy as np
import pandas as pd

### !!!!
app = '_1'
### !!!!

if len(sys.argv)>1:
    config_name_paths = sys.argv[1]
else:
    config_name_paths = '../paths_F5.json'

with open(config_name_paths, 'r') as file:
    configs_paths = json.load(file)

dir_data = configs_paths['dir_res']
dir_res = configs_paths['dir_LC']
catalog_path = configs_paths['dir_cat'] + configs_paths['cat_name']
G = pd.read_csv(catalog_path,  names=['ra','dec','mag','source_id'], sep=' ')

dir_JD =  dir_res + 'JD/'
dir_Mags = dir_res + 'Mags/'
dir_Stat = dir_res + 'Stat/'

dir_RLC = dir_res + 'RLC/'
if not os.path.isdir(dir_RLC):
    os.mkdir(dir_RLC)

files = pd.Series(os.listdir(dir_Mags))
ymu = files.str[5:-4].values


print('\n            RLC_month_obj ---> RLC_obj            \n')
for ym_i in ymu:
    print(ym_i)
    Mags = np.load(dir_Mags + f'Mags_{ym_i}.npy', allow_pickle=True).tolist()
    JD = np.load(dir_JD + f'JD_{ym_i}.npy', allow_pickle=True)
    lc_list = os.listdir(dir_RLC)
    for g_inx, mag in Mags.items():
        print(g_inx, end=', ')
        mag = Mags[g_inx]
        name = f'RLC_{g_inx}.npy'
        path = dir_RLC + name
        if name in lc_list:
            RLC = np.load(path)            
        else:
            RLC = np.empty((0, 2))
            
        u = ~np.isnan(mag)
        if np.any(u):
            RLC_add = np.array([JD[u], mag[u]]).T
            RLC = np.concatenate((RLC, RLC_add))
            
            np.save(path, RLC)



