#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:15:17 2026

@author: fedora
"""


import os
import sys
import json
import glob
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

for d in [dir_res, dir_JD, dir_Mags, dir_Stat]:
    if not os.path.isdir(d):
         os.mkdir(d)

LC_files = pd.Series(glob.glob(dir_data + '**/LC_*.csv', recursive=True))
LC_files = LC_files.sort_values(ignore_index=True)

Nights = LC_files.str[-12:-6]
Dates = Nights.str[:2] + '.' + Nights.str[2:4] + '.' + Nights.str[4:]
ym = Nights.str[:-2]
ymu = ym.unique()

print('\n\nSTART CREATE RAW LC\n\n')
print('month-year.unique: ', ymu)
### night2month
for ym_i in ymu:
    print(f'\n\n            READ RLCN {ym_i}            \n\n')

    u_ym = ym==ym_i
    N_nights_i = sum(u_ym)
    
    files = LC_files[u_ym]
    nights = Nights[u_ym]
    
    JD = np.array([])
    JDn = {}
    LCs = {}
    D = pd.DataFrame(columns=nights, index=G.index)
    for night, file in zip(nights, files):
        print(night, end=', ')
        LC = pd.read_csv(file, index_col=0)
        LCs[night] = LC
        jd = np.float64(LC.columns)
        JD = np.concatenate((JD, jd))
        JDn[night] = len(jd)

        inx = LC.index
        N = len(LC.columns)
        D[night] = N - LC.isna().sum(axis=1)
    
    D = D.dropna(axis=0, how='all')
    if N_nights_i > 10:
        D = D.loc[(len(nights) - D.isna().sum(axis=1)) > 3]
    
    print('\n\n            RLC_night_obj ---> RLC_month_obj            \n')
    print('')
    Mags = {}
    for i, g_inx in enumerate(D.index):
        mags = []
        for n in nights:
            if D.loc[g_inx, n] > 0:
                mags.append(LCs[n].loc[g_inx].to_numpy())
            else:
                mags.append(np.full(JDn[n], np.nan))
        Mags[g_inx] = np.concatenate(mags)
        
        if i%1000==0:
            print(i, end=',')
    D.loc[-1] = JDn

    print('\n\n            SAVE            \n')
    print('JD')
    np.save(dir_JD + f'JD_{ym_i}.npy', JD)
    print('Mags')
    np.save(dir_Mags + f'Mags_{ym_i}.npy', Mags)
    print('Stat')
    D.to_csv(dir_Stat + f'Stat_{ym_i}.csv')







'''
dir_RLC = '/home/fedora/astronomy/STEP/main/data/F5/LC/RLC/'
os.mkdir(dir_RLC)



print('\n            RLC_month_obj ---> RLC_obj            \n')
for ym_i in ymu:
    Mags = np.load(dir_Mags + f'Mags_{ym_i}.npy', allow_pickle=True).tolist()
    JD = np.load(dir_JD + f'JD_{ym_i}.npy', allow_pickle=True).tolist()
    
    lc_list = pd.Series(os.listdir(dir_RLC))
    
    for g_inx in Mags:
        name = 'RLC_' + str(g_inx) + '.csv'
        if name in lc_list:
            RLC = np.load(dir_RLC + name)            
        else:
            RLC = np.array([[], []])
        mag = Mags[g_inx]
        u = ~np.isnan(mag)
        if sum(u) > 0:
            RLC_add = np.array([JD[u], mag[u]]).T
            RLC = np.concatenate((RLC, RLC_add))
            
            
'''


'''
Stat = D.sum(axis=1)

G_mag = G.loc[(G.mag>12) & (G.mag<15)].index
Stat_mag = Stat.loc[G_mag[np.isin(G_mag, Stat.index)]]

G_st = Stat_mag[Stat_mag > np.percentile(Stat_mag, 95)].index


def sigma_clip(img, n=3):
    med = np.nanmedian(img)
    std = np.nanstd(img)
    u = abs(img - med) > n * std
    img[u] = np.nan
    return img

G_inx = []
mag = {}
for n, LC in zip(nights, LCs):
    print(n, end=',')
    mag[n] = {}
    for g_inx in LC.index:
        mag[n][g_inx] = LC.loc[g_inx].values


    
for i, g_inx in enumerate(mag):
    y = mag[3520] - mag[g_inx]
    y = y - np.nanmedian(y) + i * 0.5
    plt.plot(JD, y, '.')

std = []
mags = {}
for i, g_st in enumerate(G_st):
    y = Mags[3520] - Mags[g_st]
    for i in range(3):
        y = sigma_clip(y, 2)
    y = y - np.nanmedian(y)
    mags[g_st] = y
    std.append(np.nanstd(y))

inx2 = np.argsort(std)
G_inx = G_st[inx2]
for i, g_inx in enumerate(G_inx):
    y = mags[g_inx] + i * 0.1
    plt.plot(JD,  y, '.')
    plt.text(JD[0], y[0], str(G_inx[i]))
'''


































