#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:02:05 2026

@author: fedora
"""
import os
import numpy as np
import pandas as pd
from funcs import readConfig, getGInxLC, getRLC, sigma_clip, parallel_map

import sys


import matplotlib.pyplot as plt
def get_Refs_jd(g_inx, N=3):
    # N%2=0
    jdf = K_df[g_inx].dropna().index
    G_st_i = G_st0[~np.isin(G_st0, g_inx)]
    G_br_i0 = G_br[~np.isin(G_br, g_inx)]

    mag0 = getRLC(dir_RLC, g_inx, JD)
    mag0 = sigma_clip(mag0, 5, 1)
    
    Y0 = [[] for _ in range(N)]
    Y1 = [[] for _ in range(N)]
    JD_i = []
    
    jdf = K_df[g_inx].dropna().index
    for ij, jd in enumerate(jdf):
        # get refs 
        K_jd = K_df.loc[jd]
        x0 = K_jd[g_inx]
        # standart
        x_st = K_jd[G_st_i].values
        d_x = abs(x0 - x_st)
        G_st_st0 = G_st_i[np.argsort(d_x)][:5]
        # bright
        G_br_i = G_br_i0[~np.isin(G_br_i0, G_st_st0)]
        x_br = K_jd[G_br_i].values
        d_x = abs(x0 - x_br)
        G_st_br = G_br_i[np.argsort(d_x)][:5]

        u_jd = JDf==jd
        mag_jd = mag0[u_jd]        
        t = t0[u_jd]
        jd_i = JD[u_jd]
        
        C = np.array([])
        G_st_all = np.concatenate((G_st_st0, G_st_br))
        for g_st in G_st_all:
            mag_st_jd = getRLC(dir_RLC, g_st, jd_i)
            u_g = np.isfinite(mag_jd)
            u_s = np.isfinite(mag_st_jd)
            u = u_g & u_s
            
            if np.sum(u) > 0:
                c = np.corrcoef(mag_jd[u], mag_st_jd[u])[0][1]
            else:
                c = np.nan
            C = np.append(C,c)     
        if np.sum(np.isnan(C)) < len(C):
            C[np.isnan(C)] = -1
            sts = G_st_all[np.argsort(C)[::-1]][:N]
            if len(sts)<N:
                sts = np.concatenate((sts, -np.ones(N - len(sts))))
        else:
            sts = G_st_st0[:3]

        DF = []    
        for g_st in sts:
            mag_st_jd = getRLC(dir_RLC, g_st, JD)[u_jd]
            dm_jd = mag_jd - mag_st_jd
            ### detrending
            y = sigma_clip(dm_jd.copy(), 1, 3)
            u_fin = np.isfinite(y)
            y = y[u_fin]
            x = t[u_fin]
            if sum(u_fin)>5:
                p = np.polyfit(x, y, 1)
                fit = np.polyval(p, t)
                y_fit = dm_jd - fit
                y_fit = y_fit - np.nanmedian(y_fit)
      
                print(np.polyfit(x, y - np.polyval(p, x), 1)[0])
                '''
                plt.figure()
                plt.plot(t, dm_jd, '.')
                plt.plot(t, fit)
                plt.plot(t, y_fit, '.')
                '''
                df = y_fit
            DF.append(df)
            
        for i in range(N):
            Y0[i].append(DF[i])
            Y1[i].append(DF[i+N])
        if ij%10==0:
            print(ij, end=',')
        JD_i.append(jd_i)
    
    JD_res = np.concatenate(JD_i)     
    LC = [JD_res] 
    for i in range(N):
        LC.append(np.concatenate(Y0[i]))
    for i in range(N):
        LC.append(np.concatenate(Y1[i]))
     
    LC = pd.DataFrame(DF).T
    LC['JD'] = JD
    LC['t'] = t0
    LC = LC.dropna(subset=LC.columns[:-2], how='all')
    np.save(f'{dir_res}CLC_{g_inx}.npy', LC)



if len(sys.argv)>1:
    config_name = sys.argv[1]
else:
    config_name = 'configs/createCLC_F5.json'
#############################################################
# Preparation
#############################################################
#### read configs
(dir_RLC, dir_res, cat_path, refs_path,
 st_mag_max, st_W_quantile, st_ro_min,
 G_st_del,
 field, ver_LC) = readConfig(config_name)

G_inx, files = getGInxLC(dir_RLC)
F = pd.read_csv(cat_path, sep=' ', names=['ra', 'dec', 'mag', 'source_id'])
G_st0 = np.load('/home/fedora/astronomy/STEP/main/results/catalogs/refstars/G_st0_F5.npy')

MAG = F.loc[(F.mag>12.5) & (F.mag<14.5), 'mag'].sort_values()
G_br = MAG[MAG.index.isin(G_inx)].index

W = []
for file in files:
    W.append(os.path.getsize(dir_RLC + file))
n_max = np.argmax(W)
w_max = W[n_max]
W = pd.Series(W, index=G_inx, name='n')
path = dir_RLC + f'RLC_{str(G_inx[n_max])}.npy' 
RLC0 = np.load(path)
JD = RLC0[:, 0]
t0 = (JD - np.min(JD)) * 24
JDf = np.floor(JD)
JDfu = np.unique(JDf)



K_df = pd.read_csv('/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F5/K_df_F5.csv', index_col=0)
K_df.columns = K_df.columns.astype(int)






'''
tasks = [(g_inx,) for g_inx in G_inx]
results = parallel_map(calibrate, tasks, max_workers=7, chunksize=1)
'''
















