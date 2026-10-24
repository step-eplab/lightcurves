#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 14:13:15 2026

@author: fedora
"""


import os
import sys
import numpy as np
import pandas as pd

from funcs import readConfig, getGInxLC
    
if len(sys.argv)>1:
    config_name = sys.argv[1]
else:
    config_name = 'configs/createCLC_F5.json'
#############################################################
# Preparation
#############################################################
#### read configs
(dir_LC, dir_res, cat_path, refs_path,
 st_mag_max, st_W_quantile, st_ro_min,
 G_st_del,
 field, ver_LC) = readConfig(config_name)

F = pd.read_csv(cat_path, sep=' ', names=['ra', 'dec', 'mag', 'source_id'])

dir_RLC = dir_LC + ''

MAG = F.loc[(F.mag>12), 'mag'].sort_values()
#############################
G_inx, files = getGInxLC(dir_RLC)
MAG = MAG[MAG.index.isin(G_inx)]
G_br = MAG[(MAG>12.5) & (MAG<15)].index

W = []
for file in files:
    W.append(os.path.getsize(dir_RLC + file))
n_max = np.argmax(W)
w_max = W[n_max]
W = pd.Series(W, index=G_inx, name='n')

path = dir_RLC + f'RLC_{str(G_inx[n_max])}.npy'   
RLC = np.load(path)
JD = RLC[:, 0]
t0 = (JD - np.min(JD)) * 24
JDf = np.floor(JD)
JDfu = np.unique(JDf)

R = pd.read_csv(refs_path, index_col=0)
R = pd.merge(R, W, how='left', left_index=True, right_index=True).dropna(subset=['n'])
R = pd.merge(R, F['mag'], how='left', left_index=True, right_index=True )

'''
st_mag_max = 14.5
st_W_quantile = 0.9
st_ro_min = 0.01
'''

u1 = R.mag < st_mag_max
u2 = R['n'] > W.quantile(st_W_quantile)
u3 = R.ro_min > st_ro_min
u4 = ~R.index.isin(G_st_del)
cond = u1 & u2 & u3 & u4
for u in u1, u2, u3, u4, cond:
    print(sum(u))
    
G_st0 = R[cond].index

# g0 = 16

def getRLC(dir_RLC, g_inx, JD_max=[]):
    path = dir_RLC + f'RLC_{str(g_inx)}.npy'
    if os.path.isfile(path):
        RLC = pd.DataFrame(np.load(path), columns=['JD', 'mag'])
        RLC['f'] = 1 - (RLC['mag'] - RLC['mag'].median())
        if len(JD_max)>0:
            add = pd.DataFrame({'JD':JD_max})
            RLC = pd.merge(add, RLC, on='JD', how='left')
        return RLC['f'].values

import matplotlib.pyplot as plt
f0 = getRLC(dir_RLC, 1180, JD)
plt.plot(JD, f0, '.')



for j in range(int(len(G_st0)/10) + 1):
    plt.figure(figsize=(24,24))
    for i, g_st in enumerate(G_st0[(j * 10) : ((j+1) * 10)]):
        f = getRLC(dir_RLC, g_st, JD)
        y = f - f0
        y = y - np.nanmedian(y)
        plt.plot(JD, y + i, '.')
        plt.text(2460646, i, str(g_st))
    plt.xlim([2460645.086589193, 2460649.56156983])
    plt.ylim([-0.388906504208238259, 9.514177461244394])
    plt.grid()
    plt.tight_layout()

#245, 632, 1180, 1838, 2116, 2288

'''
plt.plot(F.ra, F.dec, '.')
plt.plot(F.loc[G_st0,'ra'], F.loc[G_st0,'dec'], '.')
'''

def sigma_clip(y, lvl=3, n=1):
    med = np.nanmedian(y)
    std = np.nanstd(y)
    u = np.zeros(len(y), dtype=bool)
    for i in range(n):
        u |= abs(y - med) > lvl * std
    y[u] = np.nan
    return y
def get_K(g_inx, dir_RLC, JD):
    f = getRLC(dir_RLC, g_inx, JD)
    f = sigma_clip(f, 3, 1)
    u_fin = np.isfinite(f)
    JDfu_i = np.unique(np.floor(JD[u_fin]))
    if len(JDfu_i)>0:
        JD_res = []
        K_res = []
        for jd in JDfu_i:
            jd_u = JDf==jd
            y = f[jd_u & u_fin]
            x = t0[jd_u & u_fin]
            if len(x)>0:
                p = np.polyfit(x, y, 3)
                k = np.sum(p)
                JD_res.append(jd.item())
                K_res.append(k.item())
    return JD_res, K_res


results = {}
for i, g_inx in enumerate(G_inx):
    jd, k = get_K(g_inx, dir_RLC, JD)
    results[g_inx] = [jd, k]
    if i%100==0:
        print(i, end=', ')


K_df = pd.DataFrame(index=JDfu, columns=G_inx)
for g_inx in G_inx:
    jd, k = results[g_inx]
    K_df.loc[np.isin(JDfu, jd), g_inx] = k
    
save_cats = '/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F5/'
#K_df.to_csv(save_cats + 'K_df_F5.csv')
#K_df = pd.read_csv(save_cats + 'K_df_F5.csv', index_col=0)
K_df.columns = K_df.columns.astype(int)


from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def _call_star(args):
    func, item = args
    return func(*item)

def parallel_map(func, iterable, max_workers=None, chunksize=1):
    """
    Универсальный multiprocessing.

    func         — функция для выполнения
    iterable     — аргументы задач (список кортежей)
    max_workers  — число процессов
    chunksize    — число задач на один пакет
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(tqdm(
            executor.map(
                _call_star,
                [(func, item) for item in iterable],
                chunksize=chunksize,
            ),
            total=len(iterable),
            desc="Processing",
        ))


def findRefs2(g_inx, N=5):
    jdf = K_df[g_inx].dropna().index
    G_st_i = G_st0[~np.isin(G_st0, g_inx)]
    G_br_i0 = G_br[~np.isin(G_br, g_inx)]
    G0 = []
    G1 = []
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
        
        G0.append(G_st_st0)
        G1.append(G_st_br)

    G0 = np.concatenate(G0)
    G1 = np.concatenate(G1)

    sts0 = pd.Series(np.array(list(G0)).flatten()).value_counts().index[:3]
    sts1 = pd.Series(np.array(list(G1)).flatten()).value_counts().index[:3]
    sts = np.concatenate((sts0,sts1))
    return g_inx, np.int32(sts)

    
tasks = [(g_inx,) for g_inx in G_inx]
results = parallel_map(findRefs2, tasks, max_workers=7, chunksize=1)

Refs = {}
for g_inx, sts in results:
    if sts is not None:
        Refs[g_inx] = sts
Refs = pd.DataFrame(Refs).T

Refs.to_csv(save_cats + 'Refs_Kdf_F5.csv')


def calibrate(g_inx):
    f0 = getRLC(dir_RLC, g_inx, JD)  
    sts = Refs.loc[g_inx]
    DF = []
    for g_st in sts:
        f = getRLC(dir_RLC, g_st, JD)
        df = f - f0
        df = sigma_clip(df, 5, 1)
        u_nan = np.isnan(df)
        for jd in JDfu:
            u_jd = JDf==jd
            y = df[u_jd & (~u_nan)]
            x = t0[u_jd & (~u_nan)]
            y = sigma_clip(y, 1, 3)
            
            u = ~np.isnan(y)
            if np.sum(u)>10:
                p = np.polyfit(x[u], y[u], 1)
                fit = np.polyval(p, t0[u_jd])
                y_fit = df[u_jd] - fit
                y_fit = y_fit - np.nanmedian(y_fit)
            else:
                y_fit = np.nan
            df[u_jd] = y_fit
        DF.append(df)
    LC = pd.DataFrame(DF).T
    LC['JD'] = JD
    LC['t'] = t0
    LC = LC.dropna(subset=LC.columns[:-2], how='all')
    np.save(f'{dir_res}CLC_{g_inx}.npy', LC)

    
tasks = [(g_inx,) for g_inx in G_inx]
results = parallel_map(calibrate, tasks, max_workers=7, chunksize=1)

#for g_inx in 47, 4041, 9588, 2300:
#    calibrate(g_inx)

'''
i  = 0
for file in ["/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F4/CLC/CLC_2000.npy",
"/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F4/CLC/CLC_672.npy",
"/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F4/CLC/CLC_5353.npy",
"/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F4/CLC/CLC_3484.npy",
"/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F4/CLC/CLC_412.npy",
"/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F4/CLC/CLC_10979.npy",
"/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F4/CLC/CLC_9499.npy"]:
    LC = np.load(file).T
    jd = LC[6]
    mag = LC[1]
    plt.plot(jd, mag + i, '.')
    i+=1

'''




















