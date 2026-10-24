#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:19:53 2026

@author: fedora
"""

dir_CLC = '/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F4/CLC/' #CLCF/'
def getSLC(g_inx):
    file = dir_CLC + 'CLC_' +  str(g_inx) + '.npy'
    LC = np.load(file).T
    JD = LC[6]

    n = np.argsort(np.sum(np.isnan(LC), axis=1)[:-2])[:3]
    mag =np.nanmedian(LC[n], axis=0) # LC[1] #
    mag = sigma_clip(mag, 3, 1)
    u = np.isfinite(mag)    
    f = 10**(-(mag[u]-1)/2.5) + 1
    JD = JD[u]
    t = (JD - np.min(JD)) * 24
    LC = pd.DataFrame({'JD':JD,'t':t,'f':f})
    
    LC.to_csv(f'{dir_SLC}{ver_slc}/SLC_{g_inx}.csv')
   # except ValueError:
    #    print(f'VAL E {g_inx}')

dir_SLC = '/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F4/SLC/'
ver_slc = 'SLC_F_st3max_sc31'
G_inxu, files = getGInxLC(dir_CLC)

tasks = [(g_inx,) for g_inx in G_inxu] 
results = parallel_map(getSLC, tasks, max_workers=7, chunksize=1)

#10042, 43, 44, 45, 46, 48, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62

dir_SLC = '/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F5/SLC/SLC_F_3Nmax6/'
G_inx, files = getGInxLC(dir_SLC)


for file in files:
    SLC = pd.read_csv(dir_SLC + file, index_col=0)
    SLC = SLC.dropna(subset=['f'])
    SLC.to_csv(dir_SLC + file)
    print(file, end=',')

np.load(file, allow_pickle=True)



'''
import numpy as np
import matplotlib.pyplot as plt

dir_CLC = '/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F4/CLC/'
g_inx = 3021
file = dir_CLC + 'CLC_' +  str(g_inx) + '.npy'
LC = np.load(file).T
JD = LC[6]
for i in range(5):
   mag = LC[i]
   plt.plot(JD, mag + i, '.')

'''
'''

for g_inx in 23,   386,  2300,  2694,  3355,  4041,  5543,  6469,  7601, 9588,  9638,  9829, 10142, 11417, 13635:
    file = dir_CLC + 'CLC_' +  str(g_inx) + '.npy'
    LC = np.load(file).T
    JD = LC[6]
    plt.figure()
    std = np.nanstd(LC[0])
    dm = std * 4
    for i in range(6):
       mag = LC[i]
       plt.plot(JD, mag + i * dm, '.')
    plt.title(g_inx)
    

g_inx = 23

for g_inx in 5853,  352, 7616, 6579:
    file = dir_RLC + 'RLC_' +  str(g_inx) + '.npy'
    LC = np.load(file).T
    plt.figure()
    plt.plot(LC[0], LC[1], '.')


'''








