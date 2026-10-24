#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:42:53 2026

@author: fedora
"""

import numpy as np
import pandas as pd
from funcs import getGInxLC


dirs = ["/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F5/RLC_2024/",
"/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F5/RLC_2025/",
"/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F5/RLC_2026/"]

dir_save = '/home/fedora/astronomy/STEP/main/results/lightcurves/V7/F5/RLC/'

Gs = []
Fs = []
for d in dirs:
    g, f = getGInxLC(d)
    Gs.append(g)
    Fs.append(f)
    
G_inx = np.unique(np.concatenate(Gs))

DF = pd.DataFrame(index=G_inx)
for i in range(3):
    DF.loc[DF.index.isin(Gs[i]), i] = True
DF[DF.isna()] = False


for ij, g_inx in enumerate(G_inx):
    r = DF.loc[g_inx]
    inx = r[r==True].index
    RLC = []
    name = f'RLC_{g_inx}.npy'
    for i in inx:
        rlc = np.load(f'{dirs[i]}{name}')
        RLC.append(rlc)
    RLC = np.concatenate(RLC)
    np.save(f'{dir_save}{name}', RLC)
    if ij%100==0:
        print(ij, end=', ')