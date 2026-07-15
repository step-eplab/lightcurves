#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:28:00 2026

@author: fedora
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import multiprocessing
from astropy.io import fits

def night2month(g_inx, dir_tmp, ym_i, DF):
    # create temp file for objects with month lenght
    JD = np.float64(DF.columns)
    g_inx_s = str(int(g_inx))
    g_dir = dir_tmp + g_inx_s + '/'
    g_name = g_dir + 'LC_' + g_inx_s +  '_' + ym_i + '.npy'
    m = DF.loc[g_inx]
    u = ~np.isnan(m) & (m < 0)
    jd = JD[u]
    m = m[u]
    
    LC = np.array([jd, m]).T
    if not os.path.isdir(g_dir):
        os.mkdir(g_dir)       
    np.save(g_name, LC)

    if g_inx%1000==0:
        print(g_inx, end=', ')

def monthObj2obj(S, g_inx, dir_tmp, dir_obj):
    # create RLC
    g_inx_s = str(g_inx)
    g_dir = dir_tmp + g_inx_s + '/'

    lc_list = pd.Series(os.listdir(g_dir))
    lc_list = lc_list[lc_list.str.contains('LC_' + g_inx_s + '_')].values
    if len(lc_list)>0:
        LC = np.load(g_dir + lc_list[0])
        for lc in lc_list[1:]:
            LC1 = np.load(g_dir + lc) 
            LC = np.concatenate((LC, LC1))
    else:
        LC = np.load(g_dir + lc_list)

    inx = np.argsort(LC[:,0])
    LC = LC[inx]

    S[g_inx] = len(LC)

    np.save(dir_obj + 'LC_' + g_inx_s + '.npy', LC)
    os.system('rm -r ' + g_dir)    
    return S

import json


def main():
    if len(sys.argv)>1:
        config_name_paths = sys.argv[1]
    else:
        config_name_paths = '../paths_F4.json'

    with open(config_name_paths, 'r') as file:
        configs = json.load(file)

    dir_data = configs['dir_res']
    dir_res = configs['dir_LC']
    
    dir_raw_m = dir_res + 'raw_months/'
    dir_obj = dir_res + 'RLC/'
    dir_tmp = dir_res + 'temp/'
    dir_stat = dir_res + 'stat/'
    
    for d in [dir_res, dir_raw_m, dir_obj, dir_tmp, dir_stat]:
        if not os.path.isdir(d):
             os.mkdir(d)
    
    LC_files = pd.Series(glob.glob(dir_data + '**/LC_*.csv', recursive=True))
    LC_files = LC_files.sort_values(ignore_index=True)
    print(LC_files)
    ym = LC_files.str[-10:-6]
    print('START CREATE RAW LC \n')
    print('!!! mag < 0 !!!')
    
    print(ym.unique())
    
    n = os.cpu_count()
    
    for ym_i in  ym.unique():
        u_ym = ym==ym_i
        G_inx_ym = []
        print(dir_raw_m + 'LC_' + ym_i + '.fits')
    
        if (sum(u_ym)>0) & (not os.path.isfile(dir_raw_m + 'LC_' + ym_i +  '.fits')):
            DF = pd.DataFrame(dtype=np.float32)
            i=0
            for file in LC_files[u_ym][:1]:
                print(file)
                LC = pd.read_csv(file, index_col='g_inx', dtype=np.float32)
                DF = pd.concat((DF, LC), axis=1)
    
                DF = DF.sort_index()        
                DF = DF.reindex(np.sort(DF.columns), axis=1)
                L = np.sum(~np.isnan(DF), 1)
                #!!!!!!!!!!!!!
                DF = DF[L>50]
                print(len(DF))
                if len(DF)>0:
                    mags = DF.values
                    inx = DF.index
                    jd = np.float64(DF.columns)
                    
                    jd_hdu = fits.ImageHDU(jd)
                    inx_hdu = fits.ImageHDU(np.int32(inx))
                    mags_hdu = fits.PrimaryHDU(mags)
                     
                    print('\n SAVE ' + ym_i + '\n')
                    ##################################
                    print('LC_yymmdd.fits ----> LC_yymm.fits')
                    hdul = fits.HDUList([mags_hdu, jd_hdu, inx_hdu])        
                    hdul.writeto(dir_raw_m + 'LC_' + ym_i + '.fits', overwrite=True)      
                    ##################################
                    print('LC_yymm.fits -----> LC_obj_mm.npy')
                    G_inx = DF.index[:5]
                    G_inx_ym = np.concatenate((G_inx_ym, G_inx))
    
                    N = len(G_inx)//n + 1
                    print(N, '\n')
                    for i in range(N): 
                        G_inx_i = G_inx[i * n : (i + 1) * n]   
                        print(i, end=',')
                        
                        processes = []                    
                        for g_inx in G_inx_i:     
                            p = multiprocessing.Process(target=night2month, 
                                                        args=(g_inx, dir_tmp, ym_i, DF))
                            processes.append(p)
                            p.start()
                        for p in processes:
                            p.join()
    
        G_inx_ym = np.int32(np.unique(G_inx_ym))
        S_df = pd.DataFrame({'Np':[]})
    
        N = len(G_inx_ym)//n + 1   
        ###############################
        print('LC_obj_mm.npy -------> LC_obj.csv')
        print('len(G_inx) = ', len(G_inx_ym))
        print('N = ', N)
        for i in range(N): 
            G_inx_i = G_inx_ym[i * n : (i + 1) * n]   
            print(i, end=',')
            
            processes = []
            manager = multiprocessing.Manager()
            S = manager.dict()
            
            for g_inx in G_inx_i:     
                p = multiprocessing.Process(target=monthObj2obj, 
                                            args=(S, g_inx, dir_tmp, dir_obj))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            
            K = S.keys()
            V = S.values()
        
            S_df = pd.concat((S_df, pd.DataFrame(data=V, index=K, columns=['Np'])))
          
        S_df.to_csv(dir_stat + 'Stat' + '.csv')
    
    os.system('rm -r ' + dir_tmp)
    

if __name__ == "__main__":
    main()























