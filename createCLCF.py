#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:02:27 2026

@author: fedora
"""

def calibrate(g_inx):
    f0 = getRLC(dir_RLC, g_inx, JD)  
    sts = Refs.loc[g_inx]
    DF = []
    for g_st in sts:
        f = getRLC(dir_RLC, g_st, JD)
        df = f - f0
        df = sigma_clip(df, sigma=5, maxiters=1)
        u_nan = np.isnan(df)
        for jd in JDfu:
            u_jd = JDf==jd
            y = df[u_jd & (~u_nan)]
            x = t0[u_jd & (~u_nan)]
            y = sigma_clip(y, sigma=1, maxiters=3)
            
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





    
tasks = [(g_inx,) for g_inx in G_inx]
results = parallel_map(calibrate, tasks, max_workers=7, chunksize=1)
