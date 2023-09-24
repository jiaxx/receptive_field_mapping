

import numpy as np
import scipy as sp
import os, sys,glob, copy
import pandas as pd


mouse_IDs = [ '424445','415148', '416356','416861','419112','419116'] 

for mouse_ID in mouse_IDs:

    basepath = '/Users/xiaoxuanj/work/work_allen/Ephys/mouse'+mouse_ID
    
    df = pd.read_csv('/Volumes/local1/work_allen/Ephys/mouse'+mouse_ID+'/matrix/mouse'+mouse_ID+'_cortex_meta.csv', index_col=0)
    FR = df.FR.values
    df = df[FR>2]
    df = df.reset_index().drop(['index'], axis=1)


    # load connectivity data
    ccg_grating = dl.data_loader(mouse_ID, datatype='CCG_grating')
    half = ccg_grating.shape[-2]/2
    X = np.nanmean(np.nanmean(ccg_grating[:,:,half-13:half,np.arange(1,8,2)], axis=2), axis=2)-np.nanmean(np.nanmean(ccg_grating[:,:,half:half+13,np.arange(1,8,2)], axis=2), axis=2)
    del ccg_grating

    # load RF df (from Gabor fit), select units with RF on screen
    df_rf = pd.read_csv('/Users/xiaoxuanj/work/work_allen/Ephys/processed_data/RF_features/mouse'+mouse_ID+'_rf_features.csv', index_col=0)

    df_tmp=pd.merge(df,df_rf, on=['unit_id', 'probe_id', 'channel_id'], how='inner')
    df_tmp = df_tmp[(df_tmp['rf_center_x1'].values>1) & (df_tmp['rf_center_x1']<7) & (df_tmp['rf_center_y1']>1) & (df_tmp['rf_center_y1']<7)]

    select_idx = []
    for idx, row in df.iterrows():
        probe_id=row['probe_id']
        channel_id=row['channel_id']
        unit_id=row['unit_id']
        if unit_id in df_tmp[df_tmp.probe_id==probe_id].unit_id.values:
            select_idx.append(idx)

    X = X[select_idx, :][:, select_idx]
    assert X.shape[0]==len(df_tmp)
    X[X == -np.inf] = 0
    X[X == np.inf] = 0
    X[X == np.NaN] = 0

    plt.figure()
    plt.imshow(X, cmap='bwr', vmax=0.000002, vmin=-0.000002)
    plt.title(mouse_ID)
    
    