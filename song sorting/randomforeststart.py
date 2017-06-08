# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:03:30 2017

@author: billewood
"""

import pandas
from sklearn.ensemble import RandomForestClassifier
import numpy as np

stored_data = '/auto/tdrive/billewood/intan data/song_sorting/from_FET_Julie/vocSelTable.h5'
figdir = '/auto/tdrive/billewood/intan data/song_sorting/from_FET_Julie/figure_storage'
df = pandas.read_hdf(stored_data)
features = df.columns[[2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
df['calltype'] = pandas.Series(df['calltype'], dtype="category") # make calltype categorical

nans = list()
for i in range(len(df)):
    for feature in features:
        if np.isnan(df[feature][i]):
            df.loc[i,(feature)] = df[feature].mean()
            nans.append([feature,i])

nan_ind = np.unique([number for name,number in nans])
nan_feats = np.unique([feature for feature, number in nans])

if len(nan_ind) > 0:
    print "Warning, %d entries were replaced by column averages" %len(nan_ind)
    print "Features containing NaNs are: ", nan_feats

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]

target_names = np.unique(df['calltype'])
y = train['calltype'].cat.codes

clf = RandomForestClassifier(n_jobs=2)
clf.fit(train[features], y)

preds = target_names[clf.predict(test[features])]
pandas.crosstab(test['calltype'], preds, rownames=['actual'], colnames=['preds'])


#######################################################################################
# This is for pulling the biosound from our Intan Data

import os
try:
   import cPickle as pickle
except:
   import pickle
import matplotlib.pyplot as plt
from NIXandPickleFunctions import plotSoundSeg, filteredPlot, getBioSoundProperties
from soundsig.sound import BioSound 
import numpy as np
import pandas as pandas

plotme = True
experiment_dir = "/auto/tdrive/billewood/intan data/LbY6074"
pkl_file_in = "LbY6074__161215_132633_seg_c1_corr_filtered_512.pkl"
pklFileIn = open(os.path.join(experiment_dir, 'NIX', pkl_file_in), 'rb')
# Read the data
block = pickle.load(pklFileIn)
pklFileIn.close()
block.segments[0]

for i,seg in enumerate(block.segments):
    mic = seg.analogsignals[0]
    mic = mic[:,0]
    soundIn = mic-np.mean(mic)
    biosound_data = getBioSoundProperties(soundIn, seg.name, plotme = False)
    if i == 0:
        vocSelTable = biosound_data
    else:
        vocSelTable = vocSelTable.append(biosound_data)


 # Save the results
# fh5name = 'h5files/%s.h5' % (fname)
# myBioSound.saveh5(fh5name)