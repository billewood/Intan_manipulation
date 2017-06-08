# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:35:00 2017

@author: billewood
"""

# Depencies
# Needs these packages,functions
from neo.io import NixIO
from neo.core import Epoch
import numpy as np
import os
import quantities as pq
try:
    import cpickle as pickle
except:
    import pickle
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Theunissen Lab dependencies.
from lasp.signal import lowpass_filter, highpass_filter
from lasp.sound import spectrogram, plot_spectrogram
from ipywidgets import *



# Specify the input file again. The segemented file is a pickle file that has s single block but
# multiple labelled segements corresponding to sections with song.  It was created with audiosegmentnix.py (in songephys).
experiment_dir = "/auto/tdrive/billewood/intan data/LbY6074"
pkl_file_in = "LbY6074__161215_132633_seg_c1.pkl"
#pkl_file_out = "LbY6074__161216_104808_seg_c1.pkl"  # c1 for check 1.  You could use the same name and it will overwrite
# Open the Inupt Pkl file 
pklFileIn = open(os.path.join(experiment_dir, 'NIX', pkl_file_in), 'rb')

# Read the data
block = pickle.load(pklFileIn)

pklFileIn.close()

print 'Found', len(block.segments), 'segments:'
nstudent = 0
ntutor = 0
for i,seg in enumerate(block.segments):
    if seg.name.startswith('student'):
        nstudent +=1
    elif seg.name.startswith('tutor'):
        ntutor += 1
print '\t', nstudent, 'student songs and', ntutor, 'tutor songs.'

# Add a little time before and after each event.
# times before and after were already added before segmentation
# tbefore = 0.2*pq.s # Time before begining of event
# tafter = 0.2*pq.s # Time after end of even

def plotSoundSeg(fig, seg):
    # fig pointer to figure
    # seg is the segment
    # returns the filtered sound from the loudest of the two signals.
    
    # clear figure
    fig.clear()
    
    # The sound signal
    soundSnip = seg.analogsignals[1]
    fs = soundSnip.sampling_rate
    tvals = soundSnip.times
            
    # Calculate the rms of each mic
    rms0 = np.std(soundSnip[:, 0])
    rms1 = np.std(soundSnip[:, 1])
            
    # Choose the loudest to plot
    if rms1 > rms0:
        sound = np.asarray(soundSnip[:, 1]).squeeze()
    else:
        sound = np.asarray(soundSnip[:, 0]).squeeze()
            
    # Calculate envelope and spectrogram
    sound = sound - sound.mean()
    sound_env = lowpass_filter(np.abs(sound), float(fs), 250.0)  # Sound Enveloppe
    to, fo, spect, rms = spectrogram(sound, float(fs), 1000, 50)
           
    # Plot sonogram and spectrogram
    gs = gridspec.GridSpec(100, 1)
        
    ax = fig.add_subplot(gs[0:20,0])
            
    ax.plot(tvals-tvals[0], sound/sound.max())
    ax.plot(tvals-tvals[0], sound_env/sound_env.max(), color="red", linewidth=2)
    plt.title('%s %d' % (seg.name, seg.index))
    plt.xlim(0.0, to[-1])
            
    ax = fig.add_subplot(gs[21:,0])
    plot_spectrogram(to, fo, spect, ax=ax,
        ticks=True, fmin=250, fmax=8000, colormap=None, colorbar=False, log = True, dBNoise = 50)
    plt.ylabel('Frequency')
    plt.tick_params(labelbottom='off')
    plt.xlim(0.0, to[-1])
    plt.show()
    return sound, fs
    
    