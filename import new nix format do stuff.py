# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:13:06 2017

@author: billewood
"""

import os
from matplotlib.pyplot import *
import quantities as pq
from neo.io import RHDIO, NeoHdf5IO, nixio
from neosound import sound_manager
from neosound.sound_store import HDF5Store
from lasp.signal import highpass_filter
import numpy as np
from lasp.signal import lowpass_filter, highpass_filter, coherency
from lasp.sound import spectrogram, plot_spectrogram


amp_highpass = 400;
mic_highpass = 200;
mic_lowpass = 9000;

spec_sample_rate = 1000
freq_spacing = 50

experiment_dir = "/auto/tdrive/billewood/intan data/LBlYel0923"
experiment_file = os.path.join(experiment_dir, "RHD", "LBlYe0923__160830_120325.rhd")
figures_dir = os.path.join(experiment_dir, "Figures")
rhd_importer = RHDIO(experiment_file)
block = rhd_importer.read_block()
segment = block.segments[0]

mic = [asig for asig in block.segments[0].analogsignals if asig.name == "Board ADC"][0] # This is the entire microphone recording
amp = [asig for asig in segment.analogsignals if asig.name == "Amplifier"][0] # This is the entire microphone recording

del block
del segment

fs_mic = np.int(mic.sampling_rate)
t_mic = np.asarray(mic.times)
mic = mic.squeeze()


mic = lowpass_filter(mic, fs_mic, mic_lowpass)
mic = highpass_filter(mic, fs_mic, mic_highpass)

# high pass filter all the channels
high_amp = np.vstack([highpass_filter(channel, fs_mic, 400.0) for channel in amp.T]).T
del amp
#computer variance of the total signal for each channel either by the long way:
#for i in range(high_amp.shape[1]):
#    vari[i] = np.mean((high_amp[:,i]-np.mean(high_amp[:,i]))**2)
# or, shorter:
vari = [np.var(high_amp[:,channel]) for channel in range(high_amp.shape[1])]

# Now compute a correlation number, which aims to represent how correlated the channels are with each other at moments in time, which
# shoudl represent periods with large movement artifacts.
# let x and y be two channels, for each time window: sum( (x(t)-mean(x)) * (y(t)-mean(y))]**2 and divide by overall var of channel x and y
# then average for each pair of channels, giving one number per window
simple_corr = list()
window = 250
j = 0
i = 0
while i < high_amp.shape[0]-window: 
    temp_corr = list()
    for x in range(high_amp.shape[1]):
        for y in range(high_amp.shape[1]):
            temp_corr.append((np.sum( (high_amp[i:i+window,x]-np.mean(high_amp[i:i+window,x])) *  (high_amp[i:i+window,y]-np.mean(high_amp[i:i+window,y]))) ** 2)/vari[x]* vari[y])
    simple_corr.append(np.mean(temp_corr))
    i = i+window
    j += 1
    
scaled_corr = simple_corr/np.mean(simple_corr)
## dirty way of getting corrs scaled back to ephy sdata
corrs = [np.full((1,window),scaled_corr[x]) for x in range(scaled_corr.shape[0])]
xs = range(0,540000*250,250)
xs = np.asarray(xs)

## look at some data
amp_slice = high_amp[45000:60000,:]
mic_slice = mic[45000:60000]

t, freq, timefreq, rms = spectrogram(mic_slice, fs_mic, spec_sample_rate, freq_spacing)
plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)














means = [np.mean(high_amp[:,channel]) for channel in range(high_amp.shape[1])]

#variance of all signal
#cross product in window
#normalize by variance




fs_mic = np.int(mic.sampling_rate)
t_mic = np.asarray(mic.times)
mic = mic.squeeze()


mic = lowpass_filter(mic, fs_mic, mic_lowpass)
mic = highpass_filter(mic, fs_mic, mic_highpass)

for i in range(amp.shape[1]):
    amp[:,i] = highpass_filter(amp[:,i], fs_mic, amp_highpass)
    
    




# for instance, now
hold
plot(amp[0:20,0],'r')
plot(amp[0:20,1],'y')
plot(amp[0:20,2],'g')
plot(amp[0:20,3],'c')