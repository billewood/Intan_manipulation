# -*- coding: utf-8 -*-

import os
from matplotlib.pyplot import *
import quantities as pq
from zeebeez.tdt2neo import stim, import_multielectrode
from neo.core import EpochArray
from neo.io import RHDIO, NeoHdf5IO
from neosound import sound_manager
from neosound.sound_store import HDF5Store
from lasp.signal import lowpass_filter, highpass_filter, coherency
from lasp.sound import spectrogram, plot_spectrogram
from scipy.stats import zscore
from scipy.signal import correlate, argrelextrema

# faster plotting
# %matplotlib notebook  

experiment_dir = "/auto/tdrive/billewood/intan data/LBlYel0923"
experiment_file = os.path.join(experiment_dir, "RHD", "LBlYe0923__160829_144329.rhd")
stimulus_file = os.path.join(experiment_dir, "pyoperant", "LBlYel0923_trialdata_20160829144324.csv")
stimulus_dir = os.path.join(experiment_dir, "Stimuli", "TutFam")
figures_dir = os.path.join(experiment_dir, "Figures")


# Get the neural data
h5_filename = os.path.join(experiment_dir,
                           os.path.basename(experiment_file).replace(".rhd", "_neo.h5"))
h5_exporter = NeoHdf5IO(h5_filename)
block = h5_exporter.get("/Block_0")

# Get the sound database 
stimulus_h5_filename = os.path.join(experiment_dir,
                        os.path.basename(experiment_file).replace(".rhd", "_stimuli.h5"))
sm = sound_manager.SoundManager(HDF5Store, stimulus_h5_filename)
# This is the entire microphone recording
mic = [asig for asig in block.segments[0].analogsignalarrays if asig.name == "Board ADC"][0]
fs_mic = np.int(mic.sampling_rate)
t_mic = np.asarray(mic.times)
mic = mic.squeeze()
mic = mic - np.mean(mic) # because our mic channel often has 1.652 v of dc leak from somewhere!

low_power = 2000
high_power = 5000
vb_low = 5 # lowpass for smoothing vocal band 
vb_stds = 1 # num of stds for vocal band threshhold
vd_stds = 1 # num of stds for vocal density threshold
vd_win = .25 # in s
smooth_power = 5 # not in use I think
onset_shift = -.3 # in s, move the onset this direction (should be negative usually)
low_freq_corr = 1000 # Hz, lower bound for frequencies to correlate in spectrogram
high_freq_corr = 10000 # in Hz, " "

vocal_band = lowpass_filter(mic, fs_mic, low_power)
vocal_band = highpass_filter(vocal_band, fs_mic, high_power)
vocal_band = np.abs(vocal_band)
vocal_band = lowpass_filter(vocal_band, fs_mic, vb_low)
vocal_band_thresh = vb_stds * np.std(vocal_band)
# find periods of time above the threshold...
vocal_threshed = np.zeros(vocal_band.shape)
for i in range(vocal_band.shape[0]):
    if vocal_band[i] > vocal_band_thresh: 
        vocal_threshed[i] = 1
    else: vocal_threshed[i] = 0       
    
window = np.int(np.round(vd_win * fs_mic / 2) * 2) # force an even integer
overlap = window / 2
start_time = 0
i = 1

# TODO it's pretty weird the way I'm windowing this, maybe I should just lowpass filter it?
vocal_density = np.zeros(np.int(np.floor(vocal_band.shape[0]/(window-overlap))))
while start_time + window < vocal_band.shape[0]:
    vocal_density[i] = np.mean(vocal_threshed[start_time:start_time+window])
    start_time = start_time + window - overlap
    i += 1    
     
# sound onsets: 1 is an onset, -1 is an offset
density_thresh = vd_stds * np.std(vocal_density)
vd_crosses = np.zeros(len(vocal_density))
for i in range(vocal_density.shape[0]):
    if vocal_density[i] > density_thresh:
        vd_crosses[i] = 1
sound_onsets = vd_crosses[1:len(vd_crosses)] - vd_crosses[0:len(vd_crosses)-1] 
sound_onsetsb = np.zeros(len(sound_onsets)+1)
sound_onsetsb[1:len(sound_onsetsb)] = sound_onsets # just aligns for n-1, must be a better way oh well
del sound_onsets
sound_onsets = sound_onsetsb
del sound_onsetsb
# this just checks that the first sound onset wasn't obscured by the beginning of the file
for i in range(len(sound_onsets)):
    if sound_onsets[i] == -1:
        sound_onsets[0] = -1 # the case where sound onset happens on the first data point
        print("Warning, sound appears to onset before file start, first time point being set to an onset")
    if sound_onsets[i] == 1:
        break 
# check for missing offset (sound continues past data file)
for i in range(len(sound_onsets)):
    if sound_onsets[len(sound_onsets)-i-1] == 1:
        sound_onsets[len(sound_onsets)] = -1
        print("Warning, sound appears to continue past file end, last time point being set to an offset")
    if sound_onsets[len(sound_onsets)-i-1] == -1:
        break

# convert to a list of just onsets and offsets    
sound_onset = []
sound_offset = []
for i in range(len(sound_onsets)):
    if sound_onsets[i] == 1:
        sound_onset = np.append(sound_onset, [i])
    if sound_onsets[i] == -1:
        sound_offset = np.append(sound_offset, [i])        
try:
    len(sound_onset) == len(sound_offset)
except:
    print("Number of vocal onsets and offsets not equal!")
sound_onset = sound_onset * (window - overlap) # convert to original data points
sound_onset = sound_onset + np.round(onset_shift * fs_mic)
sound_offset = sound_offset * (window - overlap) # " "

#==============================================================================
# for i in range(len(sound_onset)):
#     t, freq, timefreq, rms = spectrogram(mic[sound_onset[i]:sound_offset[i]], fs_mic, 1000, 50)  
#     plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)
#     pause(.05)
#==============================================================================
    
# just a temporary template, ~ 1 motif, maybe even a playback but who cares
i = 17
template = mic[sound_onset[i]:sound_offset[i]]
template = template[1000:14000]
templ_t, templ_freq, templ_timefreq, templ_rms = spectrogram(template, fs_mic, 1000, 50)

# Make xcorr of spectrogram - if i = 17 template is present in sound 
# TODO this is the part that will have to loop through for every vocal chunk
# for now it is just a bout that has three motifs in it
t, freq, timefreq, rms = spectrogram(mic[sound_onset[i]:sound_offset[i]], fs_mic, 1000, 50)  

# find the frequencies of interest
freq_index = [0]
iter = 0
for i in range(len(freq)):
    if (freq[i] > low_freq_corr) & (freq[i] < high_freq_corr):
        freq_index = np.append(freq_index, [i])
        iter += iter
freq_index = freq_index[1:len(freq_index)]    

# zscore specs first, and just do freqs of interest
templ_timefreq = templ_timefreq[freq_index,:]
timefreq = timefreq[freq_index,:]
templ_timefreq = zscore(templ_timefreq, axis = None)
timefreq = zscore(timefreq, axis = None)    

# loop through and do crosscos on specs, freq band by freq band, time point by time point    
spect_corr = np.zeros(timefreq.shape[1] - templ_timefreq.shape[1])
# sliding time
for time_win in range(timefreq.shape[1] - templ_timefreq.shape[1]): # no overlap before or after
   # correlate relative freq band to each other
    freq_corr = np.zeros(len(timefreq))
    for freq_i in range(len(freq_index)): 
        in1 = np.abs(templ_timefreq[freq_i])
        in2 = np.abs(timefreq[freq_i][time_win:time_win + len(in1)])
        freq_corr[freq_i] = correlate(in1,in2, mode = 'valid')
    spect_corr[time_win] = np.sum(freq_corr)

corr_list = np.append(corr_list, spect_corr)    

# find peaks
peaks = argrelextrema(spect_corr, np.greater, order = 100) # seems to work ok

    
#==============================================================================
# 
# # just to check things look ok
# figure(1)    
# plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)    
# figure(2)
# plot_spectrogram(t, freq, templ_timefreq, dBNoise=80, colorbar = False)    
# figure(3)
# plot(spect_corr)
#==============================================================================
    
# TODO Next is to iterate through vocal_density and check spectrograms, but I'm running out of time for now
# the idea will be to calculate xcorrs of both amplitude waveform and the spectrogram (row by row). Both will be zscored.
# this should give us 
#for i in range(vocal_density.shape[0]):
#    if vocal_density[i] > density_thresh:
#        t, freq, timefreq, rms = spectrogram(mic[i*(window-overlap):i*(window-overlap)+window], fs_mic, 1000, 50)
#        plot_spectrogram(t, freq, timefreq, dBNoise=80)

# xcorr for envelope and spectrogram, then make it slide, zscore, correlate


plot(vocal_density[0:200])
i = 0
t, freq, timefreq, rms = spectrogram(mic[i*(window-overlap):50*(window-overlap)+window], fs_mic, 1000, 50)
plot_spectrogram(t, freq, timefreq, dBNoise=80)

i = 75
t, freq, timefreq, rms = spectrogram(mic[i*(window-overlap):(i+25)*(window-overlap)+window], fs_mic, 1000, 50)
plot_spectrogram(t, freq, timefreq, dBNoise=80)

t, freq, timefreq, rms = spectrogram(template, fs_mic, 1000, 50)
plot_spectrogram(t, freq, timefreq, dBNoise=80)

# mic[36600000:36750000] has a song
# t, freq, timefreq, rms = spectrogram(mic[36600000:36750000], fs_mic, 1000, 50)
# plot_spectrogram(t, freq, timefreq, dBNoise=80)

