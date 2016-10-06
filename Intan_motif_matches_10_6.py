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
from lasp.hinton import hinton
from mpl_toolkits.mplot3d import Axes3D
# faster plotting
# %matplotlib notebook  

experiment_dir = "/auto/tdrive/billewood/intan data/LBlYel0923"
experiment_file = os.path.join(experiment_dir, "RHD", "LBlYe0923__160829_144329.rhd")
stimulus_file = os.path.join(experiment_dir, "pyoperant", "LBlYel0923_trialdata_20160829144324.csv")
stimulus_dir = os.path.join(experiment_dir, "Stimuli", "TutFam")
figures_dir = os.path.join(experiment_dir, "Figures")

low_power = 2000
high_power = 5000
vb_low = 5 # lowpass for smoothing vocal band 
vb_stds = 1 # num of stds for vocal band threshhold
vd_stds = 1 # num of stds for vocal density threshold
vd_win = .25 # in s
onset_shift = -.3 # in s, move the onset this direction (should be negative usually or 0, to avoid missing some vocalizations)
low_freq_corr = 1000 # Hz, lower bound for frequencies to correlate in spectrogram
high_freq_corr = 10000 # in Hz, " "
normalize_mic = True # will subtract mean from mic, because our mic often has a dc leak in it. 

shorten = 0
    
def find_vocal_periods(vocal_band, vb_stds = 1, vd_stds = 1, vd_win = .25, fs_mic = 25000, onset_shift = 0):    
    """
        Given a sound pressure waveform, vocal_band, find areas with vocalizations (extended power). 
        Strongly recommended to bandpass the wave first (for instance, just pass 2-5 kHz).
        Performs two thresholding steps (Honestly I'm not sure how helpful the second one is but that's what I'm doing for now),
        each one is done based on standard deviations above mean, with the vocal density being windows (~low pass filtered)

        Returns: 
            sound_onset, sound_offset- two vectors of the same length (hopefully!)
              
        Arguments:
            REQUIRED:
                vocal_band: sound pressssure waveform, ideally band-pass filtered
                       
            OPTIONAL            
                vb_stds: vocal band standard deviations- power is determined by thresholding above this
                vd_stds: vocal density standard deviations- power is determined by thresholding above this
                vd_win: for smoothing the vocal density vector (second thresholding step)
                fs_mic: 25000 is our standard
                onset_shift: Many vocalizations start quietly so it's often best to just move all onsets a little back in time
        TODO Not sure why I chose to window/bin the second thresholding step instead of low-passing it, should think about that more
    """
    vocal_band_thresh = vb_stds * np.std(vocal_band)
    # find periods of time above the threshold...
    vocal_threshed = np.zeros(vocal_band.shape)
    for i in range(vocal_band.shape[0]):
        if vocal_band[i] > vocal_band_thresh: 
            vocal_threshed[i] = 1
    else: vocal_threshed[i] = 0       
    
    # TODO it's pretty weird the way I'm windowing this, maybe I should just lowpass filter it?
    window = np.int(np.round(vd_win * fs_mic / 2) * 2) # force an even integer
    overlap = window / 2
    start_time = 0
    i = 0
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
    return sound_onset, sound_offset

def get_spect_corr(sound_wav, templ_timefreq, freq_index, fs_mic = 25000, spec_sample_rate = 1000, freq_spacing = 50):
    """
        Calculate cross correlation between two spectrograms over only some frequencies, across time. Expects one wav file and
        one spectrogram and a frequency index. The spectrogram 'timefreq' should already be zscored.

        Returns: 
            correlation: spect_corr
              
        Arguments:
            REQUIRED:
                sound_wav, templ_timefreq, freq_index
                      
    """    
    t, freq, timefreq, rms = spectrogram(sound_wav, fs_mic, spec_sample_rate, freq_spacing)
    timefreq = timefreq[freq_index,:]
    timefreq = zscore(timefreq, axis = None) 
    # zero pad only if the vocalization is shorter than the template
    if timefreq.shape[1] < templ_timefreq.shape[1]:
        zeropad = np.zeros((timefreq.shape[0],templ_timefreq.shape[1]-timefreq.shape[1]))
        timefreq = np.append(timefreq, zeropad, axis = 1)
    spect_corr = np.zeros(timefreq.shape[1])# - templ_timefreq.shape[1])
    spect_auto1 = np.zeros(timefreq.shape[1])
    spect_auto2 = np.zeros(timefreq.shape[1])
    freq_corr = np.zeros(len(freq_index))
    freq_in1 = np.zeros(len(freq_index))
    freq_in2 = np.zeros(len(freq_index))    
    # sliding time, taking the whole template with the vocalization sliding
    for time_win in range(timefreq.shape[1] - templ_timefreq.shape[1]): # no overlap before or after
        # correlate relative freq band to each other
        for freq_i in range(len(freq_index)): 
            in1 = np.abs(templ_timefreq[freq_i])
            in2 = np.abs(timefreq[freq_i][time_win:time_win + len(in1)])
#            freq_corr[freq_i] = correlate(in1,in2, mode = 'valid')
            freq_corr[freq_i] = np.sum(in1*in2)
            freq_in1[freq_i] = np.sum(in1*in1)
            freq_in2[freq_i] = np.sum(in2*in2)
        spect_corr[time_win] = np.sum(freq_corr)
        spect_auto1[time_win] = np.sum(freq_in1)
        spect_auto2[time_win] = np.sum(freq_in2)
    return spect_corr/np.sqrt(spect_auto1*spect_auto2)  
# plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)

def get_temporary_template(i = 17):
    """
        Returns a spectrogram and a template, if i = 17 it's some sort of song
        Really just a placeholder for a better template algorithm
        TODO replace this with real code, like user defined sections of spectrograms
    """
    template = mic[sound_onset[i]:sound_offset[i]]
    template = template[4800:18000] #shift it, particular to this template
    templ_t, templ_freq, templ_timefreq, templ_rms = spectrogram(template, fs_mic, spec_sample_rate = 1000, freq_spacing = 50)
    figure()    
    plot_spectrogram(templ_t, templ_freq, templ_timefreq, dBNoise=80, colorbar = False)
    return template, templ_t, templ_freq, templ_timefreq

def organize_playbacks(segment, sm, fs_mic = 25000):
# I stuck this all in 'sound_playback' so I could sort it by time
# good god this is ugly but i'm moving on for now
    """ 
        Returns sound_playback, which = -1 if the sound is not accounted for by 
        a playback. Only checks for overlap in time, many playbacks may have vocalizations
        in them that would be discarded and just considered playbacks with this method.
        This is still pretty rough but works for now.
    """
    stim_time = np.zeros(len(segment.epocharrays))
    stim_duration = np.zeros(len(segment.epocharrays))
    stimuli_times = list() #used temporarily
    stim_id = list()
    stim_name = list()
    i = 0
    for epoch in segment.epocharrays:
        stimuli_times.append(epoch.times)
        stim_id.append(sm.database.get_annotations(epoch.annotations["stim_id"]))
        stim_name.append(stim_id[i]['callid'])
        i = i + 1

    for i in range(len(segment.epocharrays)):
        stim_time[i] = segment.epocharrays[i].times #* 1000 # in s
        stim_duration[i] = segment.epocharrays[i].durations # in s

    sound_playback = np.zeros(len(sound_onset)) - 1    
    for i in range(len(sound_onset)):
        for j in range(len(stim_time)):
            time_diff = (((sound_onset[i] + sound_offset[i])/ 2) / fs_mic) - (stim_time[j] + (stim_duration[j] / 2))
            if np.abs(time_diff) < ((((sound_offset[i]-sound_onset[i]) / 2) / fs_mic) + (stim_duration[j] / 2)): # vocalization overlaps with a stimuli
                sound_playback[i] = j
    return sound_playback
################################################################################################################################
###############################################################################################################################
# Get the neural data
h5_filename = os.path.join(experiment_dir,
                           os.path.basename(experiment_file).replace(".rhd", "_neo.h5"))
h5_exporter = NeoHdf5IO(h5_filename)
block = h5_exporter.get("/Block_0")

# Get the sound database 
stimulus_h5_filename = os.path.join(experiment_dir,
                        os.path.basename(experiment_file).replace(".rhd", "_stimuli.h5"))
sm = sound_manager.SoundManager(HDF5Store, stimulus_h5_filename) # make a sound manager object

# Get the number of trials and make a list of stimulus ids, 'sound_data' (because the sm object is a little hard to work with)
# TODO organizing stimulus times to validate against periods of vocalization
segment = block.segments[0] # The block only has one segment
num_stimuli = len(segment.epocharrays)
#   how to access times of stimuli (because it's hard to figure out):
# segment.epocharrays[0].times
# segment.epocharrays[0].durations  # durations of stimuli
#   and how to use the sound manager object:
# sm.database.get_annotations(epoch.annotations["stim_id"])["callid"]
#   and use time slice:
# mic_slice = mic.time_slice(epoch.times-tbefore, epoch.times + epoch.durations + tafter)

# pull the microphone channel, filter it etc
mic = [asig for asig in block.segments[0].analogsignalarrays if asig.name == "Board ADC"][0] # This is the entire microphone recording
fs_mic = np.int(mic.sampling_rate)
t_mic = np.asarray(mic.times)
mic = mic.squeeze()
if normalize_mic:
    mic = mic - np.mean(mic) # because our mic channel often has 1.652 v of dc leak from somewhere!
vocal_band = lowpass_filter(mic, fs_mic, low_power)
vocal_band = highpass_filter(vocal_band, fs_mic, high_power)
vocal_band = np.abs(vocal_band)
vocal_band = lowpass_filter(vocal_band, fs_mic, vb_low)

# find periods of sound on the mic channel that is likely vocalizations
# sound_onset is in data points, not time
sound_onset, sound_offset = find_vocal_periods(vocal_band, vb_stds, vd_stds, vd_win, fs_mic, onset_shift)
if shorten:
    sound_onset = sound_onset[0:18]
    sound_offset = sound_offset[0:18]
    mic = mic[0:sound_offset[17]+fs_mic]
# ### beginning user controlled template interfaces
# changed template_onsets to a list, will have to fix the other calls of it

template = list()
templ_t = list()
templ_freq = list()
templ_timefreq = list()
template_onsets = list()
i = 0
keep_looking = 1
while keep_looking:
    i += 1
    template_temp, templ_t_temp, templ_freq_temp, templ_timefreq_temp = get_temporary_template(i)
    # user input/ plot spectrogram and mic wavform
    
    save_template = raw_input("Enter 1 to save, q to quit looking for templates, anything else to continue: ")
    if save_template == 1:      
        template.append(template_temp)
        templ_t.append(templ_t_temp)
        templ_freq.append(templ_freq_temp)
        templ_timefreq.append(templ_timefreq_temp)
        template_onsets.append(i)
    elif save_template == q:
        keep_looking = 0
        break
    
# get templates (temporary at this point still)
template = list()
templ_t = list()
templ_freq = list()
templ_timefreq = list()
template_onsets = [17, 2]
for i in template_onsets:
    template_temp, templ_t_temp, templ_freq_temp, templ_timefreq_temp = get_temporary_template(i)
    template.append(template_temp)
    templ_t.append(templ_t_temp)
    templ_freq.append(templ_freq_temp)
    templ_timefreq.append(templ_timefreq_temp)
#==============================================================================

# # find the frequencies of interest and then z-score the templates
#==============================================================================
freq_index = []
iter = 0
for i in range(len(templ_freq[0])):
    if (templ_freq[0][i] > low_freq_corr) & (templ_freq[0][i] < high_freq_corr):
        freq_index.append(i)
        iter += iter
for i in range(len(templ_timefreq)):
    templ_freq[i] = templ_freq[i][freq_index]
    templ_timefreq[i] = templ_timefreq[i][freq_index,:]
    templ_timefreq[i] = zscore(templ_timefreq[i], axis = None)

#==============================================================================
# loop through every sound_onset and calculate correlations with templates 
template_corr = list()
template_corr_peak = list()
for jj in range(len(templ_freq)):   # the number of templates
    corr_list = list()
    peak_list = list()
    longest_corr = 0
    for i in range(len(sound_onset)):
        sound_wav = mic[sound_onset[i]:sound_offset[i]]
        spect_corr = get_spect_corr(sound_wav, templ_timefreq[jj], freq_index, fs_mic)
        corr_list.append(spect_corr)
        peaks = argrelextrema(spect_corr, np.greater, order = 100) # seems to work ok / except now it maybe doesn't
        peak_list = np.append(peak_list, peaks)
        if len(spect_corr) > longest_corr: # just keeps track of the longest corr so we can pad the others with NAN later
            longest_corr = len(spect_corr)

    # to vstack corr_list the elements must all be the same size 
    corr_list_noappend = corr_list
    for i in range(len(corr_list)):
        if len(corr_list[i]) < longest_corr :
            zero_pad = np.zeros(longest_corr - len(corr_list[i]))
            corr_list[i] = np.append(corr_list[i], zero_pad)
    corr_list_array = np.vstack(corr_list)

    # make zeros = NAN; TODO this just get incorporated into the zero pad above
    corr_len_nonan = np.zeros(len(corr_list_array))
    for i in range(len(corr_list_array)):
        for j in range(corr_list_array.shape[1]):
            if corr_list_array[i,j] == 0:
                corr_list_array[i,j] = np.nan
            else:
                corr_len_nonan[i] = corr_len_nonan[i] + 1
            
    template_corr.append(corr_list_array)
    template_corr_peak.append(peak_list)     # this is giving weird numbers  DONT USE IT FOR NOW     

# =============================================================================
## calculate mean corrs, peak corrs, and plot the templates against each other

mean_corrs = list() # holds mean arrays for each template
max_corrs = list()
for i in range(len(template_corr)):
    mean_corrs.append(np.nanmean(template_corr[i], axis = 1))
    max_corrs.append(np.nanmax(template_corr[i], axis = 1))
    
# =============================================================================
# At this point we should be able to do some catagorizing of vocalizations based
# on the mean and max(peak) correlations of the templates
# template_corr is a list of arrays. each element corresponds to 1 template. 
# each template then has an array of the xcorr between it and every vocalization
# mean_corrs are the mean correlations of each template vs each vocalization, 
# peak_corrs are thea peak/max correlations of " " "

# plot the mean correlations between the first two templates and each vocalization
plot(mean_corrs[0], mean_corrs[1],'o')    

# plot the max correlations between each template and each vocalization, seems best
plot(max_corrs[0], max_corrs[1],'o')    


#   how to access times of stimuli (because it's hard to figure out):
# segment.epocharrays[0].times
# segment.epocharrays[0].durations  # durations of stimuli
#   and how to use the sound manager object:
# sm.database.get_annotations(epoch.annotations["stim_id"])["callid"]
#   and use time slice:
# mic_slice = mic.time_slice(epoch.times-tbefore, epoch.times + epoch.durations + tafter)    

# Now let's identify which vocalizations are actually playbacks!!!
# first i need stimuli in a format i can use: playbacks = -1 if it's not a playback
sound_playback = organize_playbacks(segment, sm, fs_mic)

# find max_corrs of second template (thuck) that are both very large and are not playbacks
# this is very particular to the grant but the idea is generalizable
nonplayback = np.where(sound_playback > -1)
matches = np.where(max_corrs[1][nonplayback] > 0.4)
nonplayback[0][matches] # indexes of max_corrs[1] above .5
for i in range(len(matches[0])):
    sound = mic[np.int(sound_onset[nonplayback[0][matches][i]]):np.int(sound_offset[nonplayback[0][matches][i]])]
    t, freq, timefreq, rms = spectrogram(sound, sample_rate, 1000, 50)
    figure()
    plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)

good_matches = [0,1,6,7,11,12,14]
zoomed = ([.37, .49], [.45, .66], [.4, .5], [.4, .5], [.4, .5], [.47, .57], [.54, .64])
for i in range(len(good_matches)):
    sound = mic[sound_onset[nonplayback[0][matches][good_matches[i]]]:sound_offset[nonplayback[0][matches][good_matches[i]]]]
    sound = sound[fs_mic * zoomed[i][0]:fs_mic * zoomed[i][1]]    
    t, freq, timefreq, rms = spectrogram(sound, sample_rate, 1000, 50)
    figure()
    plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)



# final syllable places
zoomed = ([.37, .49], [.45, .66], [.4, .5], [.4, .5], [.4, .5], [.47, .57], [.54, .64])
syllables = np.zeros([len(zoomed),2])
for i in range(len(zoomed)):
    syllables[i][0] = sound_onset[nonplayback[0][matches][good_matches[i]]] + (fs_mic * zoomed[i][0])
    syllables[i][1] = sound_onset[nonplayback[0][matches][good_matches[i]]] + (fs_mic * zoomed[i][1])
    sound = mic[syllables[i][0]:syllables[i][1]]
    t, freq, timefreq, rms = spectrogram(sound, sample_rate, 1000, 50)
    figure()
    plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)
 
# so if you have mic and syllables
syllables = ([[  1439250.,   1442250.],
       [  2019375.,   2024625.],
       [ 12380625.,  12383125.],
       [ 16674375.,  16676875.],
       [ 18352500.,  18355000.],
       [ 19248000.,  19250500.],
       [ 22515375.,  22517875.]])  
for i in range(len(syllables)):
    sound = mic[syllables[i][0]:syllables[i][1]]
    t, freq, timefreq, rms = spectrogram(sound, sample_rate, 1000, 50)
    figure()
    plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)



    
# sort the stimuli name etc so it's not impossible to work with    
playbacks = list()            
dtype = [('time', float), ('duration', float), ('name','S10')]
for i in range(len(stim_time)):
    playbacks.append((stim_time[i], stim_duration[i], stim_name[i]))
#==============================================================================
#     sound_data = sm.database.get_annotations(epoch.annotations["stim_id"])
#         print 'Epoch:', i, 'Trial:', epoch.annotations['trial'], 'Stim:', sound_data['callid'], sound_data['emitter'],\
# 
#==============================================================================
playbacks = np.array(playbacks, dtype = dtype)
sorted_playback = np.sort(playbacks, order = ['time'])
# check
sorted_playback['time'][0:5]


# Now we can see when a sound onset is temporally aligned with a stimuli playback.
# if they are aligned, make sound_playback (which has len(sound_onset)) equal the sound stimuli number     

            
# Find the corresponding stim wave file and calculate its enveloppe
for isound_onset in range(3):
    epoch = segment.epocharrays[isound_onset]
    s = sm.reconstruct(epoch.annotations["stim_id"])
    sound_data = sm.database.get_annotations(epoch.annotations["stim_id"])
    sample_rate = sound_data['samplerate']
    t_sound = s.times
    sound = np.asarray(s).squeeze()
    t, freq, timefreq, rms = spectrogram(sound, sample_rate, 1000, 50)
    figure()
    plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)

#==============================================================================
# for isound_onset in a[0]:
#     t, freq, timefreq, rms = spectrogram(mic[sound_onset[isound_onset]:sound_offset[isound_onset]], fs_mic, 1000, 50)
#     figure()
#     plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)
#==============================================================================



sound_env = lowpass_filter(np.abs(sound), float(sample_rate), 250.0)



#3d figure but needs to be updated for new lists of templates
#==============================================================================
# fig = figure(1)
# ax = Axes3D(fig)
# ax.scatter(mean_corrs, max_corrs, corr_len_nonan)
# ax.legend(loc = 'upper left')
# draw()
# show()
# 
#==============================================================================
#==============================================================================
# # we can make a hinton diagram but it does take a long time 
# corr_table = np.corrcoef(corr_list)
# figure(1);
# hinton(corr_table)
#==============================================================================
    
