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
from lasp.sound import spectrogram, plot_spectrogram, plot_zscore_spectrogram
from scipy.stats import zscore
from scipy.signal import correlate, argrelextrema
from lasp.hinton import hinton
from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd
# faster plotting
# %matplotlib notebook  

experiment_dir = "/auto/tdrive/billewood/intan data/LBlYel0923"
experiment_file = os.path.join(experiment_dir, "RHD", "LBlYe0923__160829_144329.rhd")
stimulus_file = os.path.join(experiment_dir, "pyoperant", "LBlYel0923_trialdata_20160829144324.csv")
stimulus_dir = os.path.join(experiment_dir, "Stimuli", "TutFam")
figures_dir = os.path.join(experiment_dir, "Figures")

avoid_stims = 0 # when searching for templates, avoids vocalizations that have stimuli in them
sound_length_thresh = .2 # excludes sound periods shorter than this (in data points for now) when searching for templates
mic_lowpass = 700
mic_highpass = 11000
vb_lowpass = 2000 # for finding vocalization areas (vocal band)
vb_highpass = 5000 # for finding vocalization area (vocal band)
vb_low = 5 # lowpass for smoothing vocal band after it has been lowpassed (second lowpass) 
vb_stds = .75 # num of stds for vocal band threshhold
vd_stds = 1 # num of stds for vocal density threshold
vd_win = .25 # in s
onset_shift = 0 # in s, move the onset this direction (should be negative usually or 0, to avoid missing some vocalizations)
low_freq_corr = 1000 # Hz, lower bound for frequencies to correlate in spectrogram
high_freq_corr = 10000 # in Hz, " "
normalize_mic = True # will subtract mean from mic, because our mic often has a dc leak in it. 

spec_sample_rate = 1000
freq_spacing = 50
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

   
def plot_template_specs(templ_t, templ_freq, templ_timefreq):
    for template in range(len(templ_t)):
        figure()
        plot_zscore_spectrogram(templ_t[template], templ_freq[template], templ_timefreq[template])
        
#==============================================================================
def get_symmetric_spect_corr(sound_timefreq, templ_timefreq, templ_len, fs_mic = 25000, spec_sample_rate = 1000, freq_spacing = 50, zscore = 0):
    """
        Calculate cross correlation between two spectrograms over only some frequencies, across time. Expects one wav file and
        one spectrogram and a frequency index. The spectrogram 'timefreq' should already be zscored, this function automatically zscores the sound_wav
        spectrogram.
        If you want to run it on certain frequencies, only pass those frequencies to this function.
        Returns: 
            correlation: spect_corr
              
        Arguments:
            REQUIRED:
                sound_wav (a sound pressure vector), 
                template_timefreq (from lasp.sound.spectrogram, the spectrogram (a time-frequency representation)), 
                
    """   
    if zscore:
        sound_timefreq = zscore(np.abs(sound_timefreq), axis = None) 
        templ_timefreq = zscore(np.abs(templ_timefreq), axis = None) 

    # zero pad so the template and vocal period are the same length
    if sound_timefreq.shape[1] < templ_timefreq.shape[1]:
        zeropad = np.zeros((sound_timefreq.shape[0],templ_timefreq.shape[1]-sound_timefreq.shape[1]))
        sound_timefreq = np.append(sound_timefreq, zeropad, axis = 1)
    elif sound_timefreq.shape[1] > templ_timefreq.shape[1]:
        zeropad = np.zeros((templ_timefreq.shape[0],sound_timefreq.shape[1]-templ_timefreq.shape[1]))
        templ_timefreq = np.append(templ_timefreq, zeropad, axis = 1) 
    spect_corr = np.zeros(sound_timefreq.shape[1] * 2)# twice the length
    amp_corr = np.zeros(len(spect_corr))
    freq_corr = np.zeros(templ_timefreq.shape[0]) # num of frequency bins
    freq_in1 = np.zeros(templ_timefreq.shape[0])
    freq_in2 = np.zeros(templ_timefreq.shape[0])   
    # sliding time, taking the whole template with the vocalization sliding
    for t in range((templ_timefreq.shape[1]-1) * 2): #
        t = t + 1
        if t < templ_timefreq.shape[1]: # sliding time, first half of timepoints
            in1 = sound_timefreq.T[:][-t:] # take all frequencies, restricted time- transposing seems like a weird way to do thsi but it's all ive got
            in2 = templ_timefreq.T[:][0:t]
        else: # sliding time, second half of timepoints
            in1 = sound_timefreq.T[:][0:templ_timefreq.shape[1] * 2 - t]            
            in2 = templ_timefreq.T[:][-(templ_timefreq.shape[1] - t):]
        # correlate relative freq band to each other
        for freq_i in range(in1.shape[1]): 
            freq_in1 = in1.T[freq_i][:]
            freq_in2 = in2.T[freq_i][:]
            freq_corr[freq_i] = np.sum(in1.T[freq_i][:] * in2.T[freq_i][:])
        spect_corr[t] = np.sum(freq_corr)
        # the sum of power at all frequencies is ~ a smoothed amlitude waveform, and the 
        # product of all times points is the numerator of the amplitude cross-correlation
        amp_corr[t] = np.sum((np.sum(in1.T, axis = 0) * np.sum(in2.T, axis = 0)))
       # the denominator for both is the length of the template
    return (spect_corr / templ_len * templ_timefreq.shape[0]), (amp_corr / templ_len * templ_timefreq.shape[0]) # / the length of the template before zeropadding 

#==============================================================================

def get_temporary_template(sound_onset, sound_offset, fs_mic, i = 17):
    """
        Returns a spectrogram and a template, if i = 17 it's some sort of song
        Really just a placeholder for a better template algorithm
        TODO replace this with real code, like user defined sections of spectrograms
    """
    template = mic[sound_onset[i]:sound_offset[i]]
   # template = template[4800:18000] #shift it, particular to this template
    templ_t, templ_freq, templ_timefreq, templ_rms = spectrogram(template, fs_mic, spec_sample_rate = 1000, freq_spacing = 50)
    plot_spectrogram(templ_t, templ_freq, templ_timefreq, dBNoise=80, colorbar = False)
    pause(.1)
    return template, templ_t, templ_freq, templ_timefreq

def organize_playbacks(segment, sm, sound_onset, sound_offset, fs_mic = 25000):
    # TODO why don't I send this function sound_onset and sound_offset????
    """ 
        Compares times with sound present vs times when playbacks occurred to determine
        which sounds are actual vocalizations.         
        Returns sound_playback, which = -1 if the sound is not accounted for by 
        a playback, and an integer corresponding to the playback id otherwise.        
        Only checks for overlap in time, many playbacks may have vocalizations
        in them that would be discarded and just considered playbacks with this method.
        This is still pretty rough but works for now.

        Also orders the stimuli which were used as sound playback, putting them in 
        an array 'stimuli', with fields 'duration', 'time', and 'name'.
        
    """
    stim_time = np.zeros(len(segment.epocharrays))
    stim_duration = np.zeros(len(segment.epocharrays))
    #stimuli_times = list() #used temporarily
 #   stim_id = list()
    stim_name = list()
    stim_env = list()
    i = 0
    for epoch in segment.epocharrays:
     #   stimuli_times.append(epoch.times)
 #       stim_id.append(sm.database.get_annotations(epoch.annotations["stim_id"]))
        stim_name.append(sm.database.get_annotations(epoch.annotations["stim_id"])['callid'])
        sample_rate = sm.database.get_annotations(epoch.annotations["stim_id"])['samplerate']
        s = sm.reconstruct(epoch.annotations["stim_id"])
        sound = np.asarray(s).squeeze()
        sound_env = lowpass_filter(np.abs(sound), float(sample_rate), 250.0)
        stim_env.append(sound_env)        
        stim_time[i] = epoch.times #* 1000 # in s
        stim_duration[i] = epoch.durations # in s
        i = i + 1
        # to check plotting
#==============================================================================
#     figure
#     hold
#     plot(stim_env[i])
#     plot(mic[round(stim_time[i]*fs_mic):round(stim_time[i]*fs_mic+5000)])
#==============================================================================

    # sort the stimuli name etc so it's not impossible to work with    
    sorted_stim_time_idx = sorted(range(len(stim_time)), key=lambda x:stim_time[x]) # iindex of stim_time, as they aren't in order
    stim_env = [stim_env[i] for i in sorted_stim_time_idx]    # organize stim_env in order of when stim_time happened
    stimuli = list()            
    dtype = [('time', float), ('duration', float), ('name','S10')]
    for i in range(len(stim_time)):
        stimuli.append((stim_time[i], stim_duration[i], stim_name[i]))
    stimuli = np.array(stimuli, dtype = dtype)    
    stimuli = np.sort(stimuli, order = ['time']) 
    
        # To figure out, roughly, which sounds are due to stimuli
    sound_playback = np.zeros(len(sound_onset), dtype = np.int) - 1
    for i in range(len(sound_onset)):
        for j in range(len(stimuli)):
            time_diff = (((sound_onset[i] + sound_offset[i])/ 2) / fs_mic) - (stimuli['time'][j] + (stimuli['duration'][j] / 2))
            if np.abs(time_diff) < ((((sound_offset[i]-sound_onset[i]) / 2) / fs_mic) + (stimuli['duration'][j] / 2)): # vocalization overlaps with a stimuli
                sound_playback[i] = j
                
    return stimuli, stim_env, sound_playback 
        
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print 'x = %.3f, y = %.1f'%(
        ix, iy)
    global coords
    global cid
    coords.append((ix, iy))
    if len(coords) == 2:
        spect_figure.canvas.mpl_disconnect(cid)
    return

def plot_stim_and_mic(onsets_to_plot,sound_playback, stimuli, sound_onset, sound_offset, close_fig = 1, pause_time = .1):
    "Pass an index corresponding to sound_onset"
    for onset in onsets_to_plot:
        if 'sp_stim' in globals():
            close(sp_stim)
        sp_stim_fig = figure('sound pressure of stimuli, if present, sound onset = %s' %onset)
        hold
        offset = 0
        if sound_playback[onset] > -1: # if there is a stimulus
            stim_start = round(stimuli['time'][sound_playback[onset]]*fs_mic)
            offset = stim_start - sound_onset[onset] # align stimulus and sound_onset
            plot(stim_env[sound_playback[onset]],'r')
            if len(stim_env[sound_playback[onset]]) > (sound_offset[onset] - sound_onset[onset]): # check if stimulus somehow goes past sound_offset
                sound_end = sound_onset[onset] + offset + len(stim_env[sound_playback[onset]])
                print('Warning, stimulus extends past sound_offset for trial %s' %onset)
            else:
                sound_end = sound_offset[onset] + offset
        else:
            sound_end = sound_offset[onset]  
            offset = 0
        pause(.1)
     
        plot(mic[sound_onset[onset] + offset:sound_end],'b')
        pause(pause_time)
        if close_fig:
            close(sp_stim_fig) 
    return
def user_select_templates(sound_playback, stim_env, stimuli, mic, sound_onset, sound_offset, fs_mic, avoid_stims = 0, sound_length_thresh = 0):
    """ 
        Displays a series of spectrograms based on sound_onset (this is not adaptable yet), 
        asking the user which ones to use as templates and then asks the user to select the 
        area of interest.
        Returns lists of template associated vars:
            template = list()
            templ_t = list()
            templ_freq = list()
            templ_timefreq = list()
            template_index = list()
            
        Set avoid_stims to 1 if you want to only browse sound onsets that don't correspond to a stimulus
        
    """  
    global cid
    global coords
    template_wav = list()
    templ_t = list()
    templ_freq = list()
    templ_timefreq = list()
    template_index = list()
    template_starts = list()
    template_ends = list()
    i = -1
    keep_looking = 1
    while keep_looking and i < len(sound_onset-2):
        i += 1
        if avoid_stims and sound_playback[i] > 0:
            print('Skipping trial %s, stimulus detected' %i)
        elif sound_offset[i] - sound_onset[i] < sound_length_thresh * fs_mic: # only implementing for long templates at the moment
            print('Skipping trial %s, vocalization too short' %i)
        else:
            plot_stim_and_mic([i], sound_playback, stimuli, sound_onset, sound_offset, close_fig = 1, pause_time = 1) # plot a figure of stimulus and amp wav to help user

            template_temp, templ_t_temp, templ_freq_temp, templ_timefreq_temp = get_temporary_template(sound_onset, sound_offset, fs_mic, i) #plot spectrogram on the figure 
            save_template = str()
            save_template = raw_input("Enter 1 to save, q to quit looking for templates, anything else to continue: ")  
            if save_template == '1':
                print("Click around template of interest")
                # show(spect_figure) # this breaks the onclick stuff, so I can't seem to bring the figure to the forefront
                cid = spect_figure.canvas.mpl_connect('button_press_event', onclick)
                waitforbuttonpress()
                waitforbuttonpress()
                template_start = int(np.floor(sound_onset[i] + (coords[-2][0] * fs_mic)))
                template_end = int(np.floor(sound_onset[i] + (coords[-1][0] * fs_mic)))
                templ_wav = mic[template_start:template_end]
                templ_t_temp, templ_freq_temp, templ_timefreq_temp, templ_rms = spectrogram(templ_wav, fs_mic, spec_sample_rate = 1000, freq_spacing = 50)
                template_wav.append(templ_wav)
                templ_t.append(templ_t_temp)
                templ_freq.append(templ_freq_temp)
                templ_timefreq.append(templ_timefreq_temp)
                template_index.append(i)
                template_starts.append(template_start)
                template_ends.append(template_end)
            elif save_template == 'q':
                keep_looking = 0
    return template_wav, template_index, templ_t, templ_freq, templ_timefreq, template_starts, template_ends    

def append_user_select_templates(sound_playback, stim_env, stimuli, mic, sound_onset, sound_offset, fs_mic, avoid_stims, sound_length_thresh, template_wav, template_index, template_t, template_freq, template_timefreq, template_starts, template_ends):
    """ 
       appends more templates to above, useful for saerching for templates towards teh end of the file or something.       
    """  
    global cid
    global coords
    i = template_index[-1] # start where you left off searching 
    keep_looking = 1
    while keep_looking and i < len(sound_onset-2):
        i += 1
        if avoid_stims and sound_playback[i] > 0:
            print('Skipping trial %s, stimulus detected' %i)
        elif sound_offset[i] - sound_onset[i] < sound_length_thresh * fs_mic: # only implementing for long templates at the moment
            print('Skipping trial %s, vocalization too short' %i)
        else:
            plot_stim_and_mic([i], sound_playback, stimuli, sound_onset, sound_offset, close_fig = 1, pause_time = 1) # plot a figure of stimulus and amp wav to help user
            template_temp, templ_t_temp, templ_freq_temp, templ_timefreq_temp = get_temporary_template(sound_onset, sound_offset, fs_mic, i) #plot spectrogram on the figure 
            save_template = str()
            save_template = raw_input("Enter 1 to save, q to quit looking for templates, anything else to continue: ")  
            if save_template == '1':
                print("Click around template of interest")
                # show(spect_figure) # this breaks the onclick stuff, so I can't seem to bring the figure to the forefront
                cid = spect_figure.canvas.mpl_connect('button_press_event', onclick)
                waitforbuttonpress()
                waitforbuttonpress()
                template_start = int(np.floor(sound_onset[i] + (coords[-2][0] * fs_mic)))
                template_end = int(np.floor(sound_onset[i] + (coords[-1][0] * fs_mic)))
                templ_wav = mic[template_start:template_end]
                templ_t_temp, templ_freq_temp, templ_timefreq_temp, templ_rms = spectrogram(templ_wav, fs_mic, spec_sample_rate = 1000, freq_spacing = 50)
                template_wav.append(templ_wav)
                template_t.append(templ_t_temp)
                template_freq.append(templ_freq_temp)
                template_timefreq.append(templ_timefreq_temp)
                template_index.append(i)
                template_starts.append(template_start)
                template_ends.append(template_end)
            elif save_template == 'q':
                keep_looking = 0  
    return template_wav, template_index, template_t, template_freq, template_timefreq, template_starts, template_ends    

def common_elements(a, b):
  a.sort()
  b.sort()
  i, j = 0, 0
  common = []
  while i < len(a) and j < len(b):
    if a[i] == b[j]:
      common.append(a[i])
      i += 1
      j += 1
    elif a[i] < b[j]:
      i += 1
    else:
      j += 1
  return common

def plot_high_corrs(high_corrs, template, mic, template_corrs, amplitude_corrs, corr_ind, amp_ind):
    """
        Aligns and plots a given template against the microphone channel. Returns alignment points.
        Requires high_corrs, an array of which sounds you want to align (index of sound_onsets), 
        and template (int), the template you are matching to.
        
    """
    alignments = list()
    figure(1)
    title('template')
    plot_zscore_spectrogram(template_t[template], template_freq[template], template_timefreq[template])   
    for sound in high_corrs:
        align = corr_ind[template][sound] 
        good_align = np.int(sound_onset[sound]) + align - np.int(np.round(.5 * len(template_correlations[template][sound]))) 
        sound_align_wav = mic[good_align - fs_mic:good_align + 2 * fs_mic]
        sound_t, sound_freq, sound_timefreq, sound_rms = spectrogram(sound_align_wav, fs_mic, spec_sample_rate = 1000, freq_spacing = 50)
#==============================================================================
#         figure(2)    
#         plot_spectrogram(sound_t, sound_freq, sound_timefreq)
#         pause(.1)
#         close(2)
#==============================================================================
        alignments.append(good_align)   
    return alignments
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
# pull the microphone channel, filter it etc
mic = [asig for asig in block.segments[0].analogsignalarrays if asig.name == "Board ADC"][0] # This is the entire microphone recording
fs_mic = np.int(mic.sampling_rate)
t_mic = np.asarray(mic.times)
mic = mic.squeeze()
if normalize_mic:
    mic = mic - np.mean(mic) # because our mic channel often has 1.652 v of dc leak from somewhere!
vocal_band = lowpass_filter(mic, fs_mic, vb_lowpass)
vocal_band = highpass_filter(vocal_band, fs_mic, vb_highpass)
unfiltered_mic = mic
mic = lowpass_filter(mic, fs_mic, mic_lowpass)
mic = highpass_filter(mic, fs_mic, mic_highpass)
mic_env = lowpass_filter(np.abs(mic), fs_mic, 250.0)
vocal_band = np.abs(vocal_band)
vocal_band = lowpass_filter(vocal_band, fs_mic, vb_low)

# find periods of sound on the mic channel that is likely vocalizations
 # sound_onset is in data points, not time
sound_onset, sound_offset = find_vocal_periods(vocal_band, vb_stds, vd_stds, vd_win, fs_mic, onset_shift)

if shorten:
    sound_onset = sound_onset[0:18]
    sound_offset = sound_offset[0:18]
    mic = mic[0:sound_offset[17]+fs_mic]
    
# Now let's identify which vocalizations are actually playbacks!!!
# first i need stimuli in a format i can use, so make a sortable array, 'stimuli', 
# containing the time of the stimuli ('time'), duration 'duration' and 'name'.
# It is sorted in order of time. stim_env is a list of all the amplitude envelopes, sorted as stimuli is.
# 'sound_playback' stores whether a given vocalization (sound_onset) corresponds to a playback or not.
# sound_playback = -1 if no playback matches the time of vocalization, and returns
# the index of stimuli otherwise. 
# thus if sound_playback[1] = 10, stim_env[10] corresponds to sound_playback[1]
stimuli, stim_env, sound_playback = organize_playbacks(segment, sm, sound_onset, sound_offset, fs_mic)
nonplayback = np.squeeze(np.where(sound_playback < 0)) # index of sound playbacks

# thus:
onsets_to_plot = [18,19,20] # pick some onsets to plot
plot_stim_and_mic(onsets_to_plot, sound_playback, stimuli, sound_onset, sound_offset) # plot them aligned with stimuli, if present
    
# ### beginning user controlled template interfaces
    
spect_figure = figure('spectogram') #this figure will be used to define the template
coords = []
cid = int
template_wav, template_index, template_t, template_freq, template_timefreq, template_starts, template_ends = user_select_templates(sound_playback, stim_env, stimuli, mic, sound_onset, sound_offset, fs_mic, avoid_stims, sound_length_thresh)

spect_figure = figure('spectogram') #this figure will be used to define the template
coords = []
cid = int
sound_length_thresh = 1.0
avoid_stims = 1
template_wav, template_index, template_t, template_freq, template_timefreq, template_starts, template_ends = append_user_select_templates(sound_playback, stim_env, stimuli, mic, sound_onset, sound_offset, fs_mic, avoid_stims, sound_length_thresh, template_wav, template_index, template_t, template_freq, template_timefreq, template_starts, template_ends)

# # find the frequencies of interest and then z-score the templates\
# TODO Make this a function
#==============================================================================
freq_index = []
num_templates = len(template_timefreq)
for i in range(len(template_freq[0])): # frequencies in template spectrograms
    if (template_freq[0][i] > low_freq_corr) & (template_freq[0][i] < high_freq_corr): # change this to np.where FUTURE
        freq_index.append(i)
for i in range(num_templates): # loop through each template
    template_freq[i] = template_freq[i][freq_index] # cut to only frequencies of interest
    template_timefreq[i] = template_timefreq[i][freq_index,:] # cut spectrogram as above
    template_timefreq[i] = zscore(np.abs(template_timefreq[i]), axis = None) # zscore whole spectrogram

#==============================================================================
# # just to plot the spectrograms, make sure they still look good
# for i in range(num_templates):
#     plot_spectrogram(templ_t[i], template_freq[i], template_timefreq[i], dBNoise=80, colorbar = False)
#==============================================================================
    # waitforbuttonpress()
# TODO calculate xcorr of smoothed envelopes to decide how to divide up longer vocalizations

#==============================================================================
# loop through every sound_onset and calculate correlations with templates 
# TODO ok, gotta figure this out- too many NANs in the maxes. TODOTODO TODO
# template_correlations will be an array of all crosscorrs with no zeros
# template_corr_peak may give an idea of how many template matches are in a given vocalization segment (multiple peaks)

template_correlations = list()
template_corr_peak = list()
amplitude_correlations = list()
for template in range(num_templates):   # the number of templates, 
    corr_list = list()
    amplitude_list = list()
    peak_list = list()
    longest_corr = 0
    longest_amp_corr = 0
    for i in range(len(sound_onset)):
        sound_wav = mic[int(sound_onset[i]):int(sound_offset[i])]
        sound_t, sound_freq, sound_timefreq, rms = spectrogram(sound_wav, fs_mic, spec_sample_rate, freq_spacing)
        sound_timefreq = sound_timefreq[freq_index,:]
        sound_timefreq = zscore(np.abs(sound_timefreq), axis = None) 
        templ_len = len(template_t[template])
 #       spect_corr = get_symmetric_spect_corr(sound_wav, template_wav[template], freq_index, fs_mic)
        spect_corr, amplitude_corr = get_symmetric_spect_corr(sound_timefreq, template_timefreq[template], templ_len, fs_mic)

#        amp_corr = get_amplitude_corr(sound_timefreq, template_timefreq, templ_len, fs_mic) 
        #spect_corr = get_spect_corr(sound_wav, template_timefreq[template], freq_index, fs_mic)
        corr_list.append(spect_corr)
        amplitude_list.append(amplitude_corr)
        peaks = argrelextrema(spect_corr, np.greater, order = 100) # seems to work ok / except now it maybe doesn't- anyway, this is not max_corrs, it'sfor tracking multiple renditions in a sound period (not working yet)
        peak_list = np.append(peak_list, peaks)

        if len(spect_corr) > longest_corr: # just keeps track of the longest corr so we can pad the others with NAN later
            longest_corr = len(spect_corr)
        if len(amplitude_corr) > longest_amp_corr:
            longest_amp_corr = len(amplitude_corr)

    # to vstack corr_list the elements must all be the same size 
    corr_list_noappend = corr_list
    for j in range(len(corr_list)):
        if len(corr_list[j]) < longest_corr:
            zero_pad = np.zeros(longest_corr - len(corr_list[j]))
            corr_list[j] = np.append(corr_list[j], zero_pad)
    corr_list_array = np.vstack(corr_list)
    
    amplitude_list_noappend = amplitude_list
    for j in range(len(amplitude_list)):
        if len(amplitude_list[j]) < longest_amp_corr:
            zero_pad = np.zeros(longest_amp_corr - len(amplitude_list[j]))
            amplitude_list[j] = np.append(amplitude_list[j], zero_pad)
    amp_list_array = np.vstack(amplitude_list)
#==============================================================================
    # make zeros = NAN; Even though there are some NANs we need to find the remaining zeros and set them to NAN
    corr_len_nonan = np.zeros(len(corr_list_array)) # just tracks nans? 
    for i in range(len(corr_list_array)):
        for j in range(corr_list_array.shape[1]):
            if corr_list_array[i,j] == 0:
                corr_list_array[i,j] = np.nan
            else:
                corr_len_nonan[i] = corr_len_nonan[i] + 1
    amp_len_nonan = np.zeros(len(amp_list_array)) # just tracks nans? 
    for i in range(len(amp_list_array)):
        for j in range(amp_list_array.shape[1]):
            if amp_list_array[i,j] == 0:
                amp_list_array[i,j] = np.nan
            else:
                amp_len_nonan[i] = corr_len_nonan[i] + 1
    template_correlations.append(corr_list_array)
    template_corr_peak.append(peak_list)     # this is giving weird numbers  DONT USE IT FOR NOW    
    amplitude_correlations.append(amp_list_array)
    

 
#==============================================================================
## calculate max correlations for each template/sound pair. This is the number for classifying
 # stored in max_corrs (FFT corrs) and max_amp_corrs (smoothed amplitude waveform corrs)
 # alignment times (index of peaks) is stored in corr_ind and amp_ind
mean_corrs = list() # holds mean arrays for each template
max_corrs = list()
max_amp_corrs = list()
corr_ind = list()
amp_ind = list()
for template in range(len(template_correlations)):
    temp_mean_corrs = list() # holds mean arrays for each template
    temp_max_corrs = list()
    temp_max_amp_corrs = list()
    temp_amp_ind = list()
    temp_corr_ind = list()
    for i in range(len(template_correlations[template])):
        temp_mean_corrs.append(np.nanmean(template_correlations[template][i]))
        temp_max_corrs.append(np.nanmax(template_correlations[template][i]))  
        temp_max_amp_corrs.append(np.nanmax(amplitude_correlations[template][i]))     
        temp_amp_ind.append(np.nanargmax(amplitude_correlations[template][i]))
        temp_corr_ind.append(np.nanargmax(template_correlations[template][i]))
    mean_corrs.append(temp_mean_corrs)
    max_corrs.append(temp_max_corrs)
    max_amp_corrs.append(temp_max_amp_corrs)
    corr_ind.append(temp_corr_ind)
    amp_ind.append(temp_amp_ind)
    

#=======NOT WORKING=======================================================================
# # plot the max correlation for each sound_onset against each template    
# for template in range(len(max_corrs)):  # for instance
#     x = np.asarray(range(len(max_corrs[template]))) # x values for each sound period
#     figure()
#     hold
#     plot(x,max_corrs[template],'o')
#     pause(.1)
#     plot(x[nonplayback],np.squeeze(max_corrs[template])[nonplayback],'or')
#     
#     
#==============================================================================
    
# For pickling important variables (obviously they need to be saved to the H5 file eventually)
with open('BlYe0923__160829_144329_11_01', 'w') as f:  
    pickle.dump([sound_onset, sound_offset, template_correlations, amplitude_correlations, amp_ind, corr_ind, template_t, template_freq, template_timefreq, nonplayback, max_corrs, max_amp_corrs], f)

# Getting back the objects:
with open('BlYe0923__160829_144329_11_01') as f:  # Python 3: open(..., 'rb')
    sound_onset, sound_offset, template_correlations, amplitude_correlations, amp_ind, corr_ind, template_t, template_freq, template_timefreq, nonplayback, max_corrs, max_amp_corrs = pickle.load(f)

#==============================================================================
# THIS IS THE END OF THE WORKING CODE< BLEOW IS THE MOST GENERIC PLOTS
# =============================================================================
    
# draw the spectrogram of the templates to remind yourself what they look like, if you want
plot_template_specs(template_t, template_freq, template_timefreq)
# =============================================================================
# At this point we should be able to do some catagorizing of vocalizations based
# on the peax correlations (max_corrs) of the each vocalization with each templates
# Plot the template correlations against each other to try to sort the vocalizations:
# plot the max correlations between each template and each vocalization, seems best
plot(max_corrs[0], max_corrs[1],'o')    
plot(np.squeeze(max_corrs[0])[nonplayback],np.squeeze(max_corrs[1])[nonplayback],'o')
# or a 3d plot

x = 0
y = 3
z = 4
fig = figure() ## spect_corrs
ax = fig.add_subplot(111, projection = '3d')
plot(max_corrs[x] ,max_corrs[y], max_corrs[z],'or')
plot(np.squeeze(max_corrs[x])[nonplayback],np.squeeze(max_corrs[y])[nonplayback], np.squeeze(max_corrs[z])[nonplayback],'o')
title('spectral correlations, blue nonplayback, red has some playback')
xlabel('x-axis')
ylabel('y-axis')

figamp = figure() ## amplitude corrs
ax = figamp.add_subplot(111, projection = '3d')
plot(max_amp_corrs[x] ,max_amp_corrs[y], max_amp_corrs[z],'or')
plot(np.squeeze(max_amp_corrs[x])[nonplayback],np.squeeze(max_amp_corrs[y])[nonplayback], np.squeeze(max_amp_corrs[z])[nonplayback],'o')
title('amplitude correlations, blue nonplayback, red has some playback')
xlabel('x-axis')
ylabel('y-axis')

# NOW IT'S A SERIES OF LITTLE PLOTTING SCRIPTS THAT ARE VERY SPECIFIC FOR PARTICULAR THINGS
# NOW IT GETS REAL VAGUE# NOW IT GETS REAL VAGUE# NOW IT GETS REAL VAGUE# NOW IT GETS REAL VAGUE
# in general store matches in high_corrs, and use plot_high_corrs to visualize some time-aligned spectrograms
# Threshold based on two templates and threshhold values and then plot some specs
#template = [3, 4]
template = 3
corr_thresh = [12000, 12000]
high_corrs = np.squeeze(np.where( (np.asarray(max_corrs[template] > corr_thresh[0])) and (np.asarray(max_corrs[template]) > corr_thresh[1])))
alignments = plot_high_corrs(high_corrs, template, mic, template_correlations, amplitude_correlations, corr_ind, amp_ind)


x = 0
y = 3
z = 4
fig = figure() ## spect_corrs
ax = fig.add_subplot(111, projection = '3d')
plot(np.squeeze(max_corrs[x])[high_corrs], np.squeeze(max_corrs[y])[high_corrs], np.squeeze(max_corrs[z])[high_corrs],'or')
plot(np.squeeze(max_corrs[x])[nonplayback],np.squeeze(max_corrs[y])[nonplayback], np.squeeze(max_corrs[z])[nonplayback],'o')
title('spectral correlations, blue nonplayback, red has some playback')
xlabel('x-axis')
ylabel('y-axis')

high_nonplayback = common_elements(high_corrs, nonplayback)



for template in range(7):
    t, freq, timefreq, rms = spectrogram(template_wav[template], fs_mic, 1000, 50)
    figure()
    plot_spectrogram(t, freq, timefreq, dBNoise=80, colorbar = False)



# THIS find long or short vocalizations, thresholds, and plots them
# THIS find long or short vocalizations, thresholds, and plots them# THIS find long or short vocalizations, thresholds, and plots them


# To plot some long onsets, high correlations, non playbacks:
short_call = 0
song = 1

if short_call:
    long_thresh = -.3 # negative means less than (~short thresh)
    corr_thresh = .4
    template = 0
elif song:
    long_thresh = .5 # negative means less than (~short thresh)
    corr_thresh = .6
    template = 2


sound_lengths = (sound_offset-sound_onset)/ fs_mic # in s
if long_thresh > 0:
    long_sounds = np.squeeze(np.where(sound_lengths > long_thresh)) # in s
else:
    long_sounds = np.squeeze(np.where(sound_lengths < - long_thresh)) # in s

high_corrs = np.squeeze(np.where(np.asarray(max_corrs[template]) > corr_thresh))
long_high = common_elements(long_sounds, high_corrs)
long_high_vocal = common_elements(long_sounds, nonplayback)
for sound in long_high_vocal:
    sound_wav = mic[np.int(sound_onset[sound]):np.int(sound_offset[sound])]
    sound_t, sound_freq, sound_timefreq, sound_rms = spectrogram(sound_wav, fs_mic, spec_sample_rate = 1000, freq_spacing = 50)
    figure(1)    
    plot_spectrogram(sound_t, sound_freq, sound_timefreq)
    pause(.2)
    close(1)
    
    
    
    
for sound in long_sounds:
    sound_wav = mic[np.int(sound_onset[sound]):np.int(sound_offset[sound])]
    sound_t, sound_freq, sound_timefreq, sound_rms = spectrogram(sound_wav, fs_mic, spec_sample_rate = 1000, freq_spacing = 50)
    figure(1)    
    plot_spectrogram(sound_t, sound_freq, sound_timefreq)
    pause(.1)
    close(1)
    
    
    
    
    
fig = figure()
ax = fig.add_subplot(111, projection = '3d')
plot(max_corrs[2][long_high_vocal], max_corrs[3][long_high_vocal], sound_lengths[long_high_vocal], 'o')
xlabel('corr0')
ylabel('corr1')
# zlabel('duration')  # doesn't work


# find max_corrs of second template (thuck) that are both very large and are not playbacks
# this is very particular to the grant but the idea is generalizable
matches = np.where(np.asarray(max_corrs[1])[nonplayback] > 0.2)
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

#==============================================================================
# for i in range(5):
#     sound_wav=mic[sound_onset[i]:sound_offset[i]]
#     sound_t, sound_freq, sound_timefreq, rms = spectrogram(sound_wav, fs_mic, spec_sample_rate, freq_spacing)
#     sound_timefreq = sound_timefreq[freq_index,:]
#     sound_timefreq = zscore(sound_timefreq, axis = None)
#     print('lenght %d %d  %d' %(len(sound_wav), sound_timefreq.shape[1], i))
#==============================================================================

# get templates (old hard-wired style)
#==============================================================================
# template = list()
# templ_t = list()
# template_freq = list()
# template_timefreq = list()
# template_index = [17, 2]
# for i in template_index:
#     template_temp, templ_t_temp, template_freq_temp, template_timefreq_temp = get_temporary_template(i)
#     template.append(template_temp)
#     templ_t.append(templ_t_temp)
#     template_freq.append(template_freq_temp)
#     template_timefreq.append(template_timefreq_temp)
#==============================================================================


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
    
