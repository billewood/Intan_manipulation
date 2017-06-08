# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 08:39:22 2016
For pulling syllable times saved with Intan_motif_matches and then plotting rasters
@author: billewood
"""

import os
from matplotlib.pyplot import *
import matplotlib.gridspec as gridspec
import quantities as pq
from zeebeez.tdt2neo import stim, import_multielectrode# Find the corresponding stim wave file and calculate its enveloppe
from neo.core import EpochArray
from neo.io import RHDIO, NeoHdf5IO
from neosound import sound_manager
from neosound.sound_store import HDF5Store
from lasp.signal import lowpass_filter, highpass_filter
from lasp.sound import spectrogram, plot_spectrogram
from lasp.hinton import hinton
import pickle


tbefore = 0.2 * pq.s # Time before begining of stimulus
tafter = 2 * pq.s # Time after end of stimulus
p_file_dir = '/auto/fhome/billewood/Code/Intan_manipulation'
p_file = 'BlYe0923__160830_120325_alignments'
experiment_dir = "/auto/tdrive/billewood/intan data/LBlYel0923"
experiment_file = os.path.join(experiment_dir, "RHD", "LBlYe0923__160830_120325.rhd")
figures_dir = os.path.join(experiment_dir, "Figures")


# Get the neural data
h5_filename = os.path.join(experiment_dir,
                           os.path.basename(experiment_file).replace(".rhd", "_neo.h5"))
h5_exporter = NeoHdf5IO(h5_filename)
block = h5_exporter.get("/Block_0")
# The block only has one segment
segment = block.segments[0]
# This is a pointer to the entire microphone recording
mic = [asig for asig in segment.analogsignalarrays if asig.name == "Board ADC"][0]
# This is a pointer the entire neural recording
amp = [asig for asig in segment.analogsignalarrays if asig.name == "Amplifier"][0]


# Get the number of trials and the sampling rate from the meta data of the digital signal (arbitrary)
digital_signals = segment.eventarrays[0]
sampling_rate = digital_signals.annotations["sampling_rate"]
ntrials = len(segment.epocharrays)

print 'Number of Trials:', ntrials
print 'Sampling rate:', sampling_rate

i = 0
for epoch in segment.epocharrays:
    i = i + 1
    if epoch.times.size == 0 :
        print 'Epoch', i, 'for trial', epoch.annotations['trial'], 'is empty'
    else:
        sound_data = sm.database.get_annotations(epoch.annotations["stim_id"])
        print 'Epoch:', i, 'Trial:', epoch.annotations['trial'], 'Stim:', sound_data['callid'], sound_data['emitter'],\
              'Start:', epoch.times, 'Duration:', epoch.durations
                    
    #########################################################################################

p_file = os.path.join(p_file_dir, p_file)
with open(p_file) as f:  
    tutee_songs, tutee_alignments, father_songs, father_alignments = pickle.load(f)

syllables = tutee_aligned_times
syllablesT = (np.asarray(syllables)/np.asarray(sampling_rate))*pq.s

# Plot the data around syllablesT
# Chose the electrode to plot
ielec = 10

# This is the entire neural recording (amp) and microphone recording (mic)
amp = [asig for asig in segment.analogsignalarrays if asig.name == "Amplifier"][0]
mic = [asig for asig in segment.analogsignalarrays if asig.name == "Board ADC"][0]

# High pass the microphone
sample_rate = float(mic.sampling_rate)
micvals = np.asarray(mic).squeeze()
micvals -= micvals.mean()
micfilt = highpass_filter(micvals, sample_rate, 400.0)
mic_env = lowpass_filter(np.abs(micfilt), float(sample_rate), 125.0)
max_mic = np.std(mic_env)

# Find good plot boundaries
sample_rate = float(amp.sampling_rate)
amp_all = np.asarray(amp)[:, ielec]
low_amp = lowpass_filter(amp_all, sample_rate, 400.0)
low_maxabs = np.std(low_amp)
high_amp =highpass_filter(amp_all, sample_rate, 400.0) 
high_maxabs = np.std(high_amp)
neural_signal_env = lowpass_filter(np.abs(high_amp), float(sample_rate), 50.0);
maxS = np.max(neural_signal_env)

# Find all the stims
all_stims = []
for epoch in segment.epocharrays:
    all_stims.append(epoch.annotations['md5'])
    
# Get the unique stims
unique_stims = np.unique(all_stims)

# Loop through unique stims and plot all the corresponding data
## PLOT spec and neural data
fig = figure(figsize=(15,10))  
gs = gridspec.GridSpec(30 * len(syllables), 1) #I added teh five to shorten for troubleshooting

for i in range(syllablesT.shape[0]):   # shortened for troubleshooting
    
    # Plot the data.
    t_start_plot = -tbefore
    t_end_plot = (syllablesT[i] + tafter)                   
    # Get the microphone envelope for the given epoch
    mic_slice = mic.time_slice(syllablesT[i]-tbefore, syllablesT[i] + tafter)
    sample_rate = mic_slice.sampling_rate
    t_mic = mic_slice.times
    mic_slice = np.asarray(mic_slice).squeeze()
    mic_slice -= mic_slice.mean()
    micfilt = highpass_filter(mic_slice, sample_rate, 400.0)
    mic_env = lowpass_filter(np.abs(micfilt), float(sample_rate), 125.0)

    # Lowpass and high pass neural recordings
    amp_slice = amp.time_slice(syllablesT[i]-tbefore, syllablesT[i] + tafter)
    sample_rate = float(amp_slice.sampling_rate)
    t_amp = amp_slice.times
    amp_slice = np.asarray(amp_slice)[:, ielec]
    high_amp =highpass_filter(amp_slice, sample_rate, 400.0) 
    neural_signal_env = lowpass_filter(np.abs(high_amp), float(sample_rate), 50.0);

    to, fo, spect, rms = spectrogram(mic_slice, sample_rate, 1000, 50)                  
    # Plot the stimulus
    ax = fig.add_subplot(gs[i*30:i*30+5,0])
    plot_spectrogram(to, fo, spect, ax=ax, 
                ticks=True, fmin=250, fmax=8000, colormap=None, colorbar=False, log = True, dBNoise = 50)

    ylabel('Frequency')
    tick_params(labelbottom='off')
            
    # Supressing signal during high noise.
    high_amp *= (1-neural_signal_env/maxS)**6
            
    # This is plotted in z-scores
    ax = fig.add_subplot(gs[30*i+6:30*i+24,0])
    ax.plot(t_amp-t_amp[0]-tbefore, high_amp/high_maxabs, linewidth = 0.5, color = 'black')
    ax.plot(t_mic-t_mic[0]-tbefore, mic_env/max_mic, linewidth = 1, color= 'red')
    ylim(-4,4)
    tick_params(labelbottom='off', labelleft='off')
#    xlim(t_start_plot, t_end_plot)

# save figures to hard drive on eps
figname = os.path.join(figures_dir, '%s_%d.eps' % ('DC', ielec)) 
savefig(figname, format='eps')
    
###########################################################################
## PLOT just neural data
fig = figure(figsize=(15,10))  
gs = gridspec.GridSpec(30 * len(syllables), 1) #I added teh five to shorten for troubleshooting

for i in range(syllablesT.shape[0]):   # shortened for troubleshooting
    
    # Plot the data.
    t_start_plot = -tbefore
    t_end_plot = (syllablesT[i] + tafter)                   
    # Get the microphone envelope for the given epoch
    mic_slice = mic.time_slice(syllablesT[i]-tbefore, syllablesT[i] + tafter)
    sample_rate = mic_slice.sampling_rate
    t_mic = mic_slice.times
    mic_slice = np.asarray(mic_slice).squeeze()
    mic_slice -= mic_slice.mean()
    micfilt = highpass_filter(mic_slice, sample_rate, 400.0)
    mic_env = lowpass_filter(np.abs(micfilt), float(sample_rate), 125.0)

    # Lowpass and high pass neural recordings
    amp_slice = amp.time_slice(syllablesT[i]-tbefore, syllablesT[i] + tafter)
    sample_rate = float(amp_slice.sampling_rate)
    t_amp = amp_slice.times
    amp_slice = np.asarray(amp_slice)[:, ielec]
    high_amp =highpass_filter(amp_slice, sample_rate, 400.0) 
    neural_signal_env = lowpass_filter(np.abs(high_amp), float(sample_rate), 50.0);

    ylabel('Frequency')
    tick_params(labelbottom='off')
            
    # Supressing signal during high noise.
    high_amp *= (1-neural_signal_env/maxS)**6
            
    # This is plotted in z-scores
    ax = fig.add_subplot(gs[30*i:30*i+29,0])
    ax.plot(t_amp-t_amp[0]-tbefore, high_amp/high_maxabs, linewidth = 0.5, color = 'black')
    ax.plot(t_mic-t_mic[0]-tbefore, mic_env/max_mic, linewidth = 1, color= 'red')
    ylim(-4,4)
    tick_params(labelbottom='off', labelleft='off')
#    xlim(t_start_plot, t_end_plot)

# save figures to hard drive on eps
figname = os.path.join(figures_dir, '%s_%d.eps' % ('DC', ielec)) 
savefig(figname, format='eps')
                        
                    
#########################################################################33#################3
                    ##############3### If there are no stims skip this, maybe skip it either way         
# Plot all the stimulation trials (if they exist)
#==============================================================================
# # Just plot for the first trial 3 (number 4). to be replaced by range(ntrials) 
# for iepoch in range(3,4):
#     # First, get the data and do some basic processing    
#     epoch = segment.epocharrays[iepoch]
#     # Print trial number 
#     sound_data = sm.database.get_annotations(epoch.annotations["stim_id"])
#     print 'Epoch:', iepoch, 'Trial:', epoch.annotations['trial'], 'Stim:', sound_data['callid'], sound_data['emitter']
# 
#     # Take a slice of the microphone and amplifier corresponding to the chosen epoch 
#     mic_slice = mic.time_slice(epoch.times-tbefore, epoch.times + epoch.durations + tafter)
#     mic_slice_sound = mic.time_slice(epoch.times, epoch.times + epoch.durations)
#     sample_rate = mic_slice.sampling_rate
#     t_mic = mic_slice.times
#     mic_slice = np.asarray(mic_slice).squeeze()
#     mic_slice -= mic_slice.mean()
#     mic_slice_sound = np.asarray(mic_slice_sound).squeeze()
#     mic_slice_sound -= mic_slice_sound.mean()
#     amp_slice = amp.time_slice(epoch.times-tbefore, epoch.times + epoch.durations+tafter)
# 
#     # Calculate the enveloppe of the microphone recording
#     mic_env = lowpass_filter(np.abs(mic_slice), float(sample_rate), 125.0)
# 
#     # Find the corresponding stim wave file and calculate its enveloppe
#     s = sm.reconstruct(epoch.annotations["stim_id"])
#     sound_data = sm.database.get_annotations(epoch.annotations["stim_id"])
#     sample_rate = sound_data['samplerate']
#     t_sound = s.times
#     sound = np.asarray(s).squeeze()
#     sound_env = lowpass_filter(np.abs(sound), float(sample_rate), 250.0)  # Sound Enveloppe
#     to, fo, spect, rms = spectrogram(sound, sample_rate, 1000, 50)
# 
#     # Lowpass and high pass neural recordings
#     sample_rate = float(amp_slice.sampling_rate)
#     t_amp = amp_slice.times
#     amp_slice = np.asarray(amp_slice)
#     low_amp = np.vstack([lowpass_filter(channel, sample_rate, 400.0) for channel in amp_slice.T]).T
#     high_amp = np.vstack([highpass_filter(channel, sample_rate, 400.0) for channel in amp_slice.T]).T
#     
#     # Print some correlations between the high amp channels 
#     corr_table = np.corrcoef(high_amp.T)
#     figure(1);
#     hinton(corr_table)
#     
#     # Generate predictions for each channel based on the other channels
#     high_amp_noise = np.zeros(high_amp.shape)
#     high_amp_std = np.std(high_amp, axis = 0)
#     high_amp_clean = np.zeros(high_amp.shape)
#     neural_signal_env = np.zeros(high_amp.shape)
#     neural_noise_env = np.zeros(high_amp.shape)   
#     neural_clean_env = np.zeros(high_amp.shape)
#     GS = 6  # Gain for the sigmoid
# 
#     for ich1 in range(16):
#         normval = 0;
#         for ich2 in range(16):
#             if ich1 == ich2:
#                 continue
#             high_amp_noise[:, ich1] += corr_table[ich1, ich2]*(high_amp_std[ich1]/high_amp_std[ich2])*high_amp[:,ich2]
#             normval += abs(corr_table[ich1, ich2])
#         high_amp_noise[:, ich1] /= normval
#         high_amp_clean[:, ich1] = high_amp[:,ich1] - high_amp_noise[:,ich1]
#         
#         # Low pass filter both clean and noise estimate
#         neural_signal_env[:, ich1] = lowpass_filter(np.abs(high_amp[:,ich1]), float(sample_rate), 100.0);
#         neural_noise_env[:, ich1] = lowpass_filter(np.abs(high_amp_noise[:,ich1]), float(sample_rate), 100.0);
#         diff_env = np.abs(neural_signal_env[:,ich1]-neural_noise_env[:,ich1])
#         # Perform a sigmoid gain adjustment that applies correction to a greater extent in high noise areas
#         maxN = np.max(neural_noise_env[:, ich1])
#         maxD = np.max(diff_env)
#         gain = (1-(diff_env/maxD)*(neural_noise_env[:,ich1]/maxN))
#         high_amp_clean[:,ich1] *= gain   
#         # sigmoid = 1/(1+np.exp(1-GS*neural_signal_env[:,ich1]/maxS))
#         # gain = neural_signal_env[:,ich1]/neural_noise_env[:,ich1]
#         # high_amp_clean[:,ich1] = high_amp[:,ich1] - sigmoid*gain*high_amp_noise[:,ich1]
#         
#         neural_clean_env[:, ich1] = lowpass_filter(np.abs(high_amp_clean[:,ich1]), float(sample_rate), 100.0);
#     # Now generate the plots
#     # Plot sound and microphone recordings
#     t_start_plot = -tbefore
#     t_end_plot = epoch.durations + tafter
#     t_start_plot = 0.6
#     t_end_plot = 1.2
#     
#     figure(2)
#     subplot(4,1,1)
#     plot(t_mic-t_mic[0]-tbefore, mic_slice, hold=False)
#     plot(t_mic-t_mic[0]-tbefore, mic_env, color="black", linewidth=2, hold=True)
#     xlim(t_start_plot, t_end_plot)
#     ylabel('Microphone Amp (V)')
# 
#     subplot(4,1,2)
#     plot(t_sound, sound, hold=False)
#     plot(t_sound, sound_env, color="red", linewidth=2, hold=True)
#     xlim(t_start_plot, t_end_plot)
#     ylabel('Sound Stimulus Amp (V)')
# 
#     subplot(4,1,3)
#     plot(t_mic-t_mic[0]-tbefore, mic_env/max(mic_env), color="black", linewidth=2, hold=False)
#     plot(t_sound, sound_env/max(sound_env), color="red", linewidth=2, hold=True)
#     xlim(t_start_plot, t_end_plot)
#     ylabel('Envelopes')
#     xlabel('Time (s)')
# 
#     subplot(4,1,4)
#     (lags, xcval, lines, bxaxis) = xcorr(mic_slice_sound, sound, maxlags=int(.01 * sample_rate), usevlines=False, 
#                                          linestyle = 'solid', marker = None)
#     xlim(-200, 200)
# 
#     # Plot the neural data - Before noise Reduction.
#     fig = figure(3, figsize=(15,15))
#     gs = gridspec.GridSpec(110, 2)
#         
#     for iplt in range(2):
#         ax = fig.add_subplot(gs[:20,iplt])
#         # subplot(9, 2, iplt+1)
#         plot_spectrogram(to, fo, spect, ax=ax, 
#                          ticks=True, fmin=250, fmax=8000, colormap=None, colorbar=False, log = True, dBNoise = 50)
#         # plot(t_mic-t_mic[0]-tbefore, mic_env/max(mic_env), color="black", linewidth=2, hold=False)
#         # plot(t_sound, sound_env/max(sound_env), color="red", linewidth=2, hold=True)
#         ylabel('Frequency')
#         tick_params(labelbottom='off')
#         title('%s %s' % (sound_data['callid'], sound_data['emitter']))
#         xlim(t_start_plot, t_end_plot)
#     # maxamp = np.amax(high_amp)
#     # minamp = np.amin(high_amp)
#     # maxabs = max(abs(maxamp), abs(minamp))
#     maxabs = np.std(high_amp)*3.0
#     for iplt in range(16):
#         # subplot(9, 2, iplt+3)
#         if iplt < 8:
#             ax = fig.add_subplot(gs[20+10*iplt:20+10*iplt+9,0])
#         else:
#             isub = iplt%8;
#             ax = fig.add_subplot(gs[20+10*isub:20+10*isub+9,1])
#         ax.plot(t_amp-t_amp[0]-tbefore, high_amp[:,iplt], linewidth = 0.5, color = 'black')
# #        ax.plot(t_amp-t_amp[0]-tbefore, high_amp_noise[:,iplt],linewidth = 0.5, color = 'red')
#         ylim(-maxabs,maxabs)
#         xlim(t_start_plot, t_end_plot)
#         if iplt != 7:
#             tick_params(labelbottom='off', labelleft='off')
#         else:
#             xlabel('Time (s)')    
#     figname = os.path.join(figures_dir, 'NeuralHighBefore_%d.eps' % epoch.annotations['trial'])
#     savefig(figname, format='eps')
# 
#             
#     # Plot the neural data - After noise Reduction.
#     fig = figure(4, figsize=(15,15))
#     gs = gridspec.GridSpec(100, 2)
#         
#     for iplt in range(2):
#         ax = fig.add_subplot(gs[:20,iplt])
#         # subplot(9, 2, iplt+1)
#         plot_spectrogram(to, fo, spect, ax=ax, 
#                          ticks=True, fmin=250, fmax=8000, colormap=None, colorbar=False, log = True, dBNoise = 50)
#         # plot(t_mic-t_mic[0]-tbefore, mic_env/max(mic_env), color="black", linewidth=2, hold=False)
#         # plot(t_sound, sound_env/max(sound_env), color="red", linewidth=2, hold=True)
#         ylabel('Frequency')
#         tick_params(labelbottom='off')
#         title('%s %s' % (sound_data['callid'], sound_data['emitter']))
#         xlim(t_start_plot, t_end_plot)
# 
# 
#     for iplt in range(16):
#         # subplot(9, 2, iplt+3)
#         if iplt < 8:
#             ax = fig.add_subplot(gs[20+10*iplt:20+10*iplt+9,0])
#         else:
#             isub = iplt%8;
#             ax = fig.add_subplot(gs[20+10*isub:20+10*isub+9,1])
#         ax.plot(t_amp-t_amp[0]-tbefore, high_amp_clean[:,iplt], linewidth = 0.5, color = 'black')
#         #ax.plot(t_amp-t_amp[0]-tbefore, neural_signal_env[:,iplt], linewidth = 0.5, color = 'blue')
#         #ax.plot(t_amp-t_amp[0]-tbefore, neural_noise_env[:,iplt], linewidth = 0.5, color = 'red')
#         #ax.plot(t_amp-t_amp[0]-tbefore, neural_clean_env[:,iplt], linewidth = 0.5, color = 'green')
#         
#         ylim(-maxabs,maxabs)
#         xlim(t_start_plot, t_end_plot)
#         if iplt != 7:
#             tick_params(labelbottom='off', labelleft='off')
#         else:
#             xlabel('Time (s)')    
#              
# 
#             
#     figname = os.path.join(figures_dir, 'NeuralHighAfter_%d.eps' % epoch.annotations['trial'])
#     savefig(figname, format='eps')
# 
# 
#     figure(5, figsize=(10,10))
#     for iplt in range(2):
#         subplot(9, 2, iplt+1)
#         plot(t_mic-t_mic[0]-tbefore, mic_env/max(mic_env), color="black", linewidth=2, hold=False)
#         plot(t_sound, sound_env/max(sound_env), color="red", linewidth=2, hold=True)
#         ylabel('Enveloppes')
#         tick_params(labelbottom='off')
#         title('%s %s' % (sound_data['callid'], sound_data['emitter']))
#         xlim(t_start_plot, t_end_plot)
# 
#     # maxamp = np.amax(low_amp)
#     # minamp = np.amin(low_amp)
#     # maxabs = max(abs(maxamp), abs(minamp))
#     maxabs = np.std(low_amp)*3.0
#     for iplt in range(16):
#         subplot(9, 2, iplt+3)
#         plot(t_amp-t_amp[0]-tbefore, low_amp[:,iplt], linewidth = 0.5, color = 'black', hold=False)
# 
#         ylim(-maxabs, maxabs)
#         xlim(t_start_plot, t_end_plot)
#         if iplt != 14:
#             tick_params(labelbottom='off', labelleft='off')
#         else:
#             xlabel('Time (s)')
#             
#     # save figures to hard drive on eps
#     # figname = os.path.join(figures_dir, 'lfpTrial%d.eps' % epoch.annotations['trial']) 
#     # savefig(figname, format='eps')
# 
#==============================================================================
###########################################################################################################################3

