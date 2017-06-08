# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:58:46 2017

@author: billewood
"""

import os
from matplotlib.pyplot import *
import quantities as pq
from neo.io import RHDIO, NeoHdf5IO, nixio
from neosound import sound_manager
from neosound.sound_store import HDF5Store

experiment_dir = "/auto/tdrive/billewood/intan data/LBlYel0923"
experiment_file = os.path.join(experiment_dir, "RHD", "LBlYe0923__160830_120325.rhd")
figures_dir = os.path.join(experiment_dir, "Figures")
rhd_importer = RHDIO(experiment_file)
block = rhd_importer.read_block()
segment = block.segments[0]

mic = [asig for asig in block.segments[0].analogsignals if asig.name == "Board ADC"][0] # This is the entire microphone recording
amp = [asig for asig in segment.analogsignals if asig.name == "Amplifier"][0] # This is the entire microphone recording

fs_mic = np.int(mic.sampling_rate)
t_mic = np.asarray(mic.times)
mic = mic.squeeze()


# for instance, now
hold
plot(amp[0:20,0],'r')
plot(amp[0:20,1],'y')
plot(amp[0:20,2],'g')
plot(amp[0:20,3],'c')




# not sure how to save as a nix yet.
#h5_filename = os.path.join(experiment_dir,
#                           os.path.basename(experiment_file).replace(".rhd", "_neo_nix.h5"))
#h5_exporter = nixio(h5_filename) # this won't work, needs to be nix.
                         