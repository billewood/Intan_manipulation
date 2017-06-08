# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:15:22 2017

@author: billewood
"""
from neo.core import EpochArray
from neo.io import RHDIO, NeoHdf5IO
from neosound import sound_manager
from neosound.sound_store import HDF5Store

filename = '/auto/tdrive/carson/processed_intan/LbY6074__161215_145633_neo.h5'
iom = NeoHdf5IO(filename)
block = iom.read_blcok("/Block_0")
mic = [asig for asig in block.segments[0].analogsignalarrays if asig.name == "Board ADC"][0] # This is the entire microphone recording
pupil_song_timings = [asig for asig in block.segments[0].epocharrays if asig.name == "student_birdsong"][0]
tutor_song_timings = [asig for asig in block.segments[0].epocharrays if asig.name == "tutor_birdsong"][0]

 = np.int(mic.sampling_rate)
t_mic = np.asarray(mic.times)
too_long = too_long * fs_mic
mic = mic.squeeze()