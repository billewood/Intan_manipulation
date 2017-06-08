# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:22:46 2017

@author: billewood
"""

import os
from matplotlib.pyplot import *
import quantities as pq
from zeebeez.tdt2neo import stim, import_multielectrode
from neo.core import EpochArray
from neo.io import RHDIO, NeoHdf5IO
from neosound import sound_manager
from neosound.sound_store import HDF5Store
# %matplotlib notebook
from Tkinter import Tk
from tkFileDialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)

experiment_dir = "/auto/tdrive/billewood/intan data/LBlYel0923"
experiment_file = os.path.join(experiment_dir, "RHD", "LBlYe0923__160831_102048.rhd")
stimulus_file = os.path.join(experiment_dir, "pyoperant", "LBlYel0923_trialdata_20160831102040.csv")
stimulus_dir = os.path.join(experiment_dir, "Stimuli", "RepertoireShortFS25k")