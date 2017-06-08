# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:45:44 2017

@author: billewood

This is being made with the idea of automatically sorting songs and calls using a random forest based off of Julie's previous data
"""
from autodetect import find_vocal_periods 
import sys
import os as os
import numpy as np
import matplotlib.pyplot as plt


save_folder = '/auto/tdrive/billewood/intan data/song_sorting/from_FET_Julie'
data_folder = '/auto/tdrive/julie/AcousticRecordings/FullVocalizationBank/'

file_list = os.listdir(data_folder)
