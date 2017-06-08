# Import math, plotting and sound libraries
import numpy as np
import matplotlib.pyplot as plt
from soundsig.sound import BioSound 
from soundsig.sound import WavFile
import os


# Read all the wav files in this data base to make them into biosounds

os.chdir('/Users/frederictheunissen/Documents/Data/Babies/Banque Pleurs Francais originaux full /')
plotme = False

# This will be the output directory
if not os.path.exists('h5files'):
    os.makedirs('h5files')

# Find all the wave files 
isound = 0   
for fname in os.listdir('.'):
    if fname.endswith('.wav'):
        isound += 1;
        
        # Read the sound file
        print 'Processing Baby sound %d:%s\n' % (isound, fname)
        soundIn = WavFile(file_name=fname) 
        filename, file_extension = os.path.splitext(fname) 
        
        maxAmp = np.abs(soundIn.data).max()      
    
        myBioSound = BioSound(soundWave=soundIn.data.astype(float)/maxAmp, fs=float(soundIn.sample_rate), emitter=filename, calltype = filename[0])
             
    # Create BioSound Object and store some values
        myBioSound.spectroCalc(spec_sample_rate=100)
        myBioSound.rms = myBioSound.sound.std()     # The rms
       
    # Calculate amplitude enveloppe
        myBioSound.ampenv()
       
    # Calculate the power spectrum
        myBioSound.spectrum(f_high=10000)
    
    # Calculate fundamental and related values
        myBioSound.fundest(maxFund = 1500, minFund = 100, lowFc = 50, highFc = 6000, minSaliency = 0.5)
      
    # Calculate the MPS
        myBioSound.mpsCalc(window=1.0)
        
        if plotme:  
            myBioSound.plot()  
            myBioSound.play()
            waitstr = raw_input("Press Enter to continue...")
            plt.close('all')

    # Save the results
        fh5name = 'h5files/%s.h5' % (fname)
        myBioSound.saveh5(fh5name)

