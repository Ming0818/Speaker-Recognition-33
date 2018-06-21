
#from pyAudioAnalysis import audioBasicIO
#from pyAudioAnalysis import audioFeatureExtraction
import librosa
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
print "Import completed \n"
for subdir, dirs,  files in os.walk("/media/shanty/Shanty/MiniProject_Speaker_Recognition/Datasets/Testing_Negative"):
    for name in files:
        if name.endswith('.wav'):
            fd = open('TestNew.csv', 'a')
            y, sr = librosa.load(name)
            mfcc = librosa.feature.mfcc(y = y, sr = sr)
            shp = np.shape(mfcc)
            mfcc_delta =  librosa.feature.delta(mfcc)
            val = np.zeros((shp[1], 1))
            feat = np.hstack((np.transpose(mfcc),np.transpose(mfcc_delta), val ))
            np.savetxt(fd, feat, '%f', ',')
            fd.close()
	
