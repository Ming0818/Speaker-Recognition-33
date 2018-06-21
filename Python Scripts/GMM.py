"""
from sklearn.mixture import GaussianMixture as GM
import csv

with open('Training Data.csv', 'rU') as f:
    X = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]

with open('Testing Data.csv', 'rb') as f1:
    newX = [list(map(float,rec)) for rec in csv.reader(f1, delimiter = ',')]

gmm = GM()
gmm.fit([X[i][0:12] for i in range(len(X))])                                        
newY = [newX[i][13] for i in range(len(newX))]
out = gmm.predict([newX[i][0:12] for i in range(len(newX))])
print out

#F- SCORE
p = 0
n = 0
tp = 0
tn = 0

for i in range(len(newY)):
    if newY[i] == 1:
        p = p+1
        if newY[i] == out[i]:
            tp = tp + 1;
    else:
        n = n+1
        if newY[i] != out[i]:
            tn = tn + 1;
N = p+n
fp = n-tn
fn = p-tp

tp_rate = tp/p
tn_rate = tn/n

accuracy = (tp+tn)/N
print accuracy

sensitivity = tp_rate
specificity = tn_rate
precision = tp/(tp+fp)
recall = sensitivity
f_measure = 2*((precision*recall)/(precision + recall))
gmean = sqrt(tp_rate*tn_rate)
EVAL = [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]
print EVAL
"""
import numpy as np
from sklearn import preprocessing
from python_speech_features import mfcc
import os
import csv
from pyAudioAnalysis import audioBasicIO
import cPickle
from scipy.io.wavfile import read
from sklearn.mixture import GMM
import warnings
warnings.filterwarnings("ignore")
cnt = 0
#deltas = [] 
#eatures = []\
"""
features = np.asarray(())

for subdir, dirs,  files in os.walk("/media/shanty/Shanty/MiniProject_Speaker_Recognition/Datasets/voxceleb1_wav/Billie_Joe_Armstrong/Training Data"):
	for name in files:
	    if name.endswith('.wav'):
	    	[Fs, x] = audioBasicIO.readAudioFile(name)
	    	mfcc_feat = mfcc(x, Fs)#, 0.025, 0.01,20,appendEnergy = True) 
	    	mfcc_feat = preprocessing.scale(mfcc_feat)
	    	rows,cols = mfcc_feat.shape
        	deltas = np.zeros((rows,20))
        	N = 2   
            print("hi")
            for i in range(rows):
                index = []
                j = 1
                while j <= N:
                    if i-j < 0:
                        first = 0
                    else:
                        first = i-j
                
                    if i+j > rows-1:
                        second = rows-1
                    else:
                        second = i+j  
                        index.append((second,first))
                    j+=1
        	    deltas[i] = (mfcc_feat[index[0][0]]-mfcc_feat[index[0][1]] + (2 * (mfcc_feat[index[1][0]]-mfcc_feat[index[1][1]])) ) / 10
		    feat = np.hstack((mfcc_feat,deltas))
            print len(feat)
            print "\n"
            features[cnt] = feat
            cnt = cnt + 1
"""
with open('Training Data.csv', 'rU') as f:
    X = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]
gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
#featuresArray = np.array(features, ndmin = 2)
gmm.fit(X)
picklefile = "BillieJoeArmstrong.gmm"
cPickle.dump(gmm,open(picklefile,'w'))
"""

			