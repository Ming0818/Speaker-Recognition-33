from sklearn.neural_network import MLPClassifier
import csv

print 'Import Done \n'

with open('Training Data.csv', 'rU') as f:
    data = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]

with open('Testing Data.csv', 'rb') as f1:
    test = [list(map(float,rec)) for rec in csv.reader(f1, delimiter = ',')]

print 'Data accumulation done \n'

model = MLPClassifier(solver = 'lbfgs')
model.fit([data[i][0:12] for i in range(len(data))], [data[i][13] for i in range(len(data))])
print 'Model fitting Done \n'                                      
testY = [test[i][13] for i in range(len(test))]
result = model.predict([test[i][0:12] for i in range(len(test))])
print 'Prediction Done \n'
#F- SCORE
p = 0
n = 0
tp = 0
tn = 0

for i in range(len(testY)):
    if testY[i] == 1:
        p = p+1
        if testY[i] == result[i]:
            tp = tp + 1;
    else:
        n = n+1
        if testY[i] != result[i]:
            tn = tn + 1;
N = p+n
fp = n-tn
fn = p-tp
tp_rate = tp/p
tn_rate = tn/n

accuracy = (tp+tn)/N
print 'accuracy ='
print accuracy

sensitivity = tp_rate
specificity = tn_rate
precision = tp/(tp+fp)
recall = sensitivity
f_measure = 2*((precision*recall)/(precision + recall))
gmean = sqrt(tp_rate*tn_rate)
EVAL = [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]
print EVAL

