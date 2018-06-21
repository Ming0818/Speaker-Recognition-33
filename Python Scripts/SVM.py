from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
import csv

print 'Import Done \n'

with open('Training2.csv', 'rU') as f:
    data = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]

with open('Testing2.csv', 'rb') as f1:
    test = [list(map(float,rec)) for rec in csv.reader(f1, delimiter = ',')]

print 'Data accumulation done \n'

model = svm.SVC(kernel = 'rbf')
print 'Model Declaration Done \n'


model.fit([data[i][0:11] for i in range(len(data))], [data[i][12] for i in range(len(data))])  
print 'Model fitting Done \n'                                      
testY = [test[i][12] for i in range(len(test))]
result = model.predict([test[i][0:11] for i in range(len(test))])
print 'Prediction Done \n'

precision, recall, fscore, support = score(testY, result)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

tn, fp, fn, tp = confusion_matrix(testY, result)
print tp, '\n', fp, '\n', tn, '\n', fn
