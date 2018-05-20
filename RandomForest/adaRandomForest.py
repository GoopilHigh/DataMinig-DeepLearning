import numpy as np
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
from math import sqrt

train_data = np.genfromtxt('/home/goop/rendu/Tek_4/DataMining/DataSet/Ada/ADA/ada_train.data')
train_labels = np.genfromtxt('/home/goop/rendu/Tek_4/DataMining/DataSet/Ada/ADA/ada_train.labels')
val_data = np.genfromtxt('/home/goop/rendu/Tek_4/DataMining/DataSet/Ada/ADA/ada_valid.data')
val_labels = np.genfromtxt('/home/goop/rendu/Tek_4/DataMining/DataSet/Ada/ADA/ada_valid.labels')

model = RandomForestClassifier(n_estimators=10,max_features=round(sqrt(48)))
model.fit(train_data,train_labels)

def cal_ber(truth, predict):
    ab = len(predict[truth[:]==-1])
    cd = len(predict[truth[:]==1])
    b = len(predict[(truth[:]==-1) & (predict[:]==1)])
    c = len(predict[(truth[:]==1) & (predict[:]==-1)])
    return 0.5*(b/ab + c/cd)

val_pre = model.predict(val_data)
val_AUC = metrics.roc_auc_score(val_labels,val_pre)
val_BER = cal_ber(val_labels,val_pre)

print('AUC ',val_AUC)
print('BER ',val_BER)

print('BER + AUC =', val_BER+val_AUC)