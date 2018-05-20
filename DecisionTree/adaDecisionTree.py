import numpy as np
from sklearn import tree, metrics

train_data = np.genfromtxt('../Dataset/ADA/ada_train.data')
train_labels = np.genfromtxt('../Dataset/ADA/ada_train.labels')
val_data = np.genfromtxt('../Dataset/ADA/ada_valid.data')
val_labels = np.genfromtxt('../Dataset/ADA/ada_valid.labels')

model = tree.DecisionTreeClassifier(criterion='gini')  
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