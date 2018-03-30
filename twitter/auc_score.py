import sklearn.metrics as metrics
import numpy as np
import os

results_dir = ""
print("train,maxlen,test,type,auc")
for results in os.listdir(results_dir):
    results_file = np.genfromtxt(results_dir+results, delimiter=',')
    results = results.replace('_',',')
    results = results.replace('.csv','')
    print(results+','+str(metrics.roc_auc_score(results_file[:,2],results_file[:,1])))


