import pickle 
import numpy as np
import pandas as pd
import sklearn


#Code snippet to view existing results.pkl file, uncomment if have results

'''
with open('results_20confidence.pkl', 'rb') as f:
    data = pickle.load(f)

data = data.iloc[0, :]

data2 = pd.DataFrame(columns = data.index)
for col in data.index:
    for j in range(len(data[col])):
        #if col=='joint_semi':
        #    data2.loc[j, col] = data[col][j]
        #else:
        data2.loc[j, col] = data[col][j]['system accuracy']

'''

with open('temps/run0_datasize0.6_results_20confidence.pkl', 'rb') as f:
    data = pickle.load(f)



system = pd.DataFrame(columns = data.columns, index = data.index)
classifier = pd.DataFrame(columns = data.columns, index = data.index)
expert = pd.DataFrame(columns = data.columns, index = data.index)
coverage = pd.DataFrame(columns = data.columns, index = data.index)
for col in data.columns:
    for row in data.index:
        #if col=='joint_semi':
        #    data2.loc[j, col] = data[col][j]
        #else:
        system.loc[row, col] = data.loc[row][col]['system accuracy']
        classifier.loc[row, col] = data.loc[row][col]['classifier accuracy']
        expert.loc[row, col] = data.loc[row][col]['expert accuracy']
        coverage.loc[row, col] = data.loc[row][col]['coverage']

print('done')