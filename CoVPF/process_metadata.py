import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import torch
import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

Omicron_metadata = pd.read_csv('./metadata_Omicron.csv', header=0)

dic = {
    'Time': Omicron_metadata['Time'],
    'Place': Omicron_metadata['Place'],
    'Lineage': Omicron_metadata['Lineage'],
    'Mutation': Omicron_metadata['Mutation']
    }
experimental_data = pd.DataFrame(dic)

print('Country extract')
experimental_data['Country'] = ''
for i in range(len(experimental_data['Place'])):
    index_1 = experimental_data['Place'][i].find('/')
    index_2 = experimental_data['Place'][i][index_1 + 1: ].find('/')
    if index_1 != -1 and index_2 != -1:
        experimental_data['Country'][i] = experimental_data['Place'][i][(index_1 + 1) :(index_1 + index_2 + 1)]
    else:
        experimental_data['Country'][i] = experimental_data['Place'][i][(index_1 + 1) :]
    if i % 100000 == 0:
        print(f" step {i} completed.")
print('Country extract completed')


print('mutation extract')
experimental_data['Spike_mutations'] = ''
experimental_data['RBD_mutations'] = ''
temp = 'Spike_'

for i in range(len(experimental_data['Spike_mutations'])):
    Spike_mutations = []
    RBD_mutations = []
    tS = str(experimental_data['Mutation'][i]).split(',')
    for j in tS:
        index = j.find(temp)
        if index != -1:
            Spike_mutations.append(j)
            # if j[len(temp):] == 'ins214EPE' or j[len(temp):] == 'ins213GEG' or j[len(temp):] == 'ins214XXX' or j[len(temp):] == 'ins473LPTI':
            if j[len(temp) : len(temp) + 3] == 'ins' or j[-4:] == 'stop' or j[-3:] == 'del':
                pass
            else:
                try:
                    if int(j[len(temp)+1: -1]) in range(331, 531):
                        RBD_mutations.append(j[len(temp):])
                except:
                    pass

    if i % 100000 == 0:
        print(f" step {i} completed.")

    experimental_data['Spike_mutations'][i] = Spike_mutations
    experimental_data['RBD_mutations'][i] = RBD_mutations

dic1 = {
    'Time' : experimental_data['Time'],
    'Place' : experimental_data['Country'],
    'Lineage' : experimental_data['Lineage'],
    'Mutation' : experimental_data['Spike_mutations']
}
df_dic1 = pd.DataFrame(dic1)
df_dic1.dropna(how='any', axis=0, inplace=True)

dic2 = {
    'Time' : experimental_data['Time'],
    'Place' : experimental_data['Country'],
    'Lineage' : experimental_data['Lineage'],
    'Mutation' : experimental_data['RBD_mutations']
}
df_dic2 = pd.DataFrame(dic2)
df_dic2.dropna(how='any', axis=0, inplace=True)

df_dic1.to_csv('./Omicron_spike_experimental_data.csv', index = False)
df_dic2.to_csv('./Omicron_RBD_experimental_data.csv', index = False)