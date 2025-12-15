import json
import numpy as np
import pandas as pd

f = open('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\flipping_VA_neighbor.json')

# returns JSON object as
# a dictionary
data = json.load(f)
# Convert flipping_neighbors results to  JSON-serializable dictionary to a DataFrame
flipping_neighbors_results = {}
for key, json_data in data.items():
    flipping_neighbors_results[key] = pd.DataFrame(json_data['data'], columns=json_data['columns'],
                                                   index=json_data['index'])

# flipping_self loading

f2 = open('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\flipping_VA_self.json')
data = json.load(f2)
# Convert flipping_neighbors results to  JSON-serializable dictionary to a DataFrame
flipping_self_results = {}
for key, json_data in data.items():
    flipping_self_results[key] = pd.DataFrame(json_data['data'], columns=json_data['columns'], index=json_data['index'])

# original simulation

f3 = open('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\VA_output_true.txt')
data = np.loadtxt(f3)

# We select the 60th timestamps as a timescope, as we implement treatment from 0-10
real_under_t = data[:,25] # 0 index occupied by FIPS

real_after_flipping_t = list()
FIPS_LS = data[:,0]
for idx, f in enumerate(FIPS_LS):
    real_after_flipping_t.append(flipping_self_results[str(int(f))].iloc[idx,24])

real_after_flipping_neighbors = list()
FIPS_LS = data[:, 0]
for idx, f in enumerate(FIPS_LS):
    real_after_flipping_neighbors.append(flipping_neighbors_results[str(int(f))].iloc[idx, 24])

treatments = np.loadtxt("C:\\Users\\Jingy\\Documents\\PatchSim-master\\PatchSim-master\\manual_tests\\VA_treatment.txt")
treatments = treatments[treatments[:,0].argsort()]
T = treatments[:,1]