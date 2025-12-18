import pandas as pd
import numpy as np

counties = pd.read_csv('C:\\Users\\Jingy\\Documents\\PatchSim-master\\VA_Counties.txt',header=None)
pop_VA = pd.read_csv('C:\\Users\\Jingy\\Documents\\PatchSim-master\\population_VA.csv')

selected_col = ['fips','pop2023']
valid_133 = pop_VA[selected_col]

valid_133.to_csv('valid_counties_in_VA_population.csv')

counties_adj = pd.read_csv('C:\\Users\\Jingy\\Documents\\PatchSim-master\\PatchSim-master\\county_adjacency2010.csv')
counties_adj = counties_adj.loc[(counties_adj['fipsneighbor'] >= 51000) & (counties_adj['fipsneighbor'] <= 51999)]
counties_adj = counties_adj.loc[(counties_adj['fipsneighbor'] != 51515)] # 51515 is bedford, 51019 should be bedford actually

counties_adj.to_csv('counties_adj_VA.csv')

# 133 counties in VA


