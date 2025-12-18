import pandas as pd
import numpy as np
import scipy as sp
# Use UMD features

county_features = pd.read_csv('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\UMD-20210420-field_county.csv')
with open('C:\\Users\\Jingy\\Documents\\HyperSCI-master\\HyperSCI-master\\VA_county_FIPS.txt') as f:
    va_fips = f.read()

population_va = pd.read_csv('population_VA.csv')
va_fips = population_va.fips.values
adj_va = pd.read_csv('counties_adj_VA.csv')

adj_va = adj_va[adj_va['fipscounty']!=51515] #51515 is bedford city, and it's in 51019 bedford county, so 51515 is removed.
adj_va = adj_va.iloc[:,1:]


assert len(np.unique(adj_va['fipscounty'].values)) == len(va_fips)

va_features = county_features[county_features['fips'].isin(va_fips)]
selected_date = '2020-01-01'
va_features = va_features[va_features['date']==selected_date]

starting = va_features.iloc[0:133,:]
starting.columns = ['FIPS',np.unique(starting['field'].values)[0],'date','field']
starting = starting.iloc[:,0:2]

for i in range(133,len(va_features),133):
    next_sec = va_features.iloc[i:i+133,:]
    next_sec.columns = ['FIPS',np.unique(next_sec['field'].values)[0],'date','field']
    new_col = np.unique(next_sec['field'].values)[0]
    next_sec = next_sec.iloc[:,0:2]
    starting[new_col] = next_sec.iloc[:,1].values

# 133 * 40 matrix, but we still need to remove some features.
# But only some features are meaningful
selected_columns = ['FIPS', 'Social distancing index', '% staying home', 'Trips/person',
       '% out-of-county trips', '% out-of-state trips', 'Miles/person',
       'Work trips/person', 'Non-work trips/person', 'Transit mode share','Unemployment claims/1000 people', 'Unemployment rate',
       '% working from home', 'Cumulative inflation rate',
       '% change in consumption', '% people older than 60', 'Median income',
       '% African Americans', '% Hispanic Americans', '% Male',
       'Population density', 'Employment density', '# hot spots/1000 people',
    'Population']

final_va_features = starting[selected_columns]

final_va_features.to_csv('VA_features.csv')

