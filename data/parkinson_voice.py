# https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
# pip install ucimlrepo
# https://github.com/uci-ml-repo/ucimlrepo

import os

import json

from ucimlrepo import fetch_ucirepo 
  
  
if __name__=="__main__":
    # fetch dataset 
    parkinsons_telemonitoring = fetch_ucirepo(id=189) 
    
    # data (as pandas dataframes) 
    X = parkinsons_telemonitoring.data.features 
    y = parkinsons_telemonitoring.data.targets 
    
    # metadata 
    with open("parkinsons_telemonitoring.json", 'w') as f:
        json.dump(parkinsons_telemonitoring.metadata, f, indent=4)

    # variable information 
    print(parkinsons_telemonitoring.variables) 
