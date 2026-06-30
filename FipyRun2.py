# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:53:03 2023

@author: jmkoe

This code launches a concurrent futures to run simultaneous models from the
PhaseModel2.py file.  This is the main code that is called in the csh bash file
that runs the codes on the cluster.
"""

# Load in necessary libraries
import concurrent.futures
import numpy
import pandas as pd
import PhaseModel2


#Initialize parameters needed for the modeling.  The parameters are stored in
#a csv file.  The csv is organized as one row for parameter name and multiple 
#underlying rows with parameter values.
param = pd.read_csv('params2.csv')

#Create a matrix that allows you to pass a phase field initialization.  See the
#intialization code in the folder initialize for more information.
inputphi=[]
for i1 in range(len(param.loc[:,'x0'])):
        inputphi.append(numpy.single(pd.read_parquet(r'initialize/8x8/phii')))

# The following executable calls the phasemodel18 code to run in a parallel manner.
# The number of parallel processes can be modified for the computational load desired.
if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=(8)) as executor:
        output = [executor.submit(PhaseModel2.phasemodel,param,inputphi[i].reshape(-1),i)\
                  for i in range(len(param.loc[:,'x0']))]
