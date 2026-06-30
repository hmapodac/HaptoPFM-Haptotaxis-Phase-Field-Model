# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 13:52:36 2023

@author: jmkoe

This file takes the individual Fractional Occupancy data files for each time  
point and creates a master data file that can then be queried by SQL using 
duckdb for further analysis.
"""

#Load the necessary packages for running the code.
from fipy import Grid2D
import numpy
import pandas as pd
import os

#Find the folders of interest where the fractional occupancy files are at to reformat.
datloc = os.getcwd()
roots = []
dirs1 = []
files1 = []
for (root,dirs,files) in os.walk(datloc,topdown=True):
    roots.append(root)
    dirs1.append(dirs)
    files1.append(files)
rofint=[]
for i in range(len(roots)):
   if  'data' in dirs1[i]:
       rofint.append(roots[i])

#Define the meshing of the model.
#Mesh for the small 8x8 inset box where the phase field equations are solved.
wid = 8
leng = 8
dx = 0.05
dy = dx
nx = int(wid/dx)
ny = int(nx)
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
x,y = mesh.cellCenters


rofint1=[]
#Generate a loop to loop through all the data files to formulate and perform
#the restructuring of all of the data files.
for i in range(len(rofint)):
    
    #Gather the base name of the files and a file name list from the data folder.
    #All of the files are fomulated as name#.# where the #.# is the seconds in the model
    datfiles = os.listdir(rofint[i]+r'FracOcc/')
    basename = ''.join([i for i in datfiles[0] if not i.isdigit()])

    #Remove the decimal from the basename and create a numerically sorted list
    #of the data files so data masterfile reads time in order.
    #basename = basename[0:len(basename)-1]
    datnum = numpy.zeros([len(datfiles)])
    for i1 in range(len(datnum)):
        datcurr = float(datfiles[i1][len(basename):len(datfiles[i1])])
        datnum[i1]=datcurr
    order = numpy.argsort(datnum)
    datfiles1=[datfiles[order[i1]] for i1 in range(len(order))]
    
    #dat1 = pd.read_parquet(rofint[i]+r'/C/'+datfiles1[0])
    dat1 = pd.read_parquet(rofint[i]+r'/FracOcc/'+datfiles1[0])
    
    #Create (t,x,y) vectors for the entire data file and put it into a pandas
    #dataframe. The dataframe will be built upon with the modeling data. The 
    #(x,y) information and even possibly the t could be removed for reduction in 
    #datasize, but may limit SQL especially with the removal of the time.
    l2 = len(datfiles1)
    tcount=0
    t1 = numpy.zeros([len(x)*l2],dtype='single')
    x1 = numpy.zeros([len(x)*l2],dtype='single')
    y1 = numpy.zeros([len(x)*l2],dtype='single')
    l1 = len(x)
    for j in range(l2):
        tcount += 0.5
        t1[j*l1:(j+1)*l1]=tcount
        x1[j*l1:(j+1)*l1]=x
        y1[j*l1:(j+1)*l1]=y
    database = pd.DataFrame({'t':t1,
                             'x':x1,
                             'y':y1})
    del x1,t1,y1
    if l2<20000:
        rofint1.append(rofint[i])
    
    #Create vectors for storing the fractional occupancy variables.
    FrOcc = numpy.zeros([len(dat1.loc[:,'chi_n'])*len(datfiles1)],dtype='single')
    n = numpy.zeros([len(dat1.loc[:,'chi_n'])*len(datfiles1)],dtype='single')
    for j in range(l2):
        dat1 = pd.read_parquet(rofint[i]+r'/FracOcc/'+datfiles1[j])
        FrOcc[l1*j:l1*(j+1)]=numpy.array(dat1.loc[:,'chi_n'])
        n[l1*j:l1*(j+1)]=numpy.array(dat1.loc[:,'n'])
        
    data = database
    data['chi_n']=FrOcc
    del FrOcc
    data['n']=n
    del n
    
    data.to_parquet(rofint[i]+r'/FracOcc/fracocc_n')
    for i1 in range(len(datfiles[i])):
        os.remove(rofint[i]+r'/FracOcc/'+datfiles[i1])