# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:57:50 2023

@author: jmkoe

The code updates the adhesion network for the current random number set.  1/30th
of the previous adhesion network is turned over using flip and an equivalent number
of previously off positions are turned on based on the lowest gradient biased
random number site scores.
"""

#Load the necessary packages.
from fipy import Grid2D
import math
import numpy
import pandas as pd

def adchange(currC,tcurr,pathnamead,switchon,N_n1,N_noff,N_n_base,sec,\
             flip,ones,zeros,N_noff_old,N_n1old,test1,test1a,test2,boxmovex,\
             boxmovey,x0,y0):
    
    #Variable store from previous time step for the adhesion network.
    Cv = currC
    
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
    
    #Mesh for the 20x20 box where adhesion network is defined (bigbox, big voxel for n_store).
    widbig = 20
    lengbig = 20
    dxbig = 0.1
    dybig = dxbig
    nxbig = int(widbig/dxbig)
    nybig = int(lengbig/dybig)
    meshbig = Grid2D(dx=dxbig, dy=dybig, nx=nxbig, ny=nybig)
    xbig,ybig = meshbig.cellCenters

    #Finer mesh for the adhesion network (bigbox, small voxel for n_refined) for passing to inset.
    dxbig1 = 0.05
    dybig1 = dxbig1
    nxbig1 = int(widbig/dxbig1)
    nybig1 = nxbig1
    meshbig1 = Grid2D(dx=dxbig1, dy=dybig1, nx=nxbig1, ny=nybig1)
    xbig1,ybig1 = meshbig1.cellCenters
    
    #Read in the random number matrix to determine the ranking scores for all locations.
    number = pd.read_parquet(pathnamead+r'random/randomnumbers'+str(tcurr))
    
    #Define some necessary variables for later.
    k2 = 0
    switchto0 = []
    newone = []
    
    #Define number of occupied sites as fraction of total (N_n), number of 
    #occupied sites (on1new), and new number to switch on/off
    N_n = sum(Cv*meshbig.cellVolumes)
    on1new=int(numpy.round(len(meshbig.cellVolumes)*switchon*N_n/N_n_base))
    switchnew =int(numpy.round((on1new)/30))
    
    #Create vectors for ones, zeros, and flip from the current time point.
    #The setup here was done for indexing reasons from the ones, zeros and flip vectors.
    delete = numpy.ones(len(flip.loc[:,'t='+str(int(tcurr))]))
    flipcurr = numpy.array(flip.loc[:,'t='+str(int(tcurr))])
    onescurr = numpy.zeros([len(ones.loc[:,'t=1'])])
    for loop in range(len(ones.loc[:,'t='+str(int(tcurr))])):
        onescurr[loop]=ones.loc[loop,'t='+str(int(tcurr))]
    zeroscurr = numpy.zeros([len(zeros.loc[:,'t=1'])])
    for loop in range(len(zeros.loc[:,'t='+str(int(tcurr))])):
        zeroscurr[loop]=zeros.loc[loop,'t='+str(int(tcurr))]
        
    
    #Check if the old quantity of on adhesions is the same as what it was for
    #the previous time step.
    #If so, make sure that the number of on adhesions is the same as at the inital
    #time point.  If not, change a 1 in delete to 0 to add an extra zero to the 
    #flip vector according to the onescurr vector.
    if on1new==N_n1old:
        value = int(N_n1-on1new)
        if value>0:
            for i in range(value):
                delete[int(onescurr[i])]=0
    #If not, add the value of onescurr[0] to switchtozero to switch an additional
    #adhesion off so the number of adhesions reduce by one.  Also change a 1 in
    # delete to 0 to add an extra zero to the flip vector according to the 
    #onescurr vector. 
    elif on1new!=N_n1old:
        value = int(N_n1-on1new)
        switchto0 =onescurr[0]
        N_n1old = on1new
        value = int(N_n1-on1new)
        if value>1:
            for i in range(int(value-1)):
                delete[int(onescurr[i+1])]=0
    #If the value of N_n1/30 or N_noff has reduced by one, add an additional 1 
    #addition to the on set.
    if switchnew!=N_noff:
        value = N_noff-switchnew
        for i in range(int(value)):
            newone.append(int(zeroscurr[i]))
    
    #Resolve the changes from above.
    if switchto0:
        flipcurr[int(switchto0)] = 0
    if len(newone)>0:
        flipcurr[newone]=1
        if switchnew!=N_noff_old:
            test1[test1a]=0
    if 0 in delete:
        j = 0
        flipcurr1 = numpy.zeros(int(sum(delete)))
        for i in range(len(delete)):
            if delete[i]:
                flipcurr1[j]=flipcurr[i]
                j+=1
        flipcurr = flipcurr1
    n_store = numpy.zeros([len(meshbig.x)])
    sn = numpy.zeros([len(meshbig.x),3])
    n_refined = numpy.zeros([meshbig1.nx,meshbig1.ny])
    k2 = 0
    sn[:,0] = xbig
    sn[:,1] = ybig
    
    #Run a for loop over all x-y locations in the adhesion domain.  Utilize test
    #variables to check if locations were of the lowest ranking score (test1)
    #or were on previously (test2).  Those who have the lowest ranking score 
    #(if test1[j]) are turned on. Those that aren't in previously on or have lowest
    #ranking scores (elif (1-test1[j]-test2[j])) are rescored for the current
    #fractional occupancy. Lastly those that were previously on (if test2[j]) are
    #either kept on or turned off according to flipcurr; those that are turned off
    #are rescored otherwise score is kept artificially high so on adhesions aren't
    #programmed to turn on again.
    for j in range(len(meshbig.x)):
        if test1[j]:
            n_store[j] = 1
            sn[j, 2] = 10000
        elif (1-test1[j]-test2[j]):
            sn[j, 2] = - math.log10(number.loc[j, 'rand'])/Cv[j]
            
        if test2[j]:
            n_store[j] = flipcurr[k2]
            if flipcurr[k2] == 0:
                sn[j, 2] = -math.log10(number.loc[j, 'rand'])/Cv[j]
            else:
                sn[j,2]=10000
            k2 = k2+1
    
    #Utilize the n_store variable to define the adhesions in the phase field
    #subdomain and define n for the next timepoint.      
    for i2 in range(nxbig):
        for j2 in range(nybig):
            n_refined[2*j2,2*i2]=n_store[j2*nybig+i2]
            n_refined[2*j2+1,2*i2]=n_store[j2*nybig+i2]
            n_refined[2*j2,2*i2+1]=n_store[j2*nybig+i2]
            n_refined[2*j2+1,2*i2+1]=n_store[j2*nybig+i2]
    xtest1 = (x0-(wid/2)+boxmovex)
    xtest2=(x0+(wid/2)+boxmovex)
    if xtest1<0:
        roll1 = int(xtest1/0.05)
        xtest2 = xtest2-xtest1
        xtest1 = 0
        onsgrid1 = numpy.roll(n_refined,roll1,-1)
    elif xtest2>20:
        roll1 = -1*int((xtest2-20)/0.05)
        xtest1 = xtest1+(xtest2-20)
        xtest2 = 20
        onsgrid1 = numpy.roll(n_refined,roll1,-1)
    else:
        onsgrid1=n_refined
    xtest = (xbig1>xtest1)*(xbig1<xtest2)
    ytest = (ybig1>(y0-(leng/2)+boxmovey))*(ybig1<(y0+(leng/2)+boxmovey))
    adtest = xtest*ytest
    adtest = numpy.reshape(adtest,[nxbig1,nybig1])
    indad = numpy.where(adtest==1)
    n = numpy.zeros([nx*ny])
    for j1 in range(len(indad[0])):
        n[j1] = onsgrid1[indad[0][j1],indad[1][j1]]
    
    #Define the test variables for the next for loops.
    ind = numpy.argsort(sn[:,2])
    test1=numpy.zeros([len(ind)])
    test1[ind[0:switchnew]]=1
    test1a=ind[switchnew-1]
    ind = numpy.where(n_store==1)
    test2 = numpy.zeros([len(test1)])
    test2[ind[0]]=1
    
    return n,test1,test1a,test2,N_n1old,switchnew,n_refined,n_store  
