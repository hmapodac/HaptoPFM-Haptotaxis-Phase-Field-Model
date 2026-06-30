# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:17:03 2023
@author: jmkoe

This code solves for the fractional occupancy over time in the models that 
include ECM removal.  The function is called in all model circumstances in order
to save the adhesion network and fractional occupancy for all x-y positions in 
the whole domain.  The saving serves as a check to ensure the adhesion algorythm
works properly.  The adhesions could be saved separately in a different file if
desired so this function doesn't have to be called for models without ECM removal.
"""

from fipy import CellVariable, Grid2D
from fipy import ImplicitSourceTerm as Src
from fipy import TransientTerm as Trans
import numpy
import pandas as pd

def ECMrem(oldchi,phiv,krem,tstep,dt,i1,x0,y0,pathname,n_store,boxmovex,boxmovey):
    chi_nv = oldchi
    
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

    #Define fractional occupancy variable from the previous time point to 
    #solve for current.
    chi_n = CellVariable(name="Fractional Occupancy", mesh=meshbig,value=chi_nv,hasOld=True)
    
    #ECM removal is started after the cell starts to spread and migrate so prior
    #to t=600s the ECM removal rate krem = 0. 
    k_rem = 0
    if (tstep[i1]+0.005)>3:
        k_rem = krem
    
    #Take the current position of the phase field cell and place it in the 20x20
    #box where the adhesions are defined. The resized phase field is stored in 
    #phi1. The chi_n defines the underlying gradient that determines the 
    #likelihood the location is turned on. 
    phi1 = numpy.zeros([meshbig.ny*meshbig.ny])
    xtest1 = (x0-(wid/2)+boxmovex)
    xtest2 = (x0+(wid/2)+boxmovex)
    if xtest1<0:
        roll1 = -1*int(xtest1/0.1)
        xtest2 = wid
        xtest1 = 0
    elif xtest2>20:
        roll1 = int((xtest2-20)/0.1)
        xtest1 = 20-wid
        xtest2 = 20
    else:
        roll1 = 0
    xptest = (xbig>xtest1)*(xbig<xtest2)
    yptest = (ybig>(y0-(leng/2)+boxmovey))*(ybig<(y0+(leng/2)+boxmovey))
    phitest = xptest*yptest
    indphi = numpy.where(phitest==1)
    phiresize = numpy.reshape(phiv,[mesh.nx,mesh.ny])
    xpos=0
    ypos=0
    for loop in range(len(indphi[0])):
            val = phiresize[2*xpos,2*ypos]
            val += phiresize[2*xpos,2*ypos+1]
            val += phiresize[2*xpos+1,2*ypos]
            val += phiresize[2*xpos+1,2*ypos+1]
            val = val/4
            phi1[indphi[0][loop]]=val
            xpos += 1
            if xpos == wid/dxbig:
                ypos +=1
                xpos = 0
    phi1 = numpy.reshape(phi1,[nxbig,nybig])
    phi1 = numpy.roll(phi1,roll1)
    phi1 = numpy.reshape(phi1,nxbig*nybig)
    
    #Solve for the fractional occupancy and store the current timepoint data.
    #Return the fractional occupancy matrix for future timepoints.
    eqC = (Trans(var=chi_n)==-Src(coeff=(k_rem),var=chi_n*phi1))
    eqC.solve(chi_n,dt=dt)
    dataframe = numpy.array([numpy.array(chi_n.value),n_store]).transpose()
    krem = pd.DataFrame(dataframe,columns = ['chi_n','n']).\
                to_parquet(pathname+'FracOcc/fracocc_n'+str(int(i1+1)))
    currC = list(chi_n.value)    
    return currC
