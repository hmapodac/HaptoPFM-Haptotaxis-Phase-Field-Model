# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:05:17 2023

@author: jmkoe

The code takes a unit circle as an initial condition for the phase field model
and simulates the phase field equations without signaling or protrusion input for
100 seconds.  This allows the transition region of the phase field to smooth prior
to utilizing it in the modeling of cell signaling and migration.
"""

#Load necessary packages.
from fipy import Grid2D, CellVariable
from fipy import DiffusionTerm as Diff
from fipy import TransientTerm as Trans
from fipy.tools import numerix
import numpy
import pandas as pd


#Initialize constants needed for the modeling
param = pd.read_csv('params1.csv')

#Collect the variables needed for the phase field equation and gather them.  If
#multiple parameter conditions are required the code will produce multiple output
#files.
initialit = pd.DataFrame.transpose(pd.DataFrame([param.loc[:,'x0'],\
            param.loc[:,'y0'],param.loc[:,'epsilon'],param.loc[:,'gammaphi'],\
            param.loc[:,'alpha'],param.loc[:,'mu'],param.loc[:,'sigma']]))
initialit = initialit.drop_duplicates()
initialit = numpy.array(initialit)
phii1 = []

#Loop over non-duplicate initial conditions.
for i in range(len(initialit)):
    [xi,yi,eps,gammap,alpha,muv,sigma]=initialit[i,:]
    
    
    #Define the meshing of the model.
    #Mesh for the small 8x8 inset box where the phase field equations are solved.
    #Wid and leng could also be parameterized but 8x8 box was found to be sufficient
    #for reduction of edge effects.
    wid = 8
    leng = 8
    dx = 0.05
    dy = dx
    nx = int(wid/dx)
    ny = int(nx)
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    x,y = mesh.cellCenters
    xi=wid/2
    yi=leng/2
    
    #Set up phase field equation with initial condition of 1 within centered 
    #unit circle. 
    delta = 1./2
    def potl(phi,delta,px,py):
        return phi*(1-phi)*(delta-sigma*(px**2+py**2)-phi)/eps
    px = CellVariable(name="protrusion x", mesh=mesh, hasOld=True)
    py = CellVariable(name="protrusion y", mesh=mesh, hasOld=True)
    p=[px,py]
    phii = CellVariable(name="phase field initial", mesh=mesh,value=0.,hasOld=True)
    segment = (x-xi)**2+(y-yi)**2<1
    phii.setValue(1,where=segment)
    eqphii = (Trans(var=phii) == gammap*(Diff(coeff=eps,var=phii)-potl(phii,delta,px,py))\
          -alpha*numerix.dot(p,phii.grad))
    
    #Solve phase field equation for 100s for smoothing transition region.
    timesdur=0.01
    steps = int(0.5/timesdur)
    intphi = numpy.double(numpy.zeros([steps]))
    for j in range(steps):
        phii.updateOld()
        eqphii.solve(phii,dt=timesdur)
    
        intphi1 = (phii*mesh.cellVolumes).sum().value
        intphi[j]=intphi1
        delta = 1/2+muv*(intphi1-intphi[0])
    
    #save the current iteration of the phase field.
    pd.DataFrame(phii,columns=['phii']).to_parquet('phii'+str(i))