# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:08:06 2023

@author: jmkoe

This code simulates a time step of the phase field subdomain. The code has a custom
timestepping option that allows for further timestep refinement if more than 10
loops are needed to find a solution.

The code sets up the problem by initializing on the previous timesteps solution
and running a fipy timestep with the Linear PCG solver and a tolerance of 1E-10
"""

#Import necessary libraries.
from fipy import DiffusionTerm as Diff
from fipy import CellVariable, Grid2D
from fipy import ImplicitSourceTerm as Src
from fipy import TransientTerm as Trans
from fipy.tools import numerix
import math
import numpy
import os
import pandas as pd

def runmodel(n,intphi0,param2,dt,solver,boxmovex,\
             boxmovey,i1,pathname,sec,currdata):
    
    #define the meshing of the model
    wid = 8
    leng = 8
    dx = 0.05
    dy = dx
    nx = int(wid/dx)
    ny = int(nx)
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    x,y = mesh.cellCenters
    
    #Define the parameters needed for the model.
    [alpha,betaa,betap,beta,Da,Dp,epsilon,gammaphi,k0,kaf,kar,kar2,kpf,\
	 kpr,kpr2,mu,sigma,x0,y0,n01,g1,krem,source]=param2
    
    #Define initialization variables for the cell variables used by the model.
    phiv = currdata[0]
    av = currdata[1]
    pxv = currdata[2]
    pyv = currdata[3]
    
    #Define model variables and their initial conditions
    phi = CellVariable(name="phase field", mesh=mesh,value=phiv,hasOld=True)
    n1 = CellVariable(name="adhesion",mesh=mesh,value=n, hasOld=True)
    a = CellVariable(name="signaling",mesh=mesh,value=av,hasOld=True)
    px = CellVariable(name="protrusion x", mesh=mesh,value=pxv,hasOld=True)
    py = CellVariable(name="protrusion y", mesh=mesh,value=pyv,hasOld=True)
    p = [px,py]
    intphi=numpy.single((phi*mesh.cellVolumes).sum().value)
    intphia=numpy.single((a*phi*mesh.cellVolumes).sum().value)
    b =  CellVariable(name="inactive a",mesh=mesh,value=numpy.single((numpy.pi-intphia)/intphi),hasOld=True)
    
    phi.updateOld()
    px.updateOld()
    py.updateOld()
    a.updateOld()
    n1.updateOld()
    b.updateOld()
    
    #Define some extra model parameters and model equations
    delta = numpy.single(1/2+mu*(intphi-intphi0))
    phigrad = numpy.single(numpy.sqrt(phi.grad[1,:]**2+phi.grad[0,:]**2))
    intn=numpy.single((phi*n1*mesh.cellVolumes).sum().value)
    def potl(phi,delta,px,py):
        return phi*(1-phi)*(delta-sigma*(px**2+py**2)-phi)/epsilon
    def Daphi(phi):
        return (Da*phi)
    def prev(phi):
        return (kpr+kpr2/(1+(100.*phi)**2))
    def arxn(phi,n1,b,a,phigrad):
        return ((k0*phi+((kaf*a*n1*phigrad)/(1+betaa*a)))*b-kar*a\
                -kar2*a/(1+(100.*phi)**2))
    def betax(phi,a,n1):
        return kpf*a*n1*numpy.single(phi.grad.value[0,:])/(1+betap*a)
    def betay(phi,a,n1):
        return kpf*a*n1*numpy.single(phi.grad.value[1,:])/(1+betap*a)
    def phiconv(phi,px,py):
        return px*numpy.single(phi.grad.value[0,:])\
               +py*numpy.single(phi.grad.value[1,:])
    def aconv(phi,px,py,a):
        return px*phi*numpy.single(a.grad.value[0,:])\
               +py*phi*numpy.single(a.grad.value[1,:])
     
    eqphi = (Trans(var=phi) == gammaphi*(Diff(coeff=epsilon,var=phi)-potl(phi,delta,px,py))\
              -alpha*(numerix.dot(p,phi.grad)/(1+beta*intn)))
    eqpx = (Trans(var=px)==Diff(coeff=Dp,var=px)-betax(phi,a,n1)\
            -Src(coeff=prev(phi),var=px))
    eqpy = (Trans(var=py)==Diff(coeff=Dp,var=py)-betay(phi,a,n1)\
            -Src(coeff=prev(phi),var=py))
    eqa = (Trans(var=a)==Diff(coeff=Daphi(phi),var=a) +arxn(phi,n1,b,a,phigrad)\
           -alpha*(Src(coeff=phiconv(phi,px,py),var=a)+aconv(phi,px,py,a))/(1+beta*intn))
    eq = eqphi & eqpx & eqpy & eqa
    
    #run the sweep for solving the model
    res = 1e+10
    sweep = 0
    j1 = 0
    sweep1c = 0
    sweep2c = 0
    while res > 1e-3 and sweep < 10:
        #actual sweep solve
        res = eq.sweep(dt=dt, solver=solver,cacheResidual=False,cacheError=False)
        
        #Define centroid positions of the phase field cell both in the subdomain
        #and the whole adhesion domain.
        xdisp1 = numpy.single((phi*x*mesh.cellVolumes).sum().value/intphi+(x0-(wid/2))+boxmovex)
        ydisp1 = numpy.single((phi*y*mesh.cellVolumes).sum().value/intphi+(y0-(leng/2))+boxmovey)
        xdispa = numpy.single((phi*x*mesh.cellVolumes).sum().value/intphi)
        ydispa = numpy.single((phi*y*mesh.cellVolumes).sum().value/intphi)
        
        #Cell volume, signaling volume, cell surface area, and adhesion volume.
        #The cell volume is used for the volume conservation term and the surface
        #area can also be something preserved with additional terms to the model. 
        intphi=numpy.single((phi*mesh.cellVolumes).sum().value)
        intphia=numpy.single((a*phi*mesh.cellVolumes).sum().value)
        inta=numpy.single((a*mesh.cellVolumes).sum().value)
        intphix = numpy.single((abs(phi.grad.value[0,:])*mesh.cellVolumes).sum())
        intphiy = numpy.single((abs(phi.grad.value[1,:])*mesh.cellVolumes).sum())
        intdphi = numpy.single(numpy.sqrt(intphix**2+intphiy**2))
        intn=numpy.single((phi*n1*mesh.cellVolumes).sum().value)
        
        #Redefine the inactive species based on new signaling volume
        b1=numpy.single((numpy.pi-intphia)/intphi)
        b.setValue(b1)
        
        #Redefine the volume conservation term.
        delta = numpy.single(1/2+mu*(intphi-intphi0))
        
        #Signaling variance, signaling/protrusion magnitude, signaling/protrusion
        #vector direction, signaling/protrusion volume 
        phithresh = numpy.greater(phi.value,0.1)
        athresh = phithresh*a.value
        m = sum(athresh)
        counter = sum(phithresh)
        m = m/counter
        numerator = sum((athresh-m)**2)
        variance = numpy.sqrt(numerator/(counter-1))
        amax = numpy.max(a.value)
        pmax=numpy.max(p)
        Sx=numpy.single(((a.value*(x-xdispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
        Sy=numpy.single(((a.value*(y-ydispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
        Smag=numpy.single(numpy.sqrt(Sx**2+Sy**2))
        Sdir=math.atan(Sy/Sx)
        Svol = numpy.single((a.value).sum())
        P = numpy.sqrt(px.value**2+py.value**2)
        Px=numpy.single(((P*(x-xdispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
        Py=numpy.single(((P*(y-ydispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
        Pmag=numpy.single(numpy.sqrt(Px**2+Py**2))
        Pdir=math.atan(Py/Px)
        Pvol = numpy.single(numpy.sqrt(px.value**2+py.value**2).sum())# Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
        sweep += 1
    
    #This if statement does the same solve as above for a reduced timestep when
    #the solver takes >10 iterations to solve.
    if sweep>9:
        for j1 in range(2):
            res = 1e+10
            sweep1 = 0
            while res > 1e-3 and sweep1 < 10:
                res = eq.sweep(dt=dt/2, solver=solver)
                
                xdisp1 = numpy.single((phi*x*mesh.cellVolumes).sum().value/intphi+(x0-(wid/2))+boxmovex)
                ydisp1 = numpy.single((phi*y*mesh.cellVolumes).sum().value/intphi+(y0-(leng/2))+boxmovey)
                xdispa = numpy.single((phi*x*mesh.cellVolumes).sum().value/intphi)
                ydispa = numpy.single((phi*y*mesh.cellVolumes).sum().value/intphi)
                
                intphi=numpy.single((phi*mesh.cellVolumes).sum().value)
                intphia=numpy.single((a*phi*mesh.cellVolumes).sum().value)
                inta=numpy.single((a*mesh.cellVolumes).sum().value)
                intphix = numpy.single((abs(phi.grad.value[0,:])*mesh.cellVolumes).sum())
                intphiy = numpy.single((abs(phi.grad.value[1,:])*mesh.cellVolumes).sum())
                intdphi = numpy.single(numpy.sqrt(intphix**2+intphiy**2))
                intn=numpy.single((phi*n1*mesh.cellVolumes).sum().value)
                b1=numpy.single((numpy.pi-intphia)/intphi)
                b.setValue(b1)
                delta = numpy.single(1/2+mu*(intphi-intphi0))
                phithresh = numpy.greater(phi.value,0.1)
                athresh = phithresh*a.value
                m = sum(athresh)
                counter = sum(phithresh)
                m = m/counter
                numerator = sum((athresh-m)**2)
                variance = numpy.sqrt(numerator/(counter-1))
                amax = numpy.max(a.value)
                pmax=numpy.max(p)
                Sx=numpy.single(((a.value*(x-xdispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
                Sy=numpy.single(((a.value*(y-ydispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
                Smag=numpy.single(numpy.sqrt(Sx**2+Sy**2))
                Sdir=math.atan(Sy/Sx)
                Svol = numpy.single((a.value).sum())
                P = numpy.sqrt(px.value**2+py.value**2)
                Px=numpy.single(((P*(x-xdispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
                Py=numpy.single(((P*(y-ydispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
                Pmag=numpy.single(numpy.sqrt(Px**2+Py**2))
                Pdir=math.atan(Py/Px)
                Pvol = numpy.single(numpy.sqrt(px.value**2+py.value**2).sum())# Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
            
                sweep1 += 1
            if sweep1 == 10:
                phi.setValue(phi.old)
                px.setValue(px.old)
                py.setValue(py.old)
                a.setValue(a.old)
                b.setValue(b.old)
                p = [px,py]
                intphi=numpy.single((phi*mesh.cellVolumes).sum().value)
                intphia=numpy.single((a*phi*mesh.cellVolumes).sum().value)
                inta=numpy.single((a*mesh.cellVolumes).sum().value)
                intphix = numpy.single((abs(phi.grad.value[0,:])*mesh.cellVolumes).sum())
                intphiy = numpy.single((abs(phi.grad.value[1,:])*mesh.cellVolumes).sum())
                intdphi = numpy.single(numpy.sqrt(intphix**2+intphiy**2))
                intn=numpy.single((phi*n1*mesh.cellVolumes).sum().value)
                b1=numpy.single((numpy.pi-intphia)/intphi)
                b.setValue(b1)
                delta = numpy.single(1/2+mu*(intphi-intphi0))
                xdisp1 = numpy.single((phi*x*mesh.cellVolumes).sum().value/intphi+(x0-(wid/2))+boxmovex)
                ydisp1 = numpy.single((phi*y*mesh.cellVolumes).sum().value/intphi+(y0-(leng/2))+boxmovey)
                for j1 in range(2):
                    sweep2 = 0
                    res = 1e10
                    while res > 1e-3 and sweep2 < 10:
                        res = eq.sweep(dt=dt/4, solver=solver)
                        
                        xdisp1 = numpy.single((phi*x*mesh.cellVolumes).sum().value/intphi+(x0-(wid/2))+boxmovex)
                        ydisp1 = numpy.single((phi*y*mesh.cellVolumes).sum().value/intphi+(y0-(leng/2))+boxmovey)
                        xdispa = numpy.single((phi*x*mesh.cellVolumes).sum().value/intphi)
                        ydispa = numpy.single((phi*y*mesh.cellVolumes).sum().value/intphi)
                        
                        intphi=numpy.single((phi*mesh.cellVolumes).sum().value)
                        intphia=numpy.single((a*phi*mesh.cellVolumes).sum().value)
                        inta=numpy.single((a*mesh.cellVolumes).sum().value)
                        intphix = numpy.single((abs(phi.grad.value[0,:])*mesh.cellVolumes).sum())
                        intphiy = numpy.single((abs(phi.grad.value[1,:])*mesh.cellVolumes).sum())
                        intdphi = numpy.single(numpy.sqrt(intphix**2+intphiy**2))
                        intn=numpy.single((phi*n1*mesh.cellVolumes).sum().value)
                        b1=numpy.single((numpy.pi-intphia)/intphi)
                        b.setValue(b1)
                        delta = numpy.single(1/2+mu*(intphi-intphi0))
                        phithresh = numpy.greater(phi.value,0.1)
                        athresh = phithresh*a.value
                        m = sum(athresh)
                        counter = sum(phithresh)
                        m = m/counter
                        numerator = sum((athresh-m)**2)
                        variance = numpy.sqrt(numerator/(counter-1))
                        amax = numpy.max(a.value)
                        pmax=numpy.max(p)
                        Sx=numpy.single(((a.value*(x-xdispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
                        Sy=numpy.single(((a.value*(y-ydispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
                        Smag=numpy.single(numpy.sqrt(Sx**2+Sy**2))
                        Sdir=math.atan(Sy/Sx)
                        Svol = numpy.single((a.value).sum())
                        P = numpy.sqrt(px.value**2+py.value**2)
                        Px=numpy.single(((P*(x-xdispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
                        Py=numpy.single(((P*(y-ydispa))/numpy.sqrt((x-xdispa)**2+(y-ydispa)**2)).sum())
                        Pmag=numpy.single(numpy.sqrt(Px**2+Py**2))
                        Pdir=math.atan(Py/Px)
                        Pvol = numpy.single(numpy.sqrt(px.value**2+py.value**2).sum())# Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
            
                        sweep2 += 1
                    phi.updateOld()
                    px.updateOld()
                    py.updateOld()
                    a.updateOld()
                    n1.updateOld()
                    b.updateOld()
                    p=[px,py]
                    sweep2c += sweep2
            phi.updateOld()
            px.updateOld()
            py.updateOld()
            a.updateOld()
            n1.updateOld()
            b.updateOld()
            p=[px,py]
            sweep1c += sweep1
        
        #Define the variable values to pass back to the phasemodel file for further
        #timestepping
        phix = phi.grad.value[0,:]
        phiy = phi.grad.value[1,:]
        phi = phi.value
        px = px.value
        py = py.value
        a = a.value        
    else:
        #Define the variable values to pass back to the phasemodel file for further
        #timestepping
        phix = phi.grad.value[0,:]
        phiy = phi.grad.value[1,:]
        phi = phi.value
        px = px.value
        py = py.value
        a = a.value
    
    #Make dataframes for the variable values and analysis of the phase field
    #cell.  These are then exported to files for further analysis of the models.
    data=pd.DataFrame({'phi':phi,
                       'phix':phix,
                       'phiy':phiy,
                       'px':px,
                       'py':py,
                       'a':a,
                       'n':n})
    aux=pd.DataFrame({'b':[b1],
                       'xdisp':[xdisp1],
                       'ydisp':[ydisp1],
                       'intphi':[intphi],
                       'inta':[inta],
                       'intphia':[intphia],
                       'intdphi':[intdphi],
                       'delta':[delta],
                       'intn':[intn],
                       'variance':[variance],
                       'amax':[amax],
                       'pmax':[pmax],
                       'Sx':[Sx],
                       'Sy':[Sy],
                       '|S|':[Smag],
                       'Sdir':[Sdir],
                       'Svol':[Svol],
                       'Px':[Px],
                       'Py':[Py],
                       '|P|':[Pmag],
                       'Pdir':[Pdir],
                       'Pvol':[Pvol],
                       'movex':[boxmovex],
                       'movey':[boxmovey],
                       'sweep':[sweep],
                       'sweep1':[sweep1c],
                       'sweep2':[sweep2c]})
    #savedata = "data"+str((i1+1)/sec)
    savedata = "data"+str(int(i1+1)) #Adjustment to 1 sec timestep over 0.5 sec.
    data.to_parquet(pathname+'data/'+savedata,index=False)
    filepath = pathname+'aux'
    if not os.path.isfile(filepath):
        aux.to_parquet(filepath, engine='fastparquet',index=False)
    else:
        aux.to_parquet(filepath, engine='fastparquet',index=False,\
                       append=True)
    
    #Pass the data from the current timestep back to the phasemodel file.
    del phiv, av, pxv, pyv
    phiv = list(phi)
    av = list(a)
    pxv = list(px)
    pyv = list(py)
    currdata=[phiv,av,pxv,pyv]
    
    del wid, leng, dx,dy, nx, ny, mesh, x, y, phi, a, px, py,\
        inta, delta, phigrad, intn, potl, Daphi, prev, arxn, betax,\
        betay, phiconv, aconv, eqphi, eqpx, eqpy, eqa,eq, res, sweep,\
        j1, sweep1c, sweep2c, intphia, intphix, intphiy, intdphi,\
        b1, b, data, aux    
    
    return currdata,intphi
