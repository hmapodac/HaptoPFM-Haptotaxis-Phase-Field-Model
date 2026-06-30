# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:55:41 2023

@author: jmkoe

The code below is run by the FipyRun2.py code.  It initializes the adhesion 
network and phase field variables and then sets up a for loop to run the phase
field model.

There are 4 codes called in the for loop structure.  First, the phase field is
solved for one timestep with the current adhesion network.  Next the fractional
occupancy is updated for the models where the ECM removal is turned on.  Third,
the adhesion network is updated every other timestep (2s), and finally, the phase
field cell is recentered in the subdomain when the cell centroid has deviated 
>0.5 units in x or y.
"""
#Import necessary function code files.
import Adhesions
import FracOcc
import Displacement
import FipyModelTimestep

#Import necessary python libraries
from fipy import CellVariable, Grid2D
from fipy import LinearPCGSolver as PCG
import gc
import os
import math
from matplotlib import pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy
import pandas as pd
import time

def phasemodel(param1,phii,numparam):
    #define parameters used in the model based on passed parameter file.
    [alpha,betaa,betap,beta,Da,Dp,epsilon,gammaphi,k0,kaf,kar,kar2,kpf,\
	 kpr,kpr2,mu,sigma,x0,y0,n01,g1,krem,source]=param1.loc[numparam,:]
    
    param2=param1.loc[numparam,:]
    
    #Define a location to store the model data.
    pathnamead = r'adhesionsbig'+str(int(source))+'/'
    pathnamead2 ='grad'+str(float(g1))+'/ons' +str(float(n01))+'/'
    os.mkdir(str(numparam)+'/')
    os.mkdir(str(numparam)+'/data')
    pathname = str(numparam)+'/'
    os.mkdir(pathname+'FracOcc')

    #Save the defined parameter set to the location store.
    param2.to_csv(pathname+'parameters')
   
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

    #Some parameters for initial definition of the adhesion network.  
    sz = len(xbig)
    switchon = 0.05
    time1 = 100
    
    # Define both the "base" or reference gradient and the model gradient from parameters.
    """ The base gradient is set to be a uniform 5%: csb=0.05*exp(0*(y-y0))"""
    n0base = 0.05
    gbase = 0.
    def chi_n_base(ybig,n0base,gbase,y0):
        return n0base*math.exp(gbase*(ybig-y0))
    def chi_n(ybig,n0,g,y0):
        return n0*math.exp(g*(ybig-y0))
    n0= n01/100
    g= g1/100
    
    #Set the base number of adhesions (switchonnew) to turn on in this model.
    #In the publication, this is N_n expressed as a percentage with N_n_base reference
    #percentage being 5%.
    N_n_base = 0
    N_n=0
    for loop in range(len(ybig)):
        N_n_base += chi_n_base(ybig[loop],n0base,gbase,y0)*meshbig.cellVolumes[loop]
        N_n += chi_n(ybig[loop],n0,g,y0)*meshbig.cellVolumes[loop]
    switchonnew=switchon*N_n/N_n_base
    time1step = int(time1*100+1)
    t = numpy.linspace(0,time1,time1step)

    #Import adhesion network variables that contain the random numbers and 
    #rules for the ECM removal models.
    flip = pd.read_parquet(pathnamead+pathnamead2+'flip')
    ones = pd.read_parquet(pathnamead+pathnamead2+'ones')
    zeros = pd.read_parquet(pathnamead+pathnamead2+'zeros')
    number = pd.read_parquet(pathnamead+r'random/randomnumbers0')
    
    #Create scoring matrix to generate a gradient preferential heirarchy for 
    #each adhesion site.
    sn = numpy.zeros([sz,3])
    k=0 
    for i in range(sz):
        sn[k,0]= xbig[i]
        sn[k,1]= ybig[i]
        sn[k,2]= -math.log10(number.loc[k,'rand'])/chi_n(ybig[i],n0,g,y0)
        k = k+1
    ind = numpy.argsort(sn[:,2])
    sz1 = len(sn)
    N_n1 = round(switchonnew*sz1)
    test0 = numpy.zeros([len(ind)])
    test0[ind[0:N_n1]]=1
    
    #Create a vector to define the adhesion locations
    n_store = [0 for i1 in range(len(sn))]
    for k in range(nxbig*nybig):
        if test0[k]:
            n_store[k]=1
            sn[k,2]=10000
    
    #Create the new gradient preferential heirarchy for the next time point and
    #assign the test values for what adhesions on adhesions should turn off and 
    #what off adhesions should turn on. We will call ("Delta"t)/tau*N_n from paper
    #N_noff as its the number at every adhesion redefining time to switch off.
    ind = numpy.argsort(sn[:,2])
    N_noff = round((N_n1)/30)
    test1=numpy.zeros([len(ind)])
    test1[ind[0:N_noff]]=1
    test1a=int(ind[N_noff])
    test2 = numpy.zeros([len(ind)])
    test2[ind[len(ind)-N_n1:len(ind)]]=1
    k=0
    
    #Specify adhesion locations according to the more refined mesh. Big box adhesion
    #network is on a 20x20 box with 0.1x0.1 voxel sizing and this needs to be refined
    #to 0.05x0.05 voxel sizing for the phase field equations.
    n_refined = numpy.zeros([nxbig1,nybig1])
    for i2 in range(nxbig):
        for j2 in range(nybig):
            n_refined[2*j2,2*i2]=n_store[j2*nybig+i2]
            n_refined[2*j2+1,2*i2]=n_store[j2*nybig+i2]
            n_refined[2*j2,2*i2+1]=n_store[j2*nybig+i2]
            n_refined[2*j2+1,2*i2+1]=n_store[j2*nybig+i2]
    
    #Pull out the subset of n_refined to pass to the phase field model as the 
    #defined variable n which shows on adhesion locations in the 8x8 subset box.
    xtest = (xbig1>(x0-(wid/2)))*(xbig1<(x0+(wid/2)))
    ytest = (ybig1>(y0-(leng/2)))*(ybig1<(y0+(leng/2)))
    adtest = xtest*ytest
    adtest = numpy.reshape(adtest,[nxbig1,nybig1])
    indad = numpy.where(adtest==1)
    n = numpy.zeros([nx*ny])
    for j1 in range(len(indad[0])):
        n[j1] = n_refined[indad[0][j1],indad[1][j1]]
    
    
    # Define the Concentration variable prior to any ECM removal.
    chi_n_input = numpy.zeros([nxbig*nybig])
    for i in range(len(ybig)):
        chi_n_input[i] = chi_n(ybig[i],n0,g,y0)
    
    
    #Necessary model definitions
    dt=5E-3
    t1 = 100
    steps = t1/dt
    change = 2
    tstep = numpy.linspace(0,time1,int(time1/dt)+1)
    solver = PCG(tolerance=1e-10)
    tic = time.time()
    tcurr = 0
    
    #Setting up phi integral for the b equation, previous timepoint N_n values,
    #and checks for the box movement
    N_n1old = N_n1
    N_noff_old = N_noff
    intphi0=sum(phii*mesh.cellVolumes)
    boxmovex = 0
    boxmovey = 0
    xcheck = sum(phii*x*mesh.cellVolumes)/intphi0
    ycheck = sum(phii*y*mesh.cellVolumes)/intphi0
    
    #Initial values for the variables and setting up variable definitions.
    i1 = 0
    phiv = phii
    b=1.0
    av = k0*phii/(kar*b)
    pxv = 0.0
    pyv = 0.0
    Cv = chi_n_input
    aux=[]
    phi = CellVariable(name="phase field", mesh=mesh,value=phiv,hasOld=True)
    a = CellVariable(name="signaling",mesh=mesh,value=av,hasOld=True)
    px = CellVariable(name="protrusion x", mesh=mesh,value=pxv,hasOld=True)
    py = CellVariable(name="protrusion y", mesh=mesh,value=pyv,hasOld=True)
    phiv = list(phi.value)
    av = list(a.value)
    pxv = list(px.value)
    pyv = list(py.value)
    currdata=[phiv,av,pxv,pyv]
    currchi = list(Cv)
    
    #Deleting unnecessary information to reduce memory load.
    del x,y,xbig1,ybig1,xbig,ybig,mesh,meshbig,meshbig1,dx,dy,nx,ny,dxbig1,dybig1,nxbig1,\
        nybig1,dxbig,dybig,nybig,nxbig,phiv,b,av,pxv,pyv,Cv,phi,a,px,py
   
    #This is the main 20,000 step for loop to solve the model which passes old 
    #information of previous time point and returns new. Phase field equations
    #are solved on subdomain, the gradient is updated based on ECM removal,
    #the adhesion network is updated on every other time point, and adjusting
    #the subdomain when the phase field cell moves significantly from center.
    for i1 in range(int(steps)):
        
        #Update the old data store from the previous timepoint so we can pass
        #the new data back to currdata and keep the old data throughout the loop.
        olddata = currdata
        oldchi = currchi
        del currdata,currchi
        
        #Solve the phase field equations in the subdomain for the next time step.
        currdata,intphi=FipyModelTimestep.runmodel(n,intphi0,param2,dt,\
                    solver,boxmovex,boxmovey,i1,pathname,change,olddata)
                
        #Solve the underlying gradient for the next time step for denuding cases.
        #This could be turned off for nondenuding cases but gradient and adhesion
        #network data wont be saved for the 20x20 domain without editing other
        #parts of the code.
        currchi=FracOcc.ECMrem(oldchi,currdata[0],krem,tstep,dt,i1,x0,y0,pathname,\
                              n_store,boxmovex,boxmovey)
        
        #For every other time step (every 2 seconds based on definition in paper),
        #the adhesion network is updated by turning off 1/30th of the on adhesions
        #and turning on an equivalent number of off adhesions based on the gradient
        #biased random number set.
        if ((i1+1)/change).is_integer():
            tcurr+=1
            #Redefine adhesion matrix
            [n,test1,test1a,test2,N_n1old,N_noff_old,n_refined,n_store] = \
                                    Adhesions.adchange(currchi,tcurr,pathnamead,\
                                    switchon,N_n1,N_noff,N_n_base,change,flip,ones,\
                                    zeros,N_noff_old,N_n1old,test1,test1a,test2,\
                                    boxmovex,boxmovey,x0,y0)
        
        #Run a check on the centroid position of the cell, and if it has deviated
        #from the center of the subdomain by more than 0.5 units in any direction,
        #recenter the cell by moving the subdomain 0.5 units in the same direction
        #and reading in the new adhesion information.
        wid = 8
        leng = 8
        dx = 0.05
        dy = dx
        nx = int(wid/dx)
        ny = int(nx)
        mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
        x,y = mesh.cellCenters
        xcheck1 = sum(currdata[0]*x*mesh.cellVolumes)/intphi
        ycheck1 =sum(currdata[0]*y*mesh.cellVolumes)/intphi
        if (abs(xcheck1-xcheck)>0.5) or (abs(ycheck1-ycheck)>0.5):
            [currdata,n,boxmovex,boxmovey]=Displacement.domainshift(xcheck1,\
                                             xcheck,ycheck1,ycheck,currdata,\
                                             boxmovex,boxmovey,x0,y0,n_refined)
            
        
        del mesh,dx,dy,wid,leng,nx,ny,x,y
        #gc.collect helps the code from accumulating data overtime by allowing the
        #garbage collector to dump stored data unnecessary for the looping.
        gc.collect(generation=0)
        
    toc = time.time()

    #define the meshing of the model
    wid = 8
    dx = 0.05
    dy = dx
    nx = int(wid/dx)
    ny = int(nx)
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    x,y = mesh.cellCenters

    #define mesh for the adhesion network (bigbox, big voxel for n_store)
    widbig = 20
    lengbig = 20
    dxbig = 0.1
    dybig = dxbig
    nxbig = int(widbig/dxbig)
    nybig = int(lengbig/dybig)
    meshbig = Grid2D(dx=dxbig, dy=dybig, nx=nxbig, ny=nybig)
    xbig,ybig = meshbig.cellCenters

    #define mesh for the adhesion network (bigbox, small voxel for n_refined)
    dxbig1 = 0.05
    dybig1 = dxbig1
    nxbig1 = int(widbig/dxbig1)
    nybig1 = nxbig1
    meshbig1 = Grid2D(dx=dxbig1, dy=dybig1, nx=nxbig1, ny=nybig1)
    xbig1,ybig1 = meshbig1.cellCenters

    datfiles = os.listdir(pathname+r'data/')
    basename = ''.join([i for i in datfiles[0] if not i.isdigit()])

    basename = basename[0:len(basename)-1]
    datnum = numpy.zeros([len(datfiles)])
    for i1 in range(len(datnum)):
        datcurr = float(datfiles[i1][len(basename):len(datfiles[i1])])
        datnum[i1]=datcurr
    order = numpy.argsort(datnum)
    datfiles1=[datfiles[order[i1]] for i1 in range(len(order))]
    aux= pd.read_parquet(pathname+r'aux')

    disp = [0]
    T = [0]
    Tcos=[0]
    for i1 in range(int(len(aux)-1)):
        disp.append(numpy.sqrt((aux.loc[(i1+1),'xdisp']-\
                aux.loc[i1,'xdisp'])**2+(aux.loc[(i1+1),'ydisp']-\
                aux.loc[i1,'ydisp'])**2))
        T.append(numpy.sqrt((aux.loc[(i1+1),'xdisp']-\
                aux.loc[0,'xdisp'])**2+(aux.loc[(i1+1),'ydisp']-\
                aux.loc[0,'ydisp'])**2))
        Tcos.append(aux.loc[i1+1,'ydisp']-\
                    aux.loc[0,'ydisp'])

    datframe = numpy.array([disp,T,Tcos]).transpose()
    pd.DataFrame(datframe,columns=['displacement',\
                'T','cosT']).to_parquet(pathname+r'aux1')

    FMI = [1]
    P = [1]
    FMI1 = [1]
    P1 = [1]
    Tt = [0]
    Tcost = [0]
    dispt = [0]
    dispt1=[0]

    for i1 in range(int(len(aux)/100-1)):
        Tcost.append(aux.loc[(i1+1)*100-1,'ydisp']-\
                    aux.loc[i1*100,'ydisp'])
        Tt.append(numpy.sqrt((aux.loc[100*(i1+1)-1,'xdisp']-\
                aux.loc[100*i1,'xdisp'])**2+(aux.loc[100*(i1+1)-1,'ydisp']-\
                aux.loc[100*i1,'ydisp'])**2))
        dispt.append(sum(disp[0:100*(i1+1)-1]))
        dispt1.append(sum(disp[100*i1:100*(i1+1)-1]))
        FMI.append(Tcos[100*(i1+1)]/dispt[i1+1])
        P.append(T[100*(i1+1)-1]/dispt[i1+1])
        FMI1.append(Tcost[i1+1]/dispt1[i1+1])
        P1.append(Tt[i1+1]/dispt1[i1+1])

    datframe1 = numpy.array([dispt,dispt1,Tt,Tcost,FMI,FMI1,\
                            P,P1]).transpose()

    pd.DataFrame(datframe1,columns\
                =['disp(t2=0-t2)','disp(t2=t1-t2)','T(t2=t1-t2)',\
                'cosT(t2=t1-t2)','FMI(t2=0-t2)','FMI(t2=t1-t2)',\
                'Persis(t2=0-t2)','Persis(t2=t1-t2)'])\
                .to_parquet(pathname+r'aux2')
    
    def data(k):
        dat = numpy.array(pd.read_parquet(pathname+r"/data/"+datfiles1[k]))
        yield dat

    t = int(len(datfiles)/2)

    # duration of the video
    fpsvid = 50
    durationvid = t/fpsvid
    nx = int(nx)
    ny = int(ny)

    # matplot subplot
    fig = plt.figure(figsize=[3.5,3.5])

    # method to get frames
    def make_frame_ad(t):
        fig.clear()
        
        
        for nnow in data(2*int(t*fpsvid)):
            nnow1 = numpy.reshape(nnow[:,6],[nx,ny])
        
        # plotting line
        plt.imshow(nnow1,cmap='binary',animated=True)
        plt.xticks([0,40,80,120,160],labels=['0','2','4','6','8'],fontsize='x-large')
        plt.yticks([0,40,80,120,160],labels=['0','2','4','6','8'],fontsize='x-large')
        cbar = plt.colorbar(ticks=[0,0.5,1],shrink=1)
        cbar.ax.tick_params(labelsize=15)


        # returning numpy image
        return mplfig_to_npimage(fig)

    # creating animation
    animation = VideoClip(make_frame_ad, duration = durationvid)

    # displaying animation with auto play and looping
    animation.write_videofile(pathname+r'/adhesions.mp4',\
                              fps = fpsvid)

    # duration of the video
    fpsvid = 50
    durationvid = t/fpsvid

    # matplot subplot
    fig = plt.figure(figsize=[3.5,3.5])

    # method to get frames
    def make_frame_signaling(t):
        fig.clear()

        for dat in data(2*int(t*fpsvid)):
            a1 = numpy.reshape(dat[:,5],[nx,ny])
            philog = numpy.reshape(numpy.logical_and(dat[:,0]<0.13,dat[:,0]>0.07),[nx,ny])
        im1 = plt.imshow(a1,'nipy_spectral',interpolation=None, vmax=5,animated=True)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        im2 = plt.imshow(philog, 'binary_r', interpolation='none', alpha=0.2,animated=True)
        plt.xticks([0,30,60,90,120],labels=['0','1.5','3','4.5','6'],fontsize='x-large')
        plt.yticks([0,30,60,90,120],labels=['0','1.5','3','4.5','6'],fontsize='x-large')


        # returning numpy image
        return mplfig_to_npimage(fig)

    # creating animation
    animation = VideoClip(make_frame_signaling, duration = durationvid)

    # displaying animation with auto play and looping
    animation.write_videofile(pathname+r'/signaling.mp4',\
                              fps = fpsvid)


    return tic-toc
