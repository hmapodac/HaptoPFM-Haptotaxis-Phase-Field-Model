# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:38:51 2023

@author: jmkoe

File to generate large domain movies for the signaling/phase field cell migration
and the adhesion/fractional occupancy. Additionally, the phase field cell is 
analyzed for splitting and whether it touches the y-boundaries of the adhesion
domain.

Could be improved with SQL structuring to pull certain time point windows with 
duckdb for the movie generation.
"""

#Load the necessary packages for running the code.
import numpy
from matplotlib import pyplot as plt
import pandas as pd
import os
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from scipy import ndimage  


#Find the folders of interest where the data files are for video generation.
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

#Loop over all the model folders to generate whole box movies of the phase field
#cell and adhesion network.
for i in range(len(rofint)):
    
    #Load timepoint by timepoint cell analysis file (aux) and associated parameters.
    aux=pd.read_parquet(rofint[i]+r'/aux')
    aux.to_parquet(rofint[i]+r'/aux')
    params = numpy.array(pd.read_csv(rofint[i]+r'/parameters'))
    
    #Define the meshing of the model.
    #Mesh for the small 8x8 inset box where the phase field equations are solved.
    wid = 8
    leng = 8
    dx = 0.05
    dy = dx
    nx = int(wid/dx)
    ny = int(nx)
    #Mesh for the 20x20 box where adhesion network is defined (bigbox, big voxel for n_store).
    widbig = 20
    lengbig = 20
    dxbig = 0.1
    dybig = dxbig
    nxbig = int(widbig/dxbig)
    nybig = int(lengbig/dybig)
    #Finer mesh for the adhesion network (bigbox, small voxel for n_refined) for passing to inset.
    dxbig1 = 0.05
    dybig1 = dxbig1
    nxbig1 = int(widbig/dxbig1)
    nybig1 = nxbig1
    
    #Domain sizing
    sz = nx*ny
    szz = nxbig*nybig
    
    #Duration of the video. The frames are taken every 8 timepoints or 8 seconds.
    t = len(aux)/8
    fpsvid = 100
    durationvid = t/fpsvid
    
    #Read in the phase field data to run the video generation and analysis.
    dat = pd.read_parquet(rofint[i]+r"/data/data")

    #Function to read particular parts of the phase field variable
    def data(k):
        phi = numpy.array(dat.loc[sz*k:sz*(k+1)-1,'phi'])
        yield phi

    
    #Collect some necessary variables and predefine storage for future steps.
    datcollect5 = numpy.ones([len(aux),2])
    philog2 = numpy.zeros([nxbig1,nybig1])
    xstart = params[17,1]
    ystart = params[18,1]
    xpos = numpy.array(aux.loc[:,'xdisp'])
    ypos = numpy.array(aux.loc[:,'ydisp'])
    boxmovex=numpy.array(aux.loc[:,'movex'])
    boxmovey=numpy.array(aux.loc[:,'movey'])
    
    #The next for look checks for if the cell splits into two distinct shapes or  
    #whether the phase field touches the bottom or top of the 20x20 adhesion domain.
    for k in range(int(len(dat)/(nx*ny))):
        
        #Utilize thresholding to create a binary image that can then be used to 
        #label contiguous shapes.  The phase field model analysis is stopped if \
        #the cell divides into two shapes.
        for phi in data(k):
            philog1 = numpy.reshape((phi>1E-3),[nx,ny])
        blobs, number_of_blobs = ndimage.label(philog1)
        
        #Run a check to see if the phase field cell has touched the edge of the
        #adhesion domain. The edge of the phase field cell is defined as phi=1E-3.
        #philog2 is the repositioning of philog1 into the 20x20 domain.
        xmin = round((xstart+boxmovex[k]-4)/20*400)
        xmax = round((xstart+boxmovex[k]+4)/20*400)
        ymin = round((ystart+boxmovey[k]-4)/20*400)
        ymax = round((ystart+boxmovey[k]+4)/20*400)
        if xmin<0:
            roll1 = xmin
            xmax = 160
            xmin = 0
        elif xmax>400:
            roll1 = xmax-400
            xmin = 400-160
            xmax = 400 
        else:
            roll1 = 0
        if ymax>400:
            trim = ymax-400
            philog1 = philog1[0:160-trim,:]
            ymax =400
        elif ymin<0:
            trim = 0-ymin
            philog1 = philog1[trim:160,:]
            ymin = 0
        philog2[ymin:ymax,xmin:xmax] = philog1
        philog2 = numpy.roll(philog2,roll1)
        
        #Store the data about multiple phase cell shapes and the logical of if
        #the phase field cell touches the edge of the adhesion domain.
        if number_of_blobs==2:
            datcollect5[k,0]=0
        if numpy.any(philog2[399,:]==1) or numpy.any(philog2[0,:]==1):
            for j in range(int(len(datcollect5)-k)):
                datcollect5[k+j,1]=0
    
    #Output the above analysis to a file for storage purposes.
    datcollect51=pd.DataFrame(datcollect5,columns=['blobs','outskirt'])
    datcollect51.to_parquet(rofint[i]+r'/analysisstop')    
    
    #Function to gather the phase field and signaling variables for generating 
    #the cell migration and signaling movie.
    def data(k):
        phi = numpy.array(dat.loc[sz*k:sz*(k+1)-1,'phi'])
        a = numpy.array(dat.loc[sz*k:sz*(k+1)-1,'a'])
        yield phi,a
    
    # Preformat the figure and variables for the movie.
    fig = plt.figure(figsize=[5,5])
    a3 = numpy.zeros([nxbig1,nybig1])
    philog2 = numpy.zeros([nxbig1,nybig1])
    
    def make_frame_signaling(t):
        fig.clear()
        
        #Gather the phase field cell border and signaling variable on the gridded
        #mesh.
        for phi,a in data(8*int(t*fpsvid)):
            a2 = numpy.reshape(a,[nx,ny])
            philog1 = numpy.reshape(numpy.logical_and(phi<0.13,phi>0.07),[nx,ny])
        
        #Position the signaling variable in the larger adhesion domain.
        xmin = round((xstart+boxmovex[8*int(t*fpsvid)]-4)/20*400)
        xmax = round((xstart+boxmovex[8*int(t*fpsvid)]+4)/20*400)
        ymin = round((ystart+boxmovey[8*int(t*fpsvid)]-4)/20*400)
        ymax = round((ystart+boxmovey[8*int(t*fpsvid)]+4)/20*400)
        if xmin<0:
            roll1 = xmin
            xmax = 160
            xmin = 0
        elif xmax>400:
            roll1 = xmax-400
            xmin = 400-160
            xmax = 400 
        else:
            roll1 = 0
        if ymax>400:
            trim = ymax-400
            a2 = a2[0:160-trim,:]
            philog1 = philog1[0:160-trim,:]
            ymax =400
        elif ymin<0:
            trim = 0-ymin
            a2 = a2[trim:160,:]
            philog1 = philog1[trim:160,:]
            ymin = 0
        a3[ymin:ymax,xmin:xmax] = a2
        philog2[ymin:ymax,xmin:xmax] = philog1
        a1 = numpy.roll(a3,roll1)
        philog = numpy.roll(philog2,roll1)
        
        #Create the figure for the current frame of the movie.
        im1 = plt.imshow(a1,'nipy_spectral',interpolation=None, vmax=5,animated=True)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        im2 = plt.imshow(philog, 'binary_r', interpolation='none', alpha=0.2,animated=True)
        plt.plot(xpos[0:8*int(t*fpsvid)]/20*400,ypos[0:8*int(t*fpsvid)]/20*400,'r-')
        plt.xticks([0,100,200,300,400], labels=['0','5','10','15','20'], fontsize=14)
        plt.yticks([0,100,200,300,400], labels=['0','5','10','15','20'], fontsize=14)
        # returning numpy image
        return mplfig_to_npimage(fig)
    
    # creating animation
    animation = VideoClip(make_frame_signaling, duration = durationvid)
        
    # displaying animation with auto play and looping
    animation.write_videofile(rofint[i]+r'/signalingbigbox.mp4',\
              fps = fpsvid)
    
    ##Create movie for the adhesion network of the big domain with underlying gradient.
    
    #Load the fractional occupancy and adhesion data from the big domain.
    datchi = pd.read_parquet(rofint[i]+r'FracOcc/fracocc_n')
    
    #Function to pull the fractional occupancy and adhesions for a particular
    #timepoint.        
    def datachi(k):
        chi_n = numpy.array(datchi.loc[szz*k:szz*(k+1)-1,'chi_n'])
        n = numpy.array(datchi.loc[szz*k:szz*(k+1)-1,'n'])
        yield chi_n,n
        
    #Create a figure variable to create the frames of the movie.
    fig = plt.figure(figsize=[5,5])
    def make_frame_adhesions(t):
        fig.clear()
        
        #Create the 20x20 adhesion domain with fractional occupancy and adhesion
        #variables in the grid.
        for chi_n,n in datachi(8*int(t*fpsvid)):
            chi_n = numpy.reshape(chi_n,[nxbig,nybig])
            n = numpy.reshape(n,[nxbig,nybig])
        
        #Create the figure for the frame of the current figure.
        im1 = plt.imshow(chi_n,'binary_r',interpolation=None, vmin=0,vmax=2,animated=True)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        im2 = plt.imshow(n, 'hot', interpolation='none', alpha=0.2,animated=True)
        plt.xticks([0,50,100,150,200], labels=['0','5','10','15','20'], fontsize=14)
        plt.yticks([0,50,100,150,200], labels=['0','5','10','15','20'], fontsize=14)
        # returning numpy image
        return mplfig_to_npimage(fig)
        
    # creating animation
    animation = VideoClip(make_frame_adhesions, duration = durationvid)
            
    # displaying animation with auto play and looping
    animation.write_videofile(rofint[i]+r'/adhesionmapgrad.mp4',\
                  fps = fpsvid)

