# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:45:18 2023

@author: jmkoe

This file shifts the subdomain where the phase field model is solved in the 
larger domain of the adhesion network.  The adhesion network is periodic in the
x-dimension and bounded in y. Therefore, the x-y bounds of the subdomain must 
be checked for the x periodic nature of the domain and the y edges. Additionally,
the edge of the subdomain is zeroed as the cell is recentered to accommodate 
the shift.
"""

#Load the necessary packages for the code.
from fipy import Grid2D
import numpy

def domainshift(xcheck1,xcheck,ycheck1,ycheck,currdata,boxmovex,boxmovey,x0,y0,onsgrid):
    
    #Define variables for the current phase field data to shift in the box.
    phiv = currdata[0]
    av = currdata[1]
    pxv = currdata[2]
    pyv = currdata[3]
    
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
    
    #Finer mesh for the adhesion network (bigbox, small voxel for n_refined) for passing to inset.
    widbig = 20
    lengbig = 20
    dxbig1 = 0.05
    dybig1 = dxbig1
    nxbig1 = int(widbig/dxbig1)
    nybig1 = nxbig1
    meshbig1 = Grid2D(dx=dxbig1, dy=dybig1, nx=nxbig1, ny=nybig1)
    xbig1,ybig1 = meshbig1.cellCenters
   
    #The series of if statements checks the displacement direction of the 
    #phase field cell centroid relative to the subdomain center.  Once the 
    #direction is determined, the phase field cell is recentered by using 
    #numpy.roll, shifting the phase field 0.5 units or 10 voxels in the direction
    #opposite the displacement.
    if ((xcheck1-xcheck)>0.5):
        
        #Reshape the phase field as a grid rather than the vectorized form of
        #the data.
        phigrid = numpy.reshape(phiv,[nx,ny])
        agrid = numpy.reshape(av,[nx,ny])
        pxgrid = numpy.reshape(pxv,[nx,ny])
        pygrid = numpy.reshape(pyv,[nx,ny])
        
        #The edges of the domain are zeroed out so data isn't moved from one 
        #side of the domain to the other.
        tryx = x.reshape([nx,ny])
        tryx = tryx>1
        tryy = y.reshape([nx,ny])
        tryy = (tryy>0.5)*(tryy<(leng-0.5))
        phigrid = phigrid*tryx*tryy
        agrid = agrid*tryx*tryy
        pxgrid = pxgrid*tryx*tryy
        pygrid = pygrid*tryx*tryy
        
        #The numpy roll step recenters the cell by shifting the phase field 
        #0.5 units or 10 voxels in the direction opposite the displacement. The
        #numpy.roll function cyclically shifts the 2D matrix along a prescribed
        #axis.
        phigrid = numpy.roll(phigrid,-10,1)
        agrid = numpy.roll(agrid,-10,1)
        pxgrid = numpy.roll(pxgrid,-10,1)
        pygrid = numpy.roll(pygrid,-10,1)
        
        #Revectorize the data.
        phinew = numpy.reshape(phigrid,[nx*ny])
        anew = numpy.reshape(agrid,[nx*ny])
        pxnew = numpy.reshape(pxgrid,[nx*ny])
        pynew = numpy.reshape(pygrid,[nx*ny])
        
        #Lastly, the boxmove parameter is adjusted and the adhesions are updated
        #based on the new box position.  The center of the box is (x0+boxmovex,y0+boxmovey)
        #making the bounds +-4 in every direction.  The x and direction bounds 
        #must be checked for being >20 or <0 for parabolic conditions and edges
        #of adhesions respectively. The last part updates the adhesions in the
        #subdomain.
        boxmovex+=0.5
        xtest1 = (x0-(wid/2)+boxmovex)
        xtest2=(x0+(wid/2)+boxmovex)
        if xtest1<0:
            roll1 = int(xtest1/0.05)
            xtest2 = xtest2-xtest1
            xtest1 = 0
            onsgrid1 = numpy.roll(onsgrid,roll1,-1)
        elif xtest2>20:
            roll1 = int((xtest2-20)/0.05)
            xtest1 = xtest1+(xtest2-20)
            xtest2 = 20
            onsgrid1 = numpy.roll(onsgrid,roll1,-1)
        else:
            onsgrid1=onsgrid
        xtest = (xbig1>xtest1)*(xbig1<xtest2)
        ytest = (ybig1>(y0-(leng/2)+boxmovey))*(ybig1<(y0+(leng/2)+boxmovey))
        adtest = xtest*ytest
        adtest = numpy.reshape(adtest,[nxbig1,nybig1])
        indad = numpy.where(adtest==1)
        ons = numpy.zeros([nx*ny])
        for j1 in range(len(indad[0])):
            ons[j1] = onsgrid1[indad[0][j1],indad[1][j1]]
    
    if ((xcheck1-xcheck)<-0.5):
        
        #Reshape the phase field as a grid rather than the vectorized form of
        #the data.
        phigrid = numpy.reshape(phiv,[nx,ny])
        agrid = numpy.reshape(av,[nx,ny])
        pxgrid = numpy.reshape(pxv,[nx,ny])
        pygrid = numpy.reshape(pyv,[nx,ny])
        
        #The edges of the domain are zeroed out so data isn't moved from one 
        #side of the domain to the other.
        tryx = x.reshape([nx,ny])
        tryx = tryx<(wid-1)
        tryy = y.reshape([nx,ny])
        tryy = (tryy>0.5)*(tryy<(leng-0.5))
        phigrid = phigrid*tryx*tryy
        agrid = agrid*tryx*tryy
        pxgrid = pxgrid*tryx*tryy
        pygrid = pygrid*tryx*tryy
        
        #The numpy roll step recenters the cell by shifting the phase field 
        #0.5 units or 10 voxels in the direction opposite the displacement. The
        #numpy.roll function cyclically shifts the 2D matrix along a prescribed
        #axis.
        phigrid = numpy.roll(phigrid,10,1)
        agrid = numpy.roll(agrid,10,1)
        pxgrid = numpy.roll(pxgrid,10,1)
        pygrid = numpy.roll(pygrid,10,1)
        
        #Revectorize the data.
        phinew = numpy.reshape(phigrid,[nx*ny])
        anew = numpy.reshape(agrid,[nx*ny])
        pxnew = numpy.reshape(pxgrid,[nx*ny])
        pynew = numpy.reshape(pygrid,[nx*ny])
        
        #Lastly, the boxmove parameter is adjusted and the adhesions are updated
        #based on the new box position.  The center of the box is (x0+boxmovex,y0+boxmovey)
        #making the bounds +-4 in every direction.  The x and direction bounds 
        #must be checked for being >20 or <0 for parabolic conditions and edges
        #of adhesions respectively. The last part updates the adhesions in the
        #subdomain.
        boxmovex+=-0.5
        xtest1 = (x0-(wid/2)+boxmovex)
        xtest2=(x0+(wid/2)+boxmovex)
        if xtest1<0:
            roll1 = int(xtest1/0.05)
            xtest2 = xtest2-xtest1
            xtest1 = 0
            onsgrid1 = numpy.roll(onsgrid,roll1,-1)
        elif xtest2>20:
            roll1 = int((xtest2-20)/0.05)
            xtest1 = xtest1+(xtest2-20)
            xtest2 = 20
            onsgrid1 = numpy.roll(onsgrid,roll1,-1)
        else:
            onsgrid1=onsgrid
        xtest = (xbig1>xtest1)*(xbig1<xtest2)
        ytest = (ybig1>(y0-(leng/2)+boxmovey))*(ybig1<(y0+(leng/2)+boxmovey))
        adtest = xtest*ytest
        adtest = numpy.reshape(adtest,[nxbig1,nybig1])
        indad = numpy.where(adtest==1)
        ons = numpy.zeros([nx*ny])
        for j1 in range(len(indad[0])):
            ons[j1] = onsgrid1[indad[0][j1],indad[1][j1]]
        
    if ((ycheck1-ycheck)>0.5):
        
        #Reshape the phase field as a grid rather than the vectorized form of
        #the data.
        phigrid = numpy.reshape(phiv,[nx,ny])
        agrid = numpy.reshape(av,[nx,ny])
        pxgrid = numpy.reshape(pxv,[nx,ny])
        pygrid = numpy.reshape(pyv,[nx,ny])
        
        #The edges of the domain are zeroed out so data isn't moved from one 
        #side of the domain to the other.
        tryx = x.reshape([nx,ny])
        tryx = (tryx>0.5)*(tryx<(wid-0.5))
        tryy = y.reshape([nx,ny])
        tryy = tryy>1
        phigrid = phigrid*tryx*tryy
        agrid = agrid*tryx*tryy
        pxgrid = pxgrid*tryx*tryy
        pygrid = pygrid*tryx*tryy
        
        #The numpy roll step recenters the cell by shifting the phase field 
        #0.5 units or 10 voxels in the direction opposite the displacement. The
        #numpy.roll function cyclically shifts the 2D matrix along a prescribed
        #axis.
        phigrid = numpy.roll(phigrid,-10,0)
        agrid = numpy.roll(agrid,-10,0)
        pxgrid = numpy.roll(pxgrid,-10,0)
        pygrid = numpy.roll(pygrid,-10,0)
        
        #Revectorize the data.
        phinew = numpy.reshape(phigrid,[nx*ny])
        anew = numpy.reshape(agrid,[nx*ny])
        pxnew = numpy.reshape(pxgrid,[nx*ny])
        pynew = numpy.reshape(pygrid,[nx*ny])
        
        #Lastly, the boxmove parameter is adjusted and the adhesions are updated
        #based on the new box position.  The center of the box is (x0+boxmovex,y0+boxmovey)
        #making the bounds +-4 in every direction.  The x and direction bounds 
        #must be checked for being >20 or <0 for parabolic conditions and edges
        #of adhesions respectively. The last part updates the adhesions in the
        #subdomain.
        boxmovey+=0.5
        xtest1 = (x0-(wid/2)+boxmovex)
        xtest2=(x0+(wid/2)+boxmovex)
        if xtest1<0:
            roll1 = int(xtest1/0.05)
            xtest2 = xtest2-xtest1
            xtest1 = 0
            onsgrid1 = numpy.roll(onsgrid,roll1,-1)
        elif xtest2>20:
            roll1 = int((xtest2-20)/0.05)
            xtest1 = xtest1+(xtest2-20)
            xtest2 = 20
            onsgrid1 = numpy.roll(onsgrid,roll1,-1)
        else:
            onsgrid1=onsgrid
        xtest = (xbig1>xtest1)*(xbig1<xtest2)
        ytest = (ybig1>(y0-(leng/2)+boxmovey))*(ybig1<(y0+(leng/2)+boxmovey))
        adtest = xtest*ytest
        adtest = numpy.reshape(adtest,[nxbig1,nybig1])
        indad = numpy.where(adtest==1)
        ons = numpy.zeros([nx*ny])
        for j1 in range(len(indad[0])):
            ons[j1] = onsgrid1[indad[0][j1],indad[1][j1]]
        
    if ((ycheck1-ycheck)<-0.5):
        
        #Reshape the phase field as a grid rather than the vectorized form of
        #the data.
        phigrid = numpy.reshape(phiv,[nx,ny])
        agrid = numpy.reshape(av,[nx,ny])
        pxgrid = numpy.reshape(pxv,[nx,ny])
        pygrid = numpy.reshape(pyv,[nx,ny])
        
        #The edges of the domain are zeroed out so data isn't moved from one 
        #side of the domain to the other.
        tryx = x.reshape([nx,ny])
        tryx = (tryx>0.5)*(tryx<(wid-0.5))
        tryy = y.reshape([nx,ny])
        tryy = tryy<(leng-1)
        phigrid = phigrid*tryx*tryy
        agrid = agrid*tryx*tryy
        pxgrid = pxgrid*tryx*tryy
        pygrid = pygrid*tryx*tryy
        
        #The numpy roll step recenters the cell by shifting the phase field 
        #0.5 units or 10 voxels in the direction opposite the displacement. The
        #numpy.roll function cyclically shifts the 2D matrix along a prescribed
        #axis.
        phigrid = numpy.roll(phigrid,10,0)
        agrid = numpy.roll(agrid,10,0)
        pxgrid = numpy.roll(pxgrid,10,0)
        pygrid = numpy.roll(pygrid,10,0)
    
        #Revectorize the data.
        phinew = numpy.reshape(phigrid,[nx*ny])
        anew = numpy.reshape(agrid,[nx*ny])
        pxnew = numpy.reshape(pxgrid,[nx*ny])
        pynew = numpy.reshape(pygrid,[nx*ny])
        
        #Lastly, the boxmove parameter is adjusted and the adhesions are updated
        #based on the new box position.  The center of the box is (x0+boxmovex,y0+boxmovey)
        #making the bounds +-4 in every direction.  The x and direction bounds 
        #must be checked for being >20 or <0 for parabolic conditions and edges
        #of adhesions respectively. The last part updates the adhesions in the
        #subdomain.
        boxmovey+=-0.5
        xtest1 = (x0-(wid/2)+boxmovex)
        xtest2=(x0+(wid/2)+boxmovex)
        if xtest1<0:
            roll1 = int(xtest1/0.05)
            xtest2 = xtest2-xtest1
            xtest1 = 0
            onsgrid1 = numpy.roll(onsgrid,roll1,-1)
        elif xtest2>20:
            roll1 = int((xtest2-20)/0.05)
            xtest1 = xtest1+(xtest2-20)
            xtest2 = 20
            onsgrid1 = numpy.roll(onsgrid,roll1,-1)
        else:
            onsgrid1=onsgrid
        xtest = (xbig1>xtest1)*(xbig1<xtest2)
        ytest = (ybig1>(y0-(leng/2)+boxmovey))*(ybig1<(y0+(leng/2)+boxmovey))
        adtest = xtest*ytest
        adtest = numpy.reshape(adtest,[nxbig1,nybig1])
        indad = numpy.where(adtest==1)
        ons = numpy.zeros([nx*ny])
        for j1 in range(len(indad[0])):
            ons[j1] = onsgrid1[indad[0][j1],indad[1][j1]]
    
    #Pass the vectorized data to currdata for subsequent simulation loops.
    phiv = list(phinew)
    av = list(anew)
    pxv = list(pxnew)
    pyv = list(pynew)
    currdata=[phiv,av,pxv,pyv]
    
    return currdata,ons,boxmovex,boxmovey
