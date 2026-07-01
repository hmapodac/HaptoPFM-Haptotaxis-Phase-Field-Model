# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:27:57 2023

@author: jmkoe
"""
from fipy import Grid2D
import numpy
import numpy.random as rand
import math
import os
import pandas
import scipy.integrate as integrate
import time

wid = 20
leng = 20
yi=leng*0.4
dx = 0.1
dy = dx
nx = int(wid/dx)
ny = nx
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
x,y = mesh.cellCenters

dxx = 0.05
dyy = dxx
nxx = int(wid/dxx)
nyy = nxx
LL = dxx * nxx
meshx = Grid2D(dx=dxx, dy=dyy, nx=nxx, ny=nyy)
xx,yy = mesh.cellCenters

""" The base is set to be csb=0.5*exp(0*(y-y0))"""
a1 = 0.5
X1 = 0.
def csbase(y,a1,X1,y0):
    return a1*math.exp(X1*(y-y0))

sz = len(x)
switchon = 0.05
time1 = 100
time1step = time1*100+1
t = numpy.linspace(0,time1,time1step)

os.mkdir('random')

for i1 in range(len(t)):
    randoms = rand.random(sz)
    savename1 = 'randomnumbers'+str(i1)
    pandas.DataFrame(randoms,columns=['rand']).to_parquet('random/'+savename1,index=False)
    
X = [0.01, 0.02, 0.05, 0.1]
a = [0.25,0.5,0.75]
def cs(y,a,X,y0):
    return a*math.exp(X*(y-y0))

os.mkdir('grad0.0')
os.mkdir('grad1.0')
os.mkdir('grad2.0')
os.mkdir('grad5.0')
os.mkdir('grad10.0')
os.mkdir('grad0.0/ons2.5')
os.mkdir('grad1.0/ons2.5')
os.mkdir('grad2.0/ons2.5')
os.mkdir('grad5.0/ons2.5')
os.mkdir('grad10.0/ons2.5')
os.mkdir('grad0.0/ons5.0')
os.mkdir('grad1.0/ons5.0')
os.mkdir('grad2.0/ons5.0')
os.mkdir('grad5.0/ons5.0')
os.mkdir('grad10.0/ons5.0')
os.mkdir('grad0.0/ons7.5')
os.mkdir('grad1.0/ons7.5')
os.mkdir('grad2.0/ons7.5')
os.mkdir('grad5.0/ons7.5')
os.mkdir('grad10.0/ons7.5')

for N in range(len(X)):
    for M in range(len(a)):
        
        intbase = integrate.quad(csbase,0,leng,args=(a1,X1,yi))[0]*wid
        Cinput = numpy.zeros([nx*ny])
        for i in range(len(y)):
             Cinput[i] = cs(y[i],a[M],X[N],yi)
        intfunc = sum(Cinput*mesh.cellVolumes)
    
        switchonnew=switchon*intfunc/intbase

        prop = numpy.zeros([sz,3])
        sz1 = len(prop)
        switchon1 = round(switchonnew*sz1)
        switchon2 = round((switchon1)/30)
        flip1 = numpy.zeros([switchon1,len(t)-1])
        ones = numpy.zeros([switchon1-switchon2,len(t)-1])
        zeros = numpy.zeros([switchon2,len(t)-1])
        for k1 in range(len(t)-1):
            flip = numpy.zeros([switchon1])
            for j in range(switchon1-switchon2):
                flip[j] = 1
            flip = numpy.random.permutation(flip)
            flip1[:,k1] = flip
            ones1 = numpy.where(flip1[:,k1] == 1)[0]
            ones[:,k1] = numpy.random.permutation(ones1)
            zeros1 = numpy.where(flip1[:,k1] == 0)[0]
            zeros[:,k1] = numpy.random.permutation(zeros1)
            
        strs = ["" for i in range(len(t))]
        for i in range(len(strs)):
            strs[i]= 't='+str(i)
        
        grad = "grad"+str(X[N]*100)+"/"
        percent = "ons"+str(10*a[M])+"/"
        pathname = grad+percent
        savename3 = 'flip'
        pandas.DataFrame(flip1,columns=strs[1:len(strs)]).to_parquet(pathname+savename3,index=False)
        
        savename1 = 'ones'
        pandas.DataFrame(ones,columns=strs[1:len(strs)]).to_parquet(pathname+savename1,index=False)
        
        savename0 = 'zeros'
        pandas.DataFrame(zeros,columns=strs[1:len(strs)]).to_parquet(pathname+savename0,index=False)
        