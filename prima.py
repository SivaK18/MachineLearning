# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:19:44 2019

@author: sivak
"""

import pylab as pl
import numpy as np
import pcn
import pandas as pd

dataf=pd.read_csv("pima.csv")
pima = dataf.values ;

# Plot the first and second values for the two classes
indices0 = np.where(pima[:,8]==0)
indices1 = np.where(pima[:,8]==1)

pl.ion()
pl.plot(pima[indices0,0],pima[indices0,1],'go')
pl.plot(pima[indices1,0],pima[indices1,1],'rx')
# Perceptron training on the original dataset

p = pcn.pcn(pima[:,:8],pima[:,8:9])
p.pcntrain(pima[:,:8],pima[:,8:9],0.25,100)
p.confmat(pima[:,:8],pima[:,8:9])

# Various preprocessing steps
pima[np.where(pima[:,0]>8),0] = 8

pima[np.where(pima[:,7]<=30),7] = 1
pima[np.where((pima[:,7]>30) & (pima[:,7]<=40)),7] = 2
pima[np.where((pima[:,7]>40) & (pima[:,7]<=50)),7] = 3
pima[np.where((pima[:,7]>50) & (pima[:,7]<=60)),7] = 4
pima[np.where(pima[:,7]>60),7] = 5

pima[:,:8] = pima[:,:8]-pima[:,:8].mean(axis=0)
pima[:,:8] = pima[:,:8]/pima[:,:8].var(axis=0)

#print pima.mean(axis=0)
#print pima.var(axis=0)
#print pima.max(axis=0)
#print pima.min(axis=0)

trainin = pima[::2,:8]
testin = pima[1::2,:8]
traintgt = pima[::2,8:9]
testtgt = pima[1::2,8:9]

# Perceptron training on the preprocessed dataset
print ("Output after preprocessing of data")
p1 = pcn.pcn(trainin,traintgt)
p1.pcntrain(trainin,traintgt,0.25,100)
#p1.confmat(testin,testtgt)
p1.pcnplot(trainin,8);



pl.show()
