# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 09:33:59 2019

@author: sivak
"""

import numpy as np
import matplotlib.pyplot as plt

def linreg(inputs,targets):

	inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
	beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs),inputs)),np.transpose(inputs)),targets)

	outputs = np.dot(inputs,beta)
	#print (np.shape(beta))
	plt.plot(inputs,outputs)
	return beta