# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:24:20 2019

@author: sivak
"""

import numpy as np
class lin:
    def linreg(inputs,targets):

        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs),inputs)),np.transpose(inputs)),targets)
        outputs = np.dot(inputs,beta)
        #int shape(beta)
        print (outputs)
        return beta

