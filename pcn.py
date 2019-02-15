# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:16:41 2019

@author: sivak
"""

import numpy as np
import matplotlib.pyplot as plt
class pcn:
    """ A basic Perceptron (the same pcn.py except with the weights printed
    and it does not reorder the inputs)"""
    
    def __init__(self,inputs,targets):
        """ Constructor """
        # Set up network size
        if np.ndim(inputs)>1:
            self.nIn = np.shape(inputs)[1]
        else: 
            self.nIn = 1
    
        if np.ndim(targets)>1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = np.shape(inputs)[0]
    
        # Initialise network
        self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

    def pcntrain(self,inputs,targets,eta,nIterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
    
        # Training
        change = range(self.nData)

        for n in range(nIterations):
            
            self.activations = self.pcnfwd(inputs);
            self.weights -= eta*np.dot(np.transpose(inputs),self.activations-targets)
            print ("Iteration: ", n)
            print (self.weights)
            
            activations = self.pcnfwd(inputs)
            #print ("Final outputs are:")
            #print (activations)
        #return self.weights
        #plt.plot(inputs[:],activations[:],"b--")
        #plt.show()
    def pcnfwd(self,inputs):
        """ Run the network forward """

        # Compute activations
        activations =  np.dot(inputs,self.weights)
        
        # Threshold the activations
        return np.where(activations>0,1,0)

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
        outputs = np.dot(inputs,self.weights)
    
        nClasses = np.shape(targets)[1]

        if nClasses==1:
            nClasses = 2
            outputs = np.where(outputs>0,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nClasses,nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))
        print (cm)
        
        print (np.trace(cm)/np.sum(cm))
    def pcnplot(self,inputs,size):
        """PLotting"""
        for i in range (0,size-2):
            if (self.weights[i]==0):
                slope=0;intercept=0;
            else:
                slope = -(self.weights[0]/self.weights[i+2])/(self.weights[i]/self.weights[i+1])
                intercept=-self.weights[0]/self.weights[i+2]
            xa=np.array([]);yb=np.array([])
            for v in np.arange(0.0,19.0,0.1):
                y = (slope*i) + intercept
                j=1
                #x=2*100+1*10+j
                
                #plt.subplot(x);j=j+1;
                xa=np.append(xa,v);
                yb=np.append(yb,y);
                #plt.plot(i, y,"k.");
            plt.plot(xa[:],yb[:],"--");
         

