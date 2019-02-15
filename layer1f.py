# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:19:53 2019

@author: sivak
"""

# Make a prediction with weights

import numpy as np
import matplotlib.pyplot as plt
i=1
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

def plot(dataset,weights):
    xa=np.array([]);yb=np.array([]);
    for i in np.linspace(np.amin(dataset[:-1]),np.amax(dataset[:-1])):
        if weights[0]!=0 and weights[1]!=0 and weights[2]!=0 :
            slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
            intercept = -weights[0]/weights[2]
        else :
            slope = 0;
            intercept=1;

        #y =mx+c, m is slope and c is intercept
        y = (slope*i) + intercept
        j=1
        x=2*100+1*10+j
        
        #plt.subplot(x);j=j+1;
        xa=np.append(xa,i);
        yb=np.append(yb,y);
        #plt.plot(i, y,"k.");
    plt.plot(xa[:],yb[:],"--");
    #plt.axis([-0.1, 1.1, -1, 1.5])
    plt.plot(0,0,'o');
    plt.plot(0,1,'o');
    plt.plot(1,0,'o');
    plt.plot(1,1,'o');
    plt.xlabel("x");
    plt.ylabel("y");
    #plt.show();  
    #for i in range(0,2):
       # abline_values = [slope * i + intercept]
        
        #data = np.array(dataset)
       # j=1
       # x=2*100+1*10+j
        #plt.subplot(x);j=j+1
       # plt.plot(j,abline_values);
    
       # plt.xlabel("x"+str(j))
       # plt.ylabel("Y"+str(j))
       # plt.show()
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.2,0.1,0.5]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f ' % (epoch, l_rate, sum_error));print(weights);plot(dataset,weights)
	return weights

# Calculate weights
#dataset = [[0,0,1],[0,1,0],[1,1,0],[1,0,0]]
l_rate = 0.1
n_epoch = 15


#weights = train_weights(dataset, l_rate, n_epoch)
dataset = [[0,0,0],[0,1,1],[1,1,1],[1,0,1]]
#weights = train_weights(dataset, l_rate, n_epoch)

#dataset = [[0,0,0],[0,1,0],[1,1,1],[1,0,0]]
#weights = train_weights(dataset, l_rate, n_epoch)
#dataset = [[0,0,1],[0,1,1],[1,1,0],[1,0,1]]
weights = train_weights(dataset, l_rate, n_epoch)
#plt.subplot(2,1,5);

print(weights)
