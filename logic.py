# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:19:01 2019

@author: sivak
"""

import numpy as np
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
# AND data
ANDtargets = np.array([[0],[0],[0],[1]])
# OR data
ORtargets = np.array([[0],[1],[1],[1]])
# XOR data
XORtargets = np.array([[0],[1],[1],[0]])
import pcn as pcn_logic_eg

print ("AND logic function")
pAND = pcn_logic_eg.pcn(inputs,ANDtargets)
pAND.pcntrain(inputs,ANDtargets,0.25,6)

print ("OR logic function")
pOR = pcn_logic_eg.pcn(inputs,ORtargets)
pOR.pcntrain(inputs,ORtargets,0.25,6)

print ("XOR logic function")
pXOR = pcn_logic_eg.pcn(inputs,XORtargets)
pXOR.pcntrain(inputs,XORtargets,0.25,6)