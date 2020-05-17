# /usr/bin/ python3

import numpy as np
import os
import matplotlib.pyplot as plt
import pylab
import math
import adaptiveDesignProcedure as adp
import time
import sys

# Forest paramters
forestParams={
'Ntree'       : 200,
'tps'         : 1,
'fraction'    : 0.7,
}

# Algorithm paramters
algorithmParams={
'dth'         : 0.1,   # thresold derivata
'd2th'        : 0.9,   # thresold derivata
'VIth'        : 0.15,  # thresold Var Imp
'errTh'       : 1e-6,  # thresold per calcolo errore
'OOBth'       : 0.05,  # thresold su OOBnorm
'RADth'       : 10,  # thresold on RAD 
'maxTDSize'   : 40000,   # maximum allowed size of the training data
'AbsOOBTh'    : 0.2,   # deviation between OOBs for iteration 0 for species > 0
}

# Files (input, training, query and benckmark)
trainingFile    = 'train.dat'
forestFile      = 'ml_ExtraTrees.pkl'
queryFile       = 'query_input.dat'
queryRest       = 'query_output.dat'


input_var = ( { 'name' : 'A', 'min' : 1e-3, 'max' : 1, 'num' : 4, 'typevar' : 'lin'},
            )
            
tabulation_var = (  {'name' : 'RA', 'typevar' : 'lin'},
                    
                 )

iterator = adp.adaptiveDesignProcedure(input_var, tabulation_var, forestFile, 
						trainingFile, forestParams, algorithmParams, 
						queryFile, queryRest)

# Crate training and RF
iterator.createTrainingDataAndML()





	











