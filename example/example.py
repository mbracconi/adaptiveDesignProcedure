# /usr/bin/ python3

"""
    /*----------------------------------------------------------------*\
    |                                                                  |
    |   Adaptive Refinement Procedure for Machine Leraning             |
    |                                                                  |
    |------------------------------------------------------------------|
    |                                                                  |
    |  Author: Mauro Bracconi                                          |
    |           mauro.bracconi@polimi.it                               |                         
    |           Politecnico di Milano                                  |
    |           Dipartimento di Energia                                |
    |           Laboratory of Catalysis and Catalytic Processes        |   
    |                                                                  |
    |------------------------------------------------------------------|
    |                                                                  |
    |   Copyright(C) 2019-2020 - M. Bracconi                           |
    |                                                                  |
    |------------------------------------------------------------------|
    |                                                                  |
    |   Reference:                                                     |
    |       M. Bracconi and M. Maestri, "Training set design for       |
    |       Machine Learning techniques applied to the approximation   |
    |       of computationally intensive first-principles kinetic      |
    |       models", Chemical Engineering Journal, 2020,               |
    |       DOI: 10.1016/j.cej.2020.125469                             |
    |                                                                  |
    |------------------------------------------------------------------|
    |                                                                  |
    |   Description:                                                   |
    |       Python class implementing the Adaptive Refinment           |
    |       Procedure for Machine Learning using ExtraTress            |
    |       and derivative-based addition of new points                |   
    |                                                                  |
    |   Version:                                                       |
    |       * 1.0 (02/10/2020): adaptive refinement procedure          |
    |       * 1.1 (04/08/2020): added MinMax scaling of tabulation     |
    |                           variables                              |
    |       * 1.2 (05/09/2020): added control on second derivative     |
    |                           to improve description around          |
    |                           stationary points                      |
    |                                                                  |
    \*----------------------------------------------------------------*/
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pylab
import math
import adaptiveDesignProcedure as adp
import time
import sys

# Parameters to reproduce "Showcase of the procedure" (Section 4.1) of M. Bracconi & M. Maestri, Chemical Engineering Journal, 2020, DOI: 10.1016/j.cej.2020.125469

# Forest paramters
forestParams={
        'Ntree'       : 200,
        'tps'         : 1,
        'fraction'    : 0.9,
    }

# Algorithm paramters
algorithmParams={
        'dth'         : 0.1,     # thresold first derivative
        'd2th'        : 0.9,     # thresold second derivative
        'VIth'        : 0.15,    # thresold variable importance
        'errTh'       : 1e-6,    # thresold for MRE error evaluation (remove from MRE calculation record below this value)
        'OOBth'       : 0.01,    # termination criterium on OOBnorm
        'RADth'       : 15,      # termination criterium on Relative Approximation Error (RAD) [%]
        'maxTDSize'   : 40000,   # maximum allowed size of the training data
        'AbsOOBTh'    : 0.2,     # maximum variations between OOB for two different tabulation variables
    }

# Files (input, training, query and benckmark)
trainingFile    = 'train.dat'
forestFile      = 'ml_ExtraTrees.pkl'
queryFile       = 'query_input.dat'
queryRest       = 'query_output.dat'


# Independent variables (i.e., descriptors)
input_var = ( { 'name' : 'A', 'min' : 1e-3, 'max' : 1, 'num' : 4, 'typevar' : 'lin'}, )
            
# Tabulated variables
tabulation_var = (  {'name' : 'Y', 'typevar' : 'lin'}, )

# Number of benchmark query points
query_p = 1000

# Initialize ADPforML class
adpML = adp.adaptiveDesignProcedure(input_var, 
                                    tabulation_var, 
                                    forestFile, 
                                    trainingFile, 
                                    forestParams, 
                                    algorithmParams, 
                                    queryFile, 
                                    queryRest)

# Create benchmark dataset (optional)
adpML.createBenchmarkDataset(query_p)

# Crate training and RF
adpML.createTrainingDataAndML()




	











