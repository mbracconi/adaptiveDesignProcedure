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
    |       Example of adaptiveDesign procedure reproducing "Showcase  |
    |       of the procedure"                                          |
    |                                                                  |
    \*----------------------------------------------------------------*/
"""

import numpy as np
import adaptiveDesignProcedure as adp

## Function which evaluates the values of the function to be tabulated

def getRate(xx):
    """Compute the function value

        Parameters
        ----------
            x : np.array[number records, number input variables]
                Input data

        Return
        ----------
            y : np.array[number records, number tabulation variables]
                Function values

    """
    x = xx[:,0]
    y = xx[:,1]

    z=100*np.exp(-( 2.5e8*(x-1e-4)**2 + 1e6*(y-1e-4)**2))

    return z.reshape(-1,1)


# Parameters to reproduce "Showcase of the procedure" (Section 4.1) of M. Bracconi & M. Maestri, Chemical Engineering Journal, 2020, DOI: 10.1016/j.cej.2020.125469
# Forest paramters
forestParams={
        'Ntree'       : 200,
        'tps'         : 1,
        'fraction'    : 0.7,
    }

# Algorithm paramters
algorithmParams={
        'dth'         : 0.1,     # thresold first derivative
        'd2th'        : 0.7,     # thresold second derivative
        'VIth'        : 0.15,    # thresold variable importance
        'errTh'       : 1e-6,    # thresold for MRE error evaluation (remove from MRE calculation record below this value)
        'OOBth'       : 0.05,    # termination criterium on OOBnorm
        'RADth'       : 30,      # termination criterium on Relative Approximation Error (RAD) [%]
        'maxTDSize'   : 40000,   # maximum allowed size of the training data
        'AbsOOBTh'    : 0.2,     # maximum variations between OOB for two different tabulation variables
    }

# Files (input, training, query and benckmark)
trainingFile    = 'train.dat'
forestFile      = 'ml_ExtraTrees.pkl'
queryFile       = 'query_input.dat'
queryRest       = 'query_output.dat'


# Independent variables (i.e., descriptors)
input_var = ( { 'name' : 'A', 'min' : 1e-6, 'max' : 0.1, 'num' : 4, 'typevar' : 'log'},
	      { 'name' : 'B', 'min' : 1e-6, 'max' : 0.1, 'num' : 4, 'typevar' : 'log'}, )

# Tabulated variables
tabulation_var = (  {'name' : 'Y', 'typevar' : 'log'}, )

# Number of benchmark query points
query_p = 1000

# Initialize ADPforML class
adpML = adp.adaptiveDesignProcedure(input_var,
                                    tabulation_var,
                                    getRate,
                                    forestFile,
                                    trainingFile,
                                    forestParams,
                                    algorithmParams,
                                    queryFile,
                                    queryRest,
                                    benchmark=True,
                                    plot=False,
                                    randomState=10)

# Create benchmark dataset (optional)
adpML.createBenchmarkDataset(query_p)

# Create training and RF
adpML.createTrainingDataAndML()

print('')
print('> Training data:')
x,y,z = adpML.trainingData.T
for i in range(len(x)):
    print('%10.3e'%x[i], '%10.3e'%y[i], '%10.3f'%z[i])

print('')
print('> Predicted data:')

x = np.logspace(-6,-1,5).reshape(-1,1)
xv,yv = np.meshgrid(x, x, indexing='xy')
inp = np.hstack([xv.reshape(-1,1), yv.reshape(-1,1)])
z = adpML.predict(inp)

for i in range(len(inp[:,0])):
    print('%10.3e'%inp[i,0], '%10.3e'%inp[i,1], '%10.3f'%z[i])
