#/usr/bin python3
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
import matplotlib.pyplot as plt
import os
import shutil
import time
import traceback
import joblib

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def getRate(press):
    pA = press[:,0]
		
    z = pA
    u = 150
	
    rate = 1/(1+np.exp(-u*(z-0.5)))*(1/(z**1))+1
    return rate.reshape(-1,1)


class adaptiveDesignProcedure:
    def __init__(slf, in_variables, 
					   tab_variables, 
					   forestFile, 
					   trainingFile,
                       forestParams, 
                       algorithmParams, 
                       queryFile = None, 
                       queryTabVar = None, 
                       benchmark = True, 
                       debug = False ):
						   
        """Class constructor
        
        Parameters
        ----------
            in_variables : tuple[dictionary]
                Dictionaries for each independent species reporting: 'name' -> variable name, 'min_range' -> minimum value,'max_range' -> maximum value, 'num' -> number of points, 'typevar' -> type of variable (log,lin,inv)
                example:
                input_var = ( 
                              { 'name' : 'A', 'min' : 1e-3, 'max' : 1, 'num' : 4, 'typevar' : 'lin'},
                              { 'name' : 'B', 'min' : 1e-3, 'max' : 1, 'num' : 3, 'typevar' : 'log'}
                            )
            out_variables : tuple[dictionary]
                Dictionaries for each tabulation species reporting: 'name' -> variable name, 'typevar' -> type of variable (log,lin)
                example:
                out_variables = (  
                                  {'name' : 'R_A', 'typevar' : 'lin'},
                                  {'name' : 'R_B', 'typevar' : 'lin'},
                                )
            forestFile : string
                Path to ExtraTress final trained algorithm
            trainingFile : string
                Path to training data file 
            forestParam : dictionary
                Structure with ExtraTrees parameters
                example:
                    forestParams=
                    {
                        'Ntree'       : 200, # number of decision trees in the forest
                        'tps'         : 1,   # number of record in terminal leaves
                        'fraction'    : 0.7, # fraction of data used to grow the forest
                    }
            algorithmParam : dictionary
                Structure with Adaptive Design Procedure parameters
                example:
                    algorithmParams=
                    {
                        'dth'         : 0.1,     # thresold first derivative
                        'd2th'        : 0.9,     # thresold second derivative
                        'VIth'        : 0.15,    # thresold variable importance
                        'errTh'       : 1e-6,    # thresold for MRE error evaluation (remove from MRE calculation record below this value)
                        'OOBth'       : 0.05,    # termination criterium on OOBnorm
                        'RADth'       : 10,      # termination criterium on Relative Approximation Error (RAD) [%]
                        'maxTDSize'   : 40000,   # maximum allowed size of the training data
                        'AbsOOBTh'    : 0.2,     # maximum variations between OOB for two different tabulation variables
                    }
            queryFile : string, optional
                Path to the query file (input variables)
            queryTabVar : string, optional
                Path to the query file (output variables)
            benchmark : bool, optional
                Calculation of benchmark error and plotting
            debug : bool, optional
                Print additional information during the iterations
        """
        
        # Assign forest and training data file path
        slf.forestFile         = forestFile
        slf.trainingFile       = 'tmp/'+trainingFile
        
        # Assign forest and algorithm params
        slf.forestParams       = forestParams
        slf.algorithmParams    = algorithmParams
        
        # Assign benchmark variables
        slf.benchmark          = benchmark
        slf.queryFile          = queryFile
        slf.queryTabVar        = queryTabVar
        slf.debug              = debug
        
        # Define variables name, legnth and size
        slf.numberOfInputVariables   = len(in_variables)
        slf.numberOfSpecies          = slf.numberOfInputVariables
        slf.numberOfTabVariables     = len(tab_variables)
        
        slf.headersInVar = []
        slf.typevarInVar = []
        slf.min_range = []
        slf.max_range = []
        slf.points_spec = []
        for spec in in_variables :
            slf.headersInVar.append(spec['name'])
            slf.typevarInVar.append(spec['typevar'])
            slf.min_range.append(spec['min'])
            slf.max_range.append(spec['max'])
            slf.points_spec.append(spec['num'])
            if (spec['name'] == 'T') :
                slf.numberOfSpecies -= 1
            if(spec['typevar'] != 'log' and spec['typevar'] != 'inv' and spec['typevar'] != 'lin') :
                print ('\nFATAL ERROR: variable type for',spec['name'],'not supported (log, inv, lin)\n')
                exit()
            if(spec['min'] <= 0) :
                print ('\nFATAL ERROR: minimum range for',spec['name'],'<= 0\n')
                exit()
            if(spec['num'] < 3) :
                print ('\nFATAL ERROR: minimum number of initial points for',spec['name'],'is 3\n')
                exit()

        slf.min_range = np.array(slf.min_range)
        slf.max_range = np.array(slf.max_range)
        slf.points_spec = np.array(slf.points_spec)
                
        slf.headersTabVar = []
        slf.typevarTabVar = []
        for tabv in tab_variables :
            slf.headersTabVar.append(tabv['name'])
            slf.typevarTabVar.append(tabv['typevar'])
            if(tabv['typevar'] != 'log' and tabv['typevar'] != 'lin') :
                print ('\nFATAL ERROR: variable type for',spec['name'],'not supported (log, lin)\n')
                exit()
        
        # Define lists of procedure results
        slf.benchmarkErrorEv    = [] # [iter 0 CO, iter 1 CO, ..., iter 0 H2O, iter 1 H2O, ... ]
        slf.trainingDataSize    = []
        slf.normOOB             = [] # [ [OOB #0, OOB #1, .... OOB #N], [OOB #0, OOB #1, .... OOB #N], [OOB #0, OOB #1, .... OOB #N], ...] #lista di liste
        slf.RAD                 = [] # [ [RAD #1, .... RAD #N], [RAD #1, .... RAD #N], [RAD #1, .... RAD #N], ...] #lista di liste
        
        # Construct and set Random Forest 
        slf.reg = ExtraTreesRegressor(random_state=10, n_estimators=slf.forestParams['Ntree'], max_features="auto", bootstrap = True, oob_score = True, max_samples = slf.forestParams['fraction'], min_samples_leaf=slf.forestParams['tps'])
       
        # Construct and set the scaler
        slf.scalerout = MinMaxScaler(feature_range=(1e-6,1))
        
        # For plotting purpose
        slf.fig = plt.figure(figsize=(10,4))
        slf.ax = slf.fig.subplots(nrows=1, ncols=2)
        plt.subplots_adjust(wspace=0.5,bottom = 0.2,right=0.85)
        # Panel - 1: Average benchmakrj error evolution wrt training points size
        slf.ax[0].set_ylabel(r'Average benchmark error [%]')
        slf.ax[0].set_xlabel(r'Number of training points [-]')
        # Panel - 2: Parity plot
        slf.ax[1].set_ylabel(r'Rate ET [kmol $\mathregular{m^{-2} s^{-1}}$]')
        slf.ax[1].set_xlabel(r'Rate MK [kmol $\mathregular{m^{-2} s^{-1}}$]')
        
        # Create supporting folders
        if os.path.exists('tmp') :
            shutil.rmtree('tmp')
        os.mkdir('tmp')
        if os.path.exists('figures') :
            shutil.rmtree('figures')
        os.mkdir('figures')
        
        # Printing
        print('\n------ Adaptive generation of Training Data for Machine Learning ------')
        print('\nInput parameters:')
        print('  * Forest file:', slf.forestFile)
        print('  * Training file:', slf.trainingFile)
        print('  * Figure path:', 'figures')
        print('\n  * Forest parameters:')
        print('    {')
        print('\n'.join('        {}: {}'.format(k, v) for k, v in forestParams.items()))
        print('    }')
        print('\n  * Algorithm parameters:')
        print('    {')
        print('\n'.join('        {}: {}'.format(k, v) for k, v in algorithmParams.items()))
        print('    }')
        print('\n\n  * Variables information:')
        for t in in_variables :
            print('    {')
            print('\n'.join('        {}: {}'.format(k, v) for k, v in t.items()))
            print('    }')
            
        print('\n\n  * Tabulation information:')
        for t in tab_variables :
            print('    {')
            print('\n'.join('        {}: {}'.format(k, v) for k, v in t.items()))
            print('    }')
        

    def trainExtraTressMISO(slf,trainingData) :
        """Train ExtraTrees algorithm for absolute value of the rate and for the sign of a single variables
        
        Parameters
        ----------
            trainingData : np.array
                Matrix consisting of the training data: first numberOfVariables columns are the descriptors followed by the absolute value rate and by the sign
        """
        # Load training data
        slf.reg.set_params(random_state=np.random.randint(low=1, high=20000))
        
        # Fit Trees
        slf.reg.fit(trainingData[:,0:slf.numberOfInputVariables],trainingData[:,-1])
        
      
    def trainExtraTressMIMO(slf,trainingData) :
        """Train ExtraTrees algorithm for absolute value of the rate and for the sign of all the variables togheter
        
        Parameters
        ----------
            trainingData : np.array
                Matrix consisting of the training data: first numberOfVariables columns are the descriptors followed by the absolute value rate and by the sign
        """
        # Load training data
        slf.reg.set_params(random_state=np.random.randint(low=1, high=20000), bootstrap=False, oob_score = False)
        
        ind_data = trainingData[:,slf.numberOfInputVariables:]
        if (ind_data.shape[1] == 1) :
            ind_data = ind_data.ravel()
        
        # Fit Trees
        slf.reg.fit(trainingData[:,0:slf.numberOfInputVariables],ind_data) 
 
    def approximationError(slf, queryDataVal,oldForestQuery, typevar) :
        """Compute the iterative approximation error (RAD) used for analysis of the convergence of the procedure as Mean Squared Logarithmic Error (MSLE)
        
        Parameters
        ----------
            queryDataVal : np.array
                Matrix consisting of the query point
            oldForestQuery : np.array
                Matrix consisting of the predictions of the queryDataValobtained with the ExtraTress obtained ad the previous iteration
            typevar : string
                Define how to treat the tabulated variable (either 'log' or 'lin')
                
        Return
        ----------
            avErrA : float
                Percentage approximation error 
        """
        # Evaluate result new queries
        if(typevar == 'log') :
            newForestQuery = 10**slf.reg.predict(queryDataVal)
        elif (typevar == 'lin') :
            newForestQuery = slf.reg.predict(queryDataVal)
            
        # Select rate larger than th
        idxA = np.where(np.abs(oldForestQuery) >= slf.algorithmParams['errTh'])
        # Compute approximation error in terms of MRE
        errA = np.abs(oldForestQuery[idxA]-newForestQuery[idxA])/np.abs(oldForestQuery[idxA])
        avErrA = np.average(errA)*100.
        return avErrA
        
    def benchmarkError(slf,indexTabVariable,typevar, count, msle = False) :
        """Compute the benchmark error used for analysis of accuracy of the procedure by evaluating the Mean Squared Logarithmic Error MSLE (default) or Mean Relative Error MRE: :math:`MSLE=1/N \sum_{i=1}^{n_{query}}{(\log(y_i+1)-\log(\widehat{y}_i+1))^2}` \\ :math:`MRE=1/N \sum_{i=1}^{n_{query}}{|y_i-\widehat{y}_i|/|\widehat{y}_i|}`
        
        Parameters
        ----------
            indexTabVariable : int
                Index of the variables considered
            typevar : string
                Define how to treat the tabulated variable (either 'log' or 'lin')
            msle : bool, optional
                Matrix consisting of the predictions of the queryDataVal obtained with the ExtraTress obtained ad the previous iteration
                
        Return
        ----------
            err : np.array
                Vector of the error
        """
        ratesDI = np.loadtxt(slf.queryTabVar,skiprows=1,delimiter=',',usecols=(indexTabVariable))                
        queryData = np.loadtxt(slf.queryFile, skiprows=1, delimiter=',')
        
        if(len(queryData.shape) == 1) :
            queryData = queryData.reshape(-1,1)
        
        if(typevar == 'log') :
            pred = 10**slf.reg.predict(queryData)
        elif (typevar == 'lin') :
            pred = slf.reg.predict(queryData)
               
        rates_real_scaled = ratesDI #slf.scalerout.transform(ratesDI.reshape(-1,1)).ravel()
        rates_etrt_scaled = slf.scalerout.inverse_transform(pred.reshape(-1,1)).ravel() #pred # slf.scalerout.transform(pred.reshape(-1,1))
        
        np.savetxt('bench_'+str(count)+'_'+slf.headersInVar[indexTabVariable]+'.dat',np.c_[queryData,rates_etrt_scaled],header=str(slf.headersInVar),comments='#') 
        
        idx1 = np.where(np.abs(rates_real_scaled) >= slf.algorithmParams['errTh'])
        if msle :
            err = (np.log10(np.abs(rates_real_scaled[idx1]+1))-np.log10(np.abs(rates_etrt_scaled[idx1]+1)))**(2)
        else :
            err = np.abs(rates_real_scaled[idx1]-rates_etrt_scaled[idx1])/np.abs(np.array(rates_real_scaled[idx1]))

        return err
        
    def plotTrends(slf,indexTabVariable,iterC,typevar) :
        """Plot evolution of benchmark error and a parity plot of the benchmark predictions at each iteration
        
        Parameters
        ----------
            indexTabVariable : int
                Index of the variables considered
            iterC : int
                Current iteration of the procedure
            typevar : string
                Define how to treat the tabulated variable (either 'log' or 'lin')
                
        """

        ratesDI = np.loadtxt(slf.queryTabVar,skiprows=1,delimiter=',',usecols=(indexTabVariable))                
        queryData = np.loadtxt(slf.queryFile, skiprows=1, delimiter=',')
        
        if(len(queryData.shape) == 1) :
            queryData = queryData.reshape(-1,1)
        
        if(typevar == 'log') :
            pred = slf.scalerout.inverse_transform((10**slf.reg.predict(queryData)).reshape(-1,1))
        elif (typevar == 'lin') :
            pred = slf.scalerout.inverse_transform((slf.reg.predict(queryData)).reshape(-1,1))
       
        idx1 = np.where(np.abs(ratesDI) >= slf.algorithmParams['errTh'])

        colors = ['k','b','g','r','m']
        if(iterC == 0) :
        #slf.ax[0].scatter(np.array(slf.trainingDataSize), np.array(slf.benchmarkErrorEv), c='black')
            slf.ax[0].plot(slf.trainingDataSize[-1],slf.benchmarkErrorEv[-1], colors[indexTabVariable]+'o', markersize=5, label = slf.headersTabVar[indexTabVariable])
        else :
            slf.ax[0].plot(slf.trainingDataSize[-1],slf.benchmarkErrorEv[-1], colors[indexTabVariable]+'o', markersize=5)
        ymin, ymax = slf.ax[0].get_ylim()
        xmin, xmax = slf.ax[0].get_xlim()
        plt.sca(slf.ax[0])
        plt.xlim(min(xmin,np.min(np.array(slf.trainingDataSize)*0.8)),max(xmax,np.max(np.array(slf.trainingDataSize)*1.2)))
        plt.ylim(0,max(ymax,np.max(np.array(slf.benchmarkErrorEv))*1.2))
        plt.legend(loc='upper right')
        plt.draw()
        plt.pause(0.001)
        
        pline=np.array([np.min(ratesDI[idx1])*0.8,np.max(ratesDI[idx1])*1.2])
        ticks = np.linspace(np.min(pline),np.max(pline),6)
        if(iterC == 0) :
            plt.sca(slf.ax[1])
            plt.cla()
            plt.xlim(np.min(pline),np.max(pline))
            plt.ylim(np.min(pline),np.max(pline))
            slf.ax[1].plot(pline,pline,'k-',linewidth=1)
            slf.ax[1].plot(pline,pline*0.7,'k--',linewidth=0.5)
            slf.ax[1].plot(pline,pline*1.3,'k--',linewidth=0.5)
            plt.xticks(ticks)
            plt.yticks(ticks)

        slf.ax[1].plot(ratesDI[idx1],pred[idx1], 'o', markersize=3,label = slf.headersTabVar[indexTabVariable] + ' - #' + str(iterC))
        slf.ax[1].set_ylabel(slf.headersTabVar[indexTabVariable] + r'$\mathregular{_{ET}}$')
        slf.ax[1].set_xlabel(slf.headersTabVar[indexTabVariable] + r'$\mathregular{_{MD}}$')
        plt.sca(slf.ax[1])
        slf.ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.draw()
        plt.pause(0.001)
        plt.savefig('figures/trend_'+slf.headersInVar[indexTabVariable]+'_'+str(iterC)+'.tif', dpi=600)

    def plotParity(slf) :
        """Plot parity plot with ExtraTrees obtained at the end of the procedure
        
        Parameters
        ----------
            indexTabVariable : int
                Index of the variables considered
            iterC : int
                Current iteration of the procedure
            typevar : string
                Define how to treat the tabulated variable (either 'log' or 'lin')
                
        """

        ratesDI = np.loadtxt(slf.queryTabVar,skiprows=1,delimiter=',')                
        queryData = np.loadtxt(slf.queryFile, skiprows=1, delimiter=',')
        
        if(len(ratesDI.shape) == 1) :
            ratesDI = ratesDI.reshape(-1,1)
			
        if(len(queryData.shape) == 1) :
            queryData = queryData.reshape(-1,1)
        
        pred = slf.reg.predict(queryData)
        
        if(len(pred.shape) == 1) :
            pred = pred.reshape(-1,1)
        
        for k in range(slf.numberOfTabVariables) :
            if(slf.typevarTabVar[k] == 'log') :
                pred[:,k] = slf.scalerout.inverse_transform(10**pred)[:,k]
            elif (slf.typevarTabVar[k] == 'lin') :
                pred[:,k] = slf.scalerout.inverse_transform(pred)[:,k]

        for k in range(slf.numberOfTabVariables) :
            plt.figure()
            pline=np.array([min(np.min(ratesDI[:,k]),np.min(pred))*0.8,max(np.max(ratesDI[:,k]),np.max(pred))*1.2])
            ticks = np.linspace(np.min(pline),np.max(pline),6)
            plt.xlim(np.min(pline),np.max(pline))
            plt.ylim(np.min(pline),np.max(pline))
           
			
            plt.plot(ratesDI[:,k],pred[:,k], 'o', markersize=3)
            plt.ylabel(slf.headersTabVar[k] + r'$\mathregular{_{ET}}$')
            plt.xlabel(slf.headersTabVar[k] + r'$\mathregular{_{MD}}$')
            plt.plot(pline,pline,'k-',linewidth=1)
            plt.plot(pline,pline*0.7,'k--',linewidth=0.5)
            plt.plot(pline,pline*1.3,'k--',linewidth=0.5)
            plt.legend(slf.headersTabVar, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xticks(ticks)
            plt.yticks(ticks)
			
            plt.tight_layout()
            plt.savefig('figures/parity_'+slf.headersTabVar[k]+'.tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"})



    ## Function which select the position for the newly added points at each iteration
    def findPoints(slf, trainData, pointsToAdd):
        """Evaluate the position of the new points based on the discrete function gradient
        
        Parameters
        ----------
            trainData : np.array
                Matrix consisting of the training data variables and function rate
            pointsToAdd : np.array
                Variables considered for the addition defined based on the variable importance
                
        Return
        ----------
            newPressures : np.array
                Positions of the new points
        """
        newPressures = []
        
        firstDerivative = []
        secondDerivative = []
            
        for j in range(slf.numberOfInputVariables):
            # Retrieve the existing intervals
            p_unique, index = np.unique(trainData[:,j],return_index=True)
            n_intervals = len(p_unique)-1    
            
            # Store old points
            new_p = p_unique
            # Create a empty list with the position of nthe new points
            add_p=[]
            # Create a list which stores the derivatives in each interval
            der = []    
            trer = []
            trer2 = []
            
            # Compute the derivatives in each of the interval for direction j
            # Example: 2 species and 3 points per each
            #    o --------- o --------- o
            #    |           |           |
            #    |           |           |
            #    |           |           |
            #    o --------- o --------- o
            #    |           |           |
            #    |           |           |
            #    |           |           |
            #    o --------- o --------- o

            # By looking at the first direction (vertical) we have to compute the derivative
            # in two intervals, but since we have 3 different value of the other variable
            # we have to compute three derivative for each interval. The derivative are evaluated
            # in the middle point between the two training points (X)
            #    o --------- o --------- o
            #    |           |           |
            #    X           X           X
            #    |           |           |
            #    o --------- o --------- o
            #    |           |           |
            #    X           X           X
            #    |           |           |
            #    o --------- o --------- o

            # We need an information of derivative in an interval, hence we store the maximum
            # derivative in each interval

            # Then we refine only in the interval where the derivative is larger than a 
            # selected threasold 
            if pointsToAdd[j] > 0:  
            # ------------------------------------------------------------------------------------------------------------------
                # Compute first derivative in each interval
                for i in range(n_intervals):
                    
                    p_temp_fi = trainData[trainData[:,j] == p_unique[i]]
                    p_temp_si = trainData[trainData[:,j] == p_unique[i+1]]
                    locder = []
                   
                    point_len = len(p_temp_fi)
                    
                    for k in range(point_len) :
						
                        if (slf.typevarInVar[j] == 'log') :
                            derj = np.abs(((p_temp_si[k,slf.numberOfInputVariables]) - (p_temp_fi[k,slf.numberOfInputVariables]))/(np.log10(p_unique[i+1])-np.log10(p_unique[i])))
                        elif (slf.typevarInVar[j] == 'lin') :
                            derj = np.abs(((p_temp_si[k,slf.numberOfInputVariables]) - (p_temp_fi[k,slf.numberOfInputVariables]))/((p_unique[i+1])-(p_unique[i])))
                        elif (slf.typevarInVar[j] == 'inv') :
                            derj = np.abs(((p_temp_si[k,slf.numberOfInputVariables]) - (p_temp_fi[k,slf.numberOfInputVariables]))/(1./p_unique[i+1]-1./p_unique[i]))

                        locder.append(derj)
                    
                    der.append(max(locder))    

                der = np.abs(np.array(der))        
                # ------------------------------------------------------------------------------------------------------------------
                # Compute second derivative in each interval

                for i in range(n_intervals):
                    if i > 0 :
                        p_temp_i = trainData[trainData[:,j] == p_unique[i]]
                        p_temp_ip = trainData[trainData[:,j] == p_unique[i+1]]
                        p_temp_im = trainData[trainData[:,j] == p_unique[i-1]]
                                                
                        dx1 = p_unique[i]-p_unique[i-1]
                        dx2 = p_unique[i+1]-p_unique[i]
                        dx3 = p_unique[i+1]-p_unique[i-1]
                        
                        locapp = []
                        locappd2 = []
                        point_len = len(p_temp_i)
                        for k in range(point_len) :
                            a = p_temp_ip[k,slf.numberOfInputVariables]    
                            b = p_temp_i[k,slf.numberOfInputVariables]    
                            c = p_temp_im[k,slf.numberOfInputVariables]    
                            
                            der1jFO = (a-b)/dx2
                            der1jHO = (a*dx1**2-c*dx2**2+b*(dx1**2-dx2**2))/(dx1*dx2*(dx1+dx2))
                            der2j = (a*dx1 + c*dx2 - b*dx3)/(0.5*dx1*dx2*dx3);
                            der2j *= dx2/2.0
                            
                            locapp.append(np.abs(der1jHO-der1jFO))
                            locappd2.append(np.abs(der2j))
                        
                        trer.append(np.max(locappd2))
                        trer2.append(np.max(locapp))
                        
                    else :
                        p_temp_i = trainData[trainData[:,j] == p_unique[i]]
                        p_temp_ip = trainData[trainData[:,j] == p_unique[i+1]]
                        p_temp_ipp = trainData[trainData[:,j] == p_unique[i+2]]
                        
                        dx1 = p_unique[i+1]-p_unique[i]
                        dx2 = p_unique[i+2]-p_unique[i+1]
                        dx3 = p_unique[i+2]-p_unique[i]
                        
                        locapp = []
                        locappd2 = []
                        point_len = len(p_temp_i)
                        for k in range(point_len) :
                            a = p_temp_ipp[k,slf.numberOfInputVariables]    
                            b = p_temp_ip[k,slf.numberOfInputVariables]    
                            c = p_temp_i[k,slf.numberOfInputVariables]    
                        
                            der1jFO = (b-c)/dx1
                            der1jHO = (-a*dx1**2+b*dx3**2-c*(dx3**2-dx1**2))/(dx1*dx2*dx3)
                            
                            der2j = (a*dx1 + c*dx2 - b*dx3)/(0.5*dx1*dx2*dx3);
                            der2j *= dx2/2.0
                            locappd2.append(der2j)
                            locapp.append(np.abs(der1jHO-der1jFO))
                            
                        trer.append(np.max(locappd2))
                        trer2.append(np.max(locapp))
                            
                # ------------------------------------------------------------------------------------------------------------------
            else :
                der = 0
                trer = 0
                
            firstDerivative.append(np.abs(der))
            secondDerivative.append(np.abs(trer))
            

        maxDerList = []
        maxTDerList = []
        for j in range(slf.numberOfInputVariables):
            localMax = np.average(firstDerivative[j])
            maxDerList.append(localMax)
            localMax = np.max(secondDerivative[j])
            maxTDerList.append(localMax)
        
        maxDerList = np.array(maxDerList)
        maxDer = np.average(maxDerList)
        
        maxTDerList = np.array(maxTDerList)
        maxTDer = np.max(maxTDerList)
        
        for j in range(slf.numberOfInputVariables):
            # Retrieve the existing intervals
            p_unique, index = np.unique(trainData[:,j],return_index=True)
            n_intervals = len(p_unique)-1    
            poi = []
            # Store old points
            new_p = p_unique
            # Create a empty list with the position of nthe new points
            add_p=[]
            
            if pointsToAdd[j] > 0: 
                # Compute first derivative in each interval
                for i in range(n_intervals):
                    
                    p_temp_fi = trainData[trainData[:,j] == p_unique[i]]
                    p_temp_si = trainData[trainData[:,j] == p_unique[i+1]]
            
                    # log average between points for species, normal for temperature
                    if (slf.typevarInVar[j] == 'log') :
                        sp = np.average([np.log10(p_unique[i]),np.log10(p_unique[i+1])])
                        add_p.append(10.**sp)   
                    elif (slf.typevarInVar[j] == 'lin') :
                        sp = np.average([(p_unique[i]),(p_unique[i+1])])
                        add_p.append(sp)   
                    elif (slf.typevarInVar[j] == 'inv') :
                        sp = np.average([1./p_unique[i],1./(p_unique[i+1])])
                        add_p.append(1./sp) 
        
                    
                # Add points where:
                # 1/ The first derivative is higher than a thresold
                # 2/ The leading terms of the first derivative are larger than a thresold    
                # Both 1/ and 2/ are normalized respect to the function maximum value
                der = firstDerivative[j]
                trer = secondDerivative[j]

                for i in range(der.shape[0]) :
                    if  np.abs(der[i]/np.max(der)) > slf.algorithmParams['dth'] :
                        poi.append(add_p[i])
                    elif np.abs(trer[i]/np.max(trer)) > slf.algorithmParams['d2th']  :
                        poi.append(add_p[i])

                new_p = np.append(new_p, poi)
                    
                newPressures.append(new_p)
            else:
                newPressures.append(p_unique)
        
        return newPressures

    def addVariables(slf,indexTabVariable, equidistantPoints = 0):
        """Adaptively and iteratively add points for each tabulation variables

        Parameters
        ----------
            indexTabVariable : int
                Index of the tabulation variables considered
            equidistantPoints : int, optional
                Number of points in each direction of an optional evenly-distributed grid
        """
		
        print('\n  * Tabulation Variables:',slf.headersTabVar[indexTabVariable])

        imp = np.zeros([10,slf.numberOfInputVariables])
        
        # Initialize iterator counts and set iterate to true
        count = 0
        iterate = True
        
        # Initialize storage of OOB and RAD
        OOBspecies = []
        RADspecies = []
   
        # initialization for approximation error
        errA = 1000
        
        p_var = []

        if equidistantPoints :
            
            for i in range(slf.numberOfInputVariables) :
                if (slf.typevarInVar[i] == 'log') :
                    pi = np.logspace(np.log10(slf.min_range[i]),np.log10(slf.max_range[i]), num=equidistantPoints)
                elif (slf.typevarInVar[i] == 'lin') :
                    pi = np.linspace(slf.min_range[i],slf.max_range[i], num=equidistantPoints)
                elif (slf.typevarInVar[i] == 'inv') :
                    pi = 1./np.linspace(1./slf.max_range[i],1./slf.min_range[i], num=equidistantPoints)
                p_var.append(pi)
                
            pointsPerSpec = [equidistantPoints]*len(slf.numberOfInputVariables)

        else :
            if indexTabVariable == 0:   
                    # Initialize from strach the training points                     
                    for i in range(slf.numberOfInputVariables) :
                        if (slf.typevarInVar[i] == 'log') :
                            pi = np.logspace(np.log10(slf.min_range[i]),np.log10(slf.max_range[i]), num=slf.points_spec[i])
                        elif (slf.typevarInVar[i] == 'lin') :
                            pi = np.linspace(slf.min_range[i],slf.max_range[i], num=slf.points_spec[i])
                        elif (slf.typevarInVar[i] == 'inv') :
                            pi = 1./np.linspace(1./slf.max_range[i],1./slf.min_range[i], num=slf.points_spec[i])
                        p_var.append(pi)
                    
                    pointsPerSpec = slf.points_spec

            else:
                    # Read from file  
                    pOld=np.loadtxt(slf.trainingFile,delimiter=',',skiprows=1,usecols=tuple([i for i in range(slf.numberOfInputVariables)]))
                    
                    for i in range(slf.numberOfInputVariables) :
                        if (slf.typevarInVar[i] == 'log') :
                            pi = 10**np.unique(pOld[:,i])
                        elif (slf.typevarInVar[i] == 'lin') :
                            pi = np.unique(pOld[:,i])
                        elif (slf.typevarInVar[i] == 'inv') :
                            pi = np.sort(1./(np.unique(pOld[:,i])))
                        p_var.append(pi)

                    pointsPerSpec = [len(i) for i in p_var]

        while iterate:
            if equidistantPoints:
                print ('\n\n\n * Equidistant points ')
                
                iterate = False

                print ('   --------------------------- ')
                print ('   Points per species:', ' '.join( str(e) for e in pointsPerSpec))
                print ('   ---------------------------')
            else :
                print ('\n    * Iteration: ',count)
                if count > 0:
                    
                    biggestImp  = np.max(importances)
                    
                    pointsToAdd = np.zeros(slf.numberOfInputVariables)
                    locAdd = np.where(importances/biggestImp > slf.algorithmParams['VIth'])
                    pointsToAdd[locAdd] = 1

                    print ('      Normalized variable importance:', ' '.join( str(e) for e in (importances/biggestImp)))
                    
                    
                    newPress = slf.findPoints(trainingData[:,:], pointsToAdd)
                    pointsPerSpec = [len(j) for j in newPress]
                    
                    p_var = []
                    for i in range(slf.numberOfInputVariables) :
                        p_var.append(np.sort(newPress[i]))

                    print ('      --------------------------- ')
                    print ('      Points per species:', ' '.join( str(e) for e in pointsPerSpec))
                    print ('      ---------------------------')
                else :
                    print ('      --------------------------- ')
                    print ('      Points per species:', ' '.join( str(e) for e in pointsPerSpec))
                    print ('      ---------------------------')

            # Store past data for avoid redundant calculations
            if count > 0 :
                trD = trainingData
            
            # Generate training data meshgrid
            input_var_list = tuple(p_var)
            mgrid = np.meshgrid(*input_var_list,indexing='ij')
            trainingData = mgrid[0].ravel()
            for k in range(1,len(mgrid)) :
                trainingData = np.c_[trainingData, mgrid[k].ravel()]
            
            if (len(trainingData.shape) == 1) : # one independent variable
                trainingData = trainingData.reshape(-1,1)
            
            print ('      Total number of points: ', str(trainingData.shape[0]))

            # Compute new training points
            ratesAll = []
            rates = []
            kmcTime = time.time()
            if count > 0 :
                print ('      New points            :', str(trainingData.shape[0]-trD.shape[0]), '\n')
                added   = []
                addedra = []
                addedrs = []    
                ratesAll = []
                cont = 0            
                
                # if number of processes is not specified, it uses the number of core
                kmcTime = time.time()

                # Get rates
                ratesAll = np.array(getRate(trainingData)) 
                if(ratesAll.shape[1] != slf.numberOfTabVariables) :
                    print ('\nFATAL ERROR: shape of tabulation variable matrix is wrong, obtained:',ratesAll.shape[1],'expected:', slf.numberOfTabVariables)
                    exit()
                rates = ratesAll[:,indexTabVariable]
                rates_plot = rates
                # Scale
                #slf.scalerout.fit(rates.reshape(-1,1))
                rates = slf.scalerout.transform(rates.reshape(-1,1))
                # Store scaled value of rates
                if (slf.typevarTabVar[indexTabVariable] == 'log') : 
                    rates = np.log10(np.abs(rates)).ravel() 
                elif (slf.typevarTabVar[indexTabVariable] == 'lin') :
                    rates = np.abs(rates).ravel()
                    
                print ('      MK solved in', str(time.time()-kmcTime))

                for kk, gg in enumerate(trainingData) :
                    index = np.where(np.all((trD[:,0:slf.numberOfInputVariables] == gg),axis=1) == True)[0]

                    if len(index) == 0 :
                        addedra.append(rates[kk])
                        added.append(gg)
                        cont += 1
                                
                # Build on the fly query
                added = np.array(added)
                
                # Compute the approximation value using the previous iteration forest of the new point 
                # required for computing the approximation error
                        
                # Predict with the old forest 
                # Build query
                queryDataVal = np.empty((added.shape[0],slf.numberOfInputVariables))
                for k in range(slf.numberOfInputVariables) :
                    if (slf.typevarInVar[k] == 'log') :
                        queryDataVal[:,k] = np.log10(added[:,k])
                    elif (slf.typevarInVar[k] == 'lin') :
                       queryDataVal[:,k] = added[:,k]
                    elif (slf.typevarInVar[k] == 'inv') :
                        queryDataVal[:,k] = 1./added[:,k]
                        
                if (slf.typevarTabVar[indexTabVariable] == 'log') : 
                    oldForestQuery = 10**slf.reg.predict(queryDataVal)
                elif (slf.typevarTabVar[indexTabVariable] == 'lin') :
                    oldForestQuery = slf.reg.predict(queryDataVal)
                
            
                addedSup = queryDataVal
                addedra = np.array(addedra)
                addedrs = np.array(addedrs)
                addedra = np.expand_dims(addedra,1)
                addedSup = np.append(addedSup, slf.scalerout.inverse_transform(addedra), axis=1)
                
                # Print on file the values of the new training points (just for nice pictures)
                np.savetxt('train_'+str(count)+'_'+slf.headersInVar[indexTabVariable]+'.dat',addedSup,header=str(slf.headersInVar),comments='#') 
                
            else:
                if indexTabVariable != 0 :
                    rates=np.loadtxt('rates.dat',skiprows=1,delimiter=',',usecols=(indexTabVariable))
                    rates_plot = rates
                    slf.scalerout.fit(rates.reshape(-1,1))
                    rates=slf.scalerout.transform(rates.reshape(-1,1))
                    if (slf.typevarTabVar[indexTabVariable] == 'log') : 
                        rates = np.log10(np.abs(rates)).ravel() 
                    elif (slf.typevarTabVar[indexTabVariable] == 'lin') :
                        rates = np.abs(rates).ravel()
                   
                    print ('\n      MK loaded in', str(time.time()-kmcTime))
                    
                else :
                    # First species
                    #for gg in trainingData :
                    #    ratesAll.append(getRate(np.expand_dims(gg,1).transpose()))
                    ratesAll = getRate(trainingData) #ratesAll = pool.map(getRate, ( (np.expand_dims(gg,1)).transpose() for gg in trainingData) )
                    
                    if(ratesAll.shape[1] != slf.numberOfTabVariables) :
                        print ('\nFATAL ERROR: shape of tabulation variable matrix is wrong, obtained:',ratesAll.shape[1],'expected:', slf.numberOfTabVariables)
                        exit()
                    ratesAll = np.array(ratesAll)
                    rates = ratesAll[:,indexTabVariable]
                    rates_plot = rates
                    slf.scalerout.fit(rates.reshape(-1,1))
                    rates=slf.scalerout.transform(rates.reshape(-1,1))
                    if (slf.typevarTabVar[indexTabVariable] == 'log') : 
                        rates = np.log10(np.abs(rates)).ravel() 
                    elif (slf.typevarTabVar[indexTabVariable] == 'lin') :
                        rates = np.abs(rates).ravel()
                        
                    np.savetxt('train_'+str(count)+'_'+slf.headersInVar[indexTabVariable]+'.dat',np.c_[trainingData,slf.scalerout.inverse_transform(rates.reshape(-1,1)).ravel()],header=str(slf.headersInVar),comments='#')
                    print ('\n      MK solved in', str(time.time()-kmcTime))
                    

            # Creo training data
            rates=np.expand_dims(rates,1)
            rates_plot=np.expand_dims(rates_plot,1)
            
            trainingData = np.append(trainingData, rates, axis=1) # aggiungo valore assoluto della rate in logscale
            
            # Creo training data per ExtraTrees
            trainingDataSupp = np.empty((trainingData.shape[0],slf.numberOfInputVariables))
            for k in range(slf.numberOfInputVariables) :
                if (slf.typevarInVar[k] == 'log') :
                    trainingDataSupp[:,k] = np.log10(trainingData[:,k])
                elif (slf.typevarInVar[k] == 'lin') :
                   trainingDataSupp[:,k] = trainingData[:,k]
                elif (slf.typevarInVar[k] == 'inv') :
                    trainingDataSupp[:,k] = 1./trainingData[:,k]
           
            # Append rates
            trainingDataSupp = np.append(trainingDataSupp, rates, axis=1) #aggiungo valore assoluto

            #write training file(for both training datasets)
            np.savetxt(slf.trainingFile,trainingDataSupp,delimiter=',',header=str(slf.headersInVar)) 
            if (count > 0) :
                np.savetxt('rates.dat',ratesAll,delimiter=',',header=str(slf.headersTabVar)) 

            # Ranger grow forest (done 10 times to get a less biased OOB especially in the case of small dataset)
            OOB = []
            # Ranger grow forest (done 10 times to get a less biased OOB especially in the case of small dataset)
            
            for k in range(10) :
                slf.trainExtraTressMISO(trainingDataSupp)
                
                #OOB.append(1 - slf.reg.oob_score_)
                OOB.append(mean_squared_error(trainingDataSupp[:,-1],slf.reg.oob_prediction_))
                                
                # Store Variable importance
                imp[k,:] = np.array(permutation_importance(slf.reg,trainingDataSupp[:,0:slf.numberOfInputVariables],trainingDataSupp[:,-1],n_repeats=10,scoring='r2').importances_mean) #np.array(slf.reg.feature_importances_) 
                #print ('\n      Gini variable importance:', imp[k,:])
                #print(np.array(permutation_importance(slf.reg,trainingDataSupp[:,0:slf.numberOfInputVariables],trainingDataSupp[:,-1],n_repeats=10,scoring='r2').importances_mean))
                
            
            OOB = np.mean(np.array(OOB))  
            
            importances=np.mean(imp,axis=0) 
            if(slf.debug) :
                print ('\n      VI importance:', importances)
            joblib.dump(slf.reg, 'tmp/rf_'+slf.headersInVar[indexTabVariable]+'.pkl')
                      
            # Compute the approximation error
            # To avoid benchmarking, the procedure computes the approximation error iteration per iteration
            # This error tells the quality of the improvement given by the newly added points. If this error is small
            # it means that the new points are useless hence the iterative procedure reached the end
            if count > 0 :
                errA = slf.approximationError(queryDataVal,oldForestQuery,slf.typevarTabVar[indexTabVariable])
                RADspecies.append(errA)  #Add RAD to storage list
                
            # Add OOB storage list
            OOBspecies.append(OOB)
            print ('\n      Approximation quality:')
            print ('          Out-Of-Bag error     : ', OOB)
            if count > 0 :
                print ('          Iterative approx err : ', errA, '%')
                
            # Load rates from query file as real value     
            ratesDI = np.loadtxt(slf.queryTabVar, skiprows=1,delimiter=',',usecols=(indexTabVariable)) 

            if (slf.benchmark) :
                errMSLE = slf.benchmarkError(indexTabVariable,slf.typevarTabVar[indexTabVariable],count,msle=True)
                errMRE = slf.benchmarkError(indexTabVariable,slf.typevarTabVar[indexTabVariable],count,msle=False)
                print ('\n      Benchmark calculations:')
                print ('          Av. Benchmark error (MRE): ',np.average(errMRE)*100.,'%')           
                print ('          Max. Benchmark error     : ',np.max(errMRE)*100.,'%')
                
                slf.benchmarkErrorEv.append(np.average(errMRE)*100.)
                slf.trainingDataSize.append(trainingData.shape[0])
                slf.plotTrends(indexTabVariable,count,slf.typevarTabVar[indexTabVariable])

            # Exit strategy : max counts 8 | approx error < 5 %
            if equidistantPoints == 0 :
                if indexTabVariable == 0 :
                    if (errA > slf.algorithmParams['RADth'] or OOB > slf.algorithmParams['OOBth']) and trainingData.shape[0] < slf.algorithmParams['maxTDSize']:
                        iterate = True
                        count += 1

                    elif (errA > slf.algorithmParams['RADth'] or OOB > slf.algorithmParams['OOBth']) and trainingData.shape[0] >= slf.algorithmParams['maxTDSize'] : 
                        iterate = False
                        slf.normOOB.append(OOBspecies)
                        slf.RAD.append(RADspecies)
                        print ('\n      Maximum size of training data reached in',count+1,' iterations')
                        if slf.debug :
                            print('          OOB Evolution: ',OOBspecies)
                            print('          RAD Evolution: ',RADspecies) 
                    
                    else :
                        iterate = False
                        slf.normOOB.append(OOBspecies)
                        slf.RAD.append(RADspecies)
                        print ('\n      Accuracy constraints reached in',count+1,' iterations')
                        if slf.debug :
                            print('          OOB Evolution: ',OOBspecies)
                            print('          RAD Evolution: ',RADspecies)   
                else :
                    if count == 0:
                        maxPreviousOOB = 0.
                        for k in range(indexTabVariable) :
                            maxPreviousOOB = max(maxPreviousOOB,slf.normOOB[k][-1])
                        if slf.debug :
                            print('          MaxOld OOB:',maxPreviousOOB)
						 
                        if OOBspecies[-1] < maxPreviousOOB or np.abs(OOBspecies[-1] - maxPreviousOOB)/maxPreviousOOB < slf.algorithmParams['AbsOOBTh'] :  #se OOB è minore degli altri ->OK ma consì non fosse guarda quanto distante è OOB dal precedente e se sotto un thresold accetta la differenza
                            iterate = False
                            slf.normOOB.append(OOBspecies)
                            slf.RAD.append([])
                            print ('\n      Accuracy constraints reached in',count+1,'iterations')
                            if slf.debug :
                                print('          OOB Evolution: ',OOBspecies)
                        else :
                            iterate = True
                            count += 1
                    else :
                        if (errA > slf.algorithmParams['RADth'] or OOB > slf.algorithmParams['OOBth']) and trainingData.shape[0] < slf.algorithmParams['maxTDSize']:
                            iterate = True
                            count += 1

                        elif (errA > slf.algorithmParams['RADth'] or OOB > slf.algorithmParams['OOBth']) and trainingData.shape[0] > slf.algorithmParams['maxTDSize'] : 
                            iterate = False
                            slf.normOOB.append(OOBspecies)
                            slf.RAD.append(RADspecies)
                            print ('\n      Maximum size of training data reached in',count+1,'iterations')
                            if slf.debug :
                                print('          OOB Evolution: ',OOBspecies)
                                print('          RAD Evolution: ',RADspecies) 
                        else :
                            iterate = False
                            slf.normOOB.append(OOBspecies)
                            slf.RAD.append(RADspecies)
                            print ('\n      Accuracy constraints reached in',count+1,'iterations')
                            if slf.debug :
                                print('          OOB Evolution: ',OOBspecies)
                                print('          RAD Evolution: ',RADspecies) 
                

    def createTrainingDataAndML(slf, equidistantPoints = 0):
        """Generate the training set by means of the adaptive design procedure, train and save on forestFile the final ExtraTrees with all the rates and signs

        """
        print('\n------------------ Iterative Species Points Addition ------------------')
        # Create the training set by adaptive and iterative refinement
        for indexS in range(slf.numberOfTabVariables):
	        slf.addVariables(indexS, equidistantPoints)

        print('\n-------------------- Generating Final ExtraTrees ----------------------')
		# Create final dataset and ExtraTrees
        #Load trainingData and rates
        trainingData=np.loadtxt(slf.trainingFile,skiprows=1,delimiter=',',usecols=np.arange(slf.numberOfInputVariables))
        rates=np.loadtxt('rates.dat',skiprows=1,delimiter=',')
        
        if(len(rates.shape) == 1) :
            rates = rates.reshape(-1,1)
            
        plotData=np.c_[trainingData, rates]
        np.savetxt('plotDataFinal.dat',plotData,delimiter='\t',header=str(slf.headersInVar)) 
        
        slf.scalerout.fit(rates)
        rates=slf.scalerout.transform(rates)
        for k in range(slf.numberOfTabVariables) :
            if (slf.typevarTabVar[k] == 'log') : 
                rates[:,k] = np.log10(np.abs(rates)).ravel() 
            elif (slf.typevarTabVar[k] == 'lin') :
                rates[:,k] = np.abs(rates).ravel()

        #Construct trainingData for all species
        trainingData=np.c_[trainingData, rates]
        np.savetxt('trainFinal.dat',trainingData,delimiter=',',header=str(slf.headersInVar))     

        #RF training
        for k in range(10) :
            slf.trainExtraTressMIMO(trainingData)
            joblib.dump([slf.reg, slf.scalerout], slf.forestFile[0:slf.forestFile.index('.')]+'_'+str(k)+'.pkl',compress=1)

        #Save trees
        joblib.dump([slf.reg, slf.scalerout], slf.forestFile)

        #PredictionVSQuery
        ratesDI =np.loadtxt(slf.queryTabVar,skiprows=1,delimiter=',')
        if(len(ratesDI.shape) == 1) :
            ratesDI = ratesDI.reshape(-1,1)
                   
        queryData = np.loadtxt(slf.queryFile, skiprows=1, delimiter=',')
        
        if(len(queryData.shape) == 1) :
            queryData = queryData.reshape(-1,1)
        
        pred = slf.reg.predict(queryData)
        if(len(pred.shape) == 1) :
            pred = pred.reshape(-1,1)
        
        for k in range(slf.numberOfTabVariables) :
            if (slf.typevarTabVar[k] == 'log') : 
                pred[:,k] = 10**pred[:,k]
        
        pred = slf.scalerout.inverse_transform(pred)

        
        for index in range(slf.numberOfTabVariables):
            rateDI = ratesDI[:,index]
            rateRF = pred[:,index]  

            idx1 = np.where(np.abs(rateDI) >= slf.algorithmParams['errTh'])

            err = np.abs(rateDI[idx1]-rateRF[idx1])/np.abs(rateDI[idx1])

            print ('\n * Variables:', slf.headersTabVar[index])  
            print ('    * Av. Benchmark error   : ',np.average(err)*100.,'%')
            print ('    * Max. Benchmark error  : ',np.max(err)*100.,'%')             

        print('\n--------------------------- Procedure stats ---------------------------\n')
        if(slf.benchmark) :
            print('    * Benchmark error evolution:',slf.benchmarkErrorEv)
            slf.plotParity()
        print('    * Training data size evolution:',slf.trainingDataSize)
        
        print('\n--------------------------------- end ---------------------------------')
        
    def createBenchmarkDataset(slf, num_query):
        """Create a benchmark dataset by repetitively solving the full model

        Parameters
        ----------
            num_query : int
                Number of queries where the full model is solved
        """
        print('\n * Create Benchmark Dataset')
        # Build query dataset input
        query_set = []
        for i in range(slf.numberOfInputVariables) :
            if (slf.typevarInVar[i] == 'log') :
                pi = 10**np.random.uniform(np.log10(slf.min_range[i]),np.log10(slf.max_range[i]),size=num_query).reshape(-1,1)
            elif (slf.typevarInVar[i] == 'lin') :
                pi = np.random.uniform(slf.min_range[i],slf.max_range[i], size=num_query).reshape(-1,1)
            elif (slf.typevarInVar[i] == 'inv') :
                pi = 1./np.random.uniform(1./slf.max_range[i],1./slf.min_range[i], size=num_query).reshape(-1,1)
                
            if(len(query_set) == 0) :
                query_set = pi
            else :
                query_set = np.append(query_set, pi, axis = 1)
        
        # Compute benchmark
        query_val = getRate(query_set)
        
        for i in range(slf.numberOfInputVariables) :
            if (slf.typevarInVar[i] == 'log') :
                query_set[:,i] = np.log10(query_set[:,i])
            elif (slf.typevarInVar[i] == 'inv') :
                query_set[:,i] = 1./query_set[:,i]
        
        # Store query dataset input
        np.savetxt(slf.queryFile,query_set,delimiter=',',header=str(slf.headersInVar))   
        
        # Store values 
        np.savetxt(slf.queryTabVar,query_val,delimiter=',',header=str(slf.headersTabVar)) 
        
        
        
            
