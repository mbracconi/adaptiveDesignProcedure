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
    |   Copyright(C) 2019-2022 - M. Bracconi, M. Maestri               |
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
    \*----------------------------------------------------------------*/
"""

import numpy as np
import os
import sys
import shutil
import time
import joblib

import logging

import itertools

import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from boruta import BorutaPy

from packaging import version

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
except ImportError as e:
    pass

if(version.parse(sklearn.__version__) >= version.parse('1.1.1')):
    nfeat=1.0
else:
    nfeat='auto'


def predict(idata, forestFile) :

    reg,scal,inpVarParam,tabVarParam = joblib.load(forestFile)

    localData = []
    data = np.copy(idata)
    for numInput in range(inpVarParam['quantity']):
        # Properly scales the input variables
        if (inpVarParam['types'][numInput] == 'log'):
            data[:,numInput] = np.log10(data[:,numInput])
        elif (inpVarParam['types'][numInput] == 'inv'):
            data[:,numInput] = 1./(data[:,numInput])

    supp = reg.predict(data)
    if (len(supp.shape) == 1):
        supp = supp.reshape(-1,1)

    for numTab in range(tabVarParam['quantity']):
        if (tabVarParam['types'][numTab] == 'log'):
            supp[:,numTab] = 10**supp[:,numTab]

    pred = scal.inverse_transform(supp)

    return pred


class adaptiveDesignProcedure:
    def __init__(slf,  in_variables,
                       tab_variables,
                       approxFunction,
                       forestFile = 'ml_ExtraTrees.pkl',
                       trainingFile = 'train.dat',
                       forestParams = None,
                       algorithmParams = None,
                       queryFile = None,
                       queryTabVar = None,
                       benchmark = False,
                       plot = False,
                       debug = False,
                       useBoruta = True,
                       useBorutaWeak = True,
                       outputDir = 'adp.results',
                       randomState = None):

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
            forestParam : dictionary
                Structure with ExtraTrees parameters

                    forestParams =

                        {

                            'Ntree'       : 200, # number of decision trees in the forest

                            'tps'         : 1,   # number of record in terminal leaves

                            'fraction'    : 0.7, # fraction of data used to grow the forest

                        }

            algorithmParam : dictionary
                Structure with Adaptive Design Procedure parameters

                    algorithmParams =

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

            approxFunction : function
                Function which provides the values needed to generate the dataset and the Machine Lerning Model. The function takes one argument which is the matrix of the input file to be computed with shape (number of records, number of in variables) and return a matrix of the function values with shape (number of records, number of tabulation variables)
            forestFile : string
                Path to ExtraTress final trained algorithm
            trainingFile : string
                Path to training data file
            queryFile : string, optional
                Path to the query file (input variables)
            queryTabVar : string, optional
                Path to the query file (output variables)
            benchmark : bool, optional
                Calculation of benchmark error and plotting
            plot : bool, optional
                Plot parity and benchmark error during iterative procedure
            debug : bool, optional
                Print additional information during the iterations
            useBoruta : bool, optional
                Employ Boruta library to compute variable importance
            useBorutaWeak : bool, optional
                Consider also Boruta's weak variable in the training
        """

        # Initialize output directory
        slf.outputDir = outputDir
        if os.path.exists(slf.outputDir) :
            shutil.rmtree(slf.outputDir)
        os.mkdir(slf.outputDir)

        slf.randomState = None
        if randomState is not None:
            slf.randomState = randomState

        # Storing training data
        slf.trainingData = None

        # Default forest parameters
        slf.forestParams = {
                'Ntree'       : 200,
                'tps'         : 1,
                'fraction'    : 0.7,
            }

        # Default algorithm parameters
        slf.algorithmParams = {
                'dth'         : 0.1,     # thresold first derivative
                'd2th'        : 0.9,     # thresold second derivative
                'VIth'        : 0.15,    # thresold variable importance
                'errTh'       : 1e-6,    # thresold for MRE error evaluation (remove from MRE calculation record below this value)
                'OOBth'       : 0.05,    # termination criterium on OOBnorm
                'RADth'       : 10,      # termination criterium on Relative Approximation Error (RAD) [%]
                'maxTDSize'   : 5000,    # maximum allowed size of the training data
                'AbsOOBTh'    : 0.2,     # maximum variations between OOB for two different tabulation variables
            }

        # Initialize logger
        logger = logging.getLogger('output_adp')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        flh = logging.FileHandler(slf.outputDir+'/'+'output_adp.log', mode='w', encoding='UTF-8')
        flh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        logger.addHandler(flh)

        # Assign forest and training data file path
        slf.forestFile         = slf.outputDir+'/'+forestFile
        slf.forestFileForCFD   = slf.forestFile[0:slf.forestFile.rfind('.')]+'_forCFD.pkl'
        slf.trainingFile       = slf.outputDir+'/'+'tmp/'+trainingFile

        # Assign forest and algorithm params
        if forestParams is not None: slf.forestParams.update( forestParams )
        if algorithmParams is not None: slf.algorithmParams.update( algorithmParams )

        # Assign benchmark variables
        slf.benchmark          = benchmark
        slf.queryFile          = queryFile
        slf.queryTabVar        = queryTabVar
        slf.debug              = debug
        slf.plot               = plot
        slf.useBoruta          = useBoruta
        slf.useBorutaWeak      = useBorutaWeak

        # Define variables name, legnth and size
        slf.numberOfInputVariables   = len(in_variables)
        slf.numberOfSpecies          = slf.numberOfInputVariables
        slf.numberOfTabVariables     = len(tab_variables)

        slf.approxFunction = approxFunction

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
                logger.error('\nFATAL ERROR: variable type for ' + spec['name'] + ' not supported (log, inv, lin)\n')
                exit()
            if(spec['min'] <= 0) :
                logger.error('\nFATAL ERROR: minimum range for ' + spec['name'] + ' <= 0\n')
                exit()
            if(spec['num'] < 3) :
                logger.error('\nFATAL ERROR: minimum number of initial points for ' + spec['name'] + ' is 3\n')
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
                logger.error('\nFATAL ERROR: variable type for ' + spec['name'] + ' not supported (log, lin)\n')
                exit()

        # Define lists of procedure results
        slf.benchmarkErrorEv    = [] # [iter 0 CO, iter 1 CO, ..., iter 0 H2O, iter 1 H2O, ... ]
        slf.benchmarkMaxErrorEv = []
        slf.trainingDataSize    = []
        slf.normOOB             = [] # [ [OOB #0, OOB #1, .... OOB #N], [OOB #0, OOB #1, .... OOB #N], [OOB #0, OOB #1, .... OOB #N], ...] # list of lists
        slf.OOBScoreEv          = []
        slf.RAD                 = [] # [ [RAD #1, .... RAD #N], [RAD #1, .... RAD #N], [RAD #1, .... RAD #N], ...] # list of lists

        # Construct and set Random Forest
        slf.reg = ExtraTreesRegressor(random_state=slf.randomState, n_estimators=slf.forestParams['Ntree'], max_features=nfeat, bootstrap = True, oob_score = True, max_samples = slf.forestParams['fraction'], min_samples_leaf=slf.forestParams['tps'])

        # Construct and set the scaler
        slf.scalerout = MinMaxScaler(feature_range=(1e-6,1))

        # For plotting purpose
        if(slf.plot) :
            if( 'matplotlib' in sys.modules ):
                slf.fig = plt.figure(figsize=(10,4))
                slf.ax = slf.fig.subplots(nrows=1, ncols=2)
                plt.subplots_adjust(wspace=0.5,bottom = 0.2,right=0.85)
                # Panel - 1: Average benchmakrj error evolution wrt training points size
                slf.ax[0].set_ylabel(r'Average benchmark error [%]')
                slf.ax[0].set_xlabel(r'Number of training points [-]')
                # Panel - 2: Parity plot
                slf.ax[1].set_ylabel(r'ET [kmol $\mathregular{m^{-2} s^{-1}}$]')
                slf.ax[1].set_xlabel(r'Full Model [kmol $\mathregular{m^{-2} s^{-1}}$]')
                #slf.ax[1].ticklabel_format(axis = 'both', style = 'sci', useOffset=False)
            else:
                print('WARNING: Because matlibplot is not installed, the option ``plot=True`` will be disregarded.')
                print('         Consider to install matplotlib to visualize the results!')
                print('         $ pip install matplotlib')

        # Create supporting folders
        if os.path.exists(slf.outputDir+'/'+'tmp') :
            shutil.rmtree(slf.outputDir+'/'+'tmp')
        os.mkdir(slf.outputDir+'/'+'tmp')
        if os.path.exists(slf.outputDir+'/'+'figures') :
            shutil.rmtree(slf.outputDir+'/'+'figures')
        os.mkdir(slf.outputDir+'/'+'figures')

        # Printing
        logger.info('\n------ Adaptive generation of Training Data for Machine Learning ------')
        logger.info('\nInput parameters:')
        logger.info('  * Forest file: ' + slf.forestFile)
        logger.info('  * Training file: ' + slf.trainingFile)
        logger.info('  * Figure path: ' + slf.outputDir+'/'+'figures')
        logger.info('  * Plotting enabled: ' + str(slf.plot))
        logger.info('  * Boruta as feature selector: ' + str(slf.useBoruta))
        if (slf.useBoruta):
            logger.info('  * Use Weak Support Var in Boruta:' + str(slf.useBorutaWeak))
        logger.info('\n  * Forest parameters:')
        logger.info('    {')
        logger.info('\n'.join('        {}: {}'.format(k, v) for k, v in slf.forestParams.items()))
        logger.info('    }')
        logger.info('\n  * Algorithm parameters:')
        logger.info('    {')
        logger.info('\n'.join('        {}: {}'.format(k, v) for k, v in slf.algorithmParams.items()))
        logger.info('    }')
        logger.info('\n\n  * Variables information:')
        for t in in_variables :
            logger.info('    {')
            logger.info('\n'.join('        {}: {}'.format(k, v) for k, v in t.items()))
            logger.info('    }')

        logger.info('\n\n  * Tabulation information:')
        for t in tab_variables :
            logger.info('    {')
            logger.info('\n'.join('        {}: {}'.format(k, v) for k, v in t.items()))
            logger.info('    }')


    def trainExtraTressMISO(slf,trainingData) :
        """Train ExtraTrees algorithm for absolute value of the rate and for the sign of a single variables

        Parameters
        ----------
            trainingData : np.array
                Matrix consisting of the training data: first numberOfVariables columns are the descriptors followed by the absolute value rate and by the sign
        """
        # Load training data
        slf.reg.set_params(random_state=slf.randomState)

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
        slf.reg.set_params(random_state=slf.randomState, bootstrap=True, max_samples = 0.95, oob_score = False)

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
            newForestQuery = slf.scalerout.inverse_transform((10**slf.reg.predict(queryDataVal)).reshape(-1,1))
        elif (typevar == 'lin') :
            newForestQuery = slf.scalerout.inverse_transform((slf.reg.predict(queryDataVal)).reshape(-1,1))

        # Select rate larger than th
        idxA = np.where(np.abs(oldForestQuery) >= slf.algorithmParams['errTh'])
        # Compute approximation error in terms of MRE
        errA = np.abs(oldForestQuery[idxA]-newForestQuery[idxA])/np.abs(oldForestQuery[idxA])
        avErrA = np.average(errA)*100.
        return avErrA

    def benchmarkError(slf,indexTabVariable,typevar, count, msle = False, relative = True) :
        """Compute the benchmark error used for analysis of accuracy of the procedure by evaluating the Mean Squared Logarithmic Error MSLE (default) or Mean Relative Error MRE: :math:`MSLE=1/N \sum_{i=1}^{n_{query}}{(\log(y_i+1)-\log(\widehat{y}_i+1))^2}` \\ :math:`MRE=1/N \sum_{i=1}^{n_{query}}{|y_i-\widehat{y}_i|/|\widehat{y}_i|}`

        Parameters
        ----------
            indexTabVariable : int
                Index of the variables considered
            typevar : string
                Define how to treat the tabulated variable (either 'log' or 'lin')
            msle : bool, optional
                Matrix consisting of the predictions of the queryDataVal obtained with the ExtraTress obtained ad the previous iteration
            relative : bool, optional
                The benchmark error will be calculated with its Mean Relative Error by default. This may not be appropriate to values close to zero as tabulation variable, so it can be opted out to adopt Mean Absolute Error instead.

        Return
        ----------
            err : np.array
                Vector of the error
        """
        ratesDI = np.loadtxt(slf.outputDir+'/'+slf.queryTabVar,skiprows=1,delimiter=',',usecols=(indexTabVariable))
        queryData = np.loadtxt(slf.outputDir+'/'+slf.queryFile, skiprows=1, delimiter=',')

        if(len(queryData.shape) == 1) :
            queryData = queryData.reshape(-1,1)

        if(typevar == 'log') :
            pred = 10**slf.reg.predict(queryData)
        elif (typevar == 'lin') :
            pred = slf.reg.predict(queryData)

        rates_real_scaled = ratesDI #slf.scalerout.transform(ratesDI.reshape(-1,1)).ravel()
        rates_etrt_scaled = slf.scalerout.inverse_transform(pred.reshape(-1,1)).ravel() #pred # slf.scalerout.transform(pred.reshape(-1,1))

        np.savetxt(slf.outputDir+'/'+'bench_'+str(count)+'_'+slf.headersTabVar[indexTabVariable]+'.dat',np.c_[queryData,rates_etrt_scaled],header=str(slf.headersInVar),comments='#')

        idx1 = np.where(np.abs(rates_real_scaled) >= slf.algorithmParams['errTh'])
        if msle :
            err = (np.log10(np.abs(rates_real_scaled[idx1]+1))-np.log10(np.abs(rates_etrt_scaled[idx1]+1)))**(2)
        elif relative:
            # RELATIVE ERROR
            err = np.abs(rates_real_scaled[idx1]-rates_etrt_scaled[idx1])/np.abs(np.array(rates_real_scaled[idx1]))
        else:
            # NORMALIZED ABSOLUTE ERROR
            err = np.abs(rates_real_scaled[idx1]-rates_etrt_scaled[idx1])/np.abs(np.average(rates_real_scaled[idx1]))

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
        if 'matplotlib' not in sys.modules:
            return

        ratesDI = np.loadtxt(slf.outputDir+'/'+slf.queryTabVar,skiprows=1,delimiter=',',usecols=(indexTabVariable))
        queryData = np.loadtxt(slf.outputDir+'/'+slf.queryFile, skiprows=1, delimiter=',')

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
        ticks = np.linspace(np.min(pline),np.max(pline),4)
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
            plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

        slf.ax[1].plot(ratesDI[idx1],pred[idx1], 'o', markersize=3,label = slf.headersTabVar[indexTabVariable] + ' - #' + str(iterC))
        slf.ax[1].set_ylabel(slf.headersTabVar[indexTabVariable] + r'$\mathregular{_{ET}}$')
        slf.ax[1].set_xlabel(slf.headersTabVar[indexTabVariable] + r'$\mathregular{_{MD}}$')
        plt.sca(slf.ax[1])
        slf.ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.draw()
        plt.pause(0.001)
        plt.savefig(slf.outputDir+'/'+'figures/trend_'+slf.headersTabVar[indexTabVariable]+'_'+str(iterC)+'.png', dpi=300)

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

        if 'matplotlib' not in sys.modules:
            return

        ratesDI = np.loadtxt(slf.outputDir+'/'+slf.queryTabVar,skiprows=1,delimiter=',')
        queryData = np.loadtxt(slf.outputDir+'/'+slf.queryFile, skiprows=1, delimiter=',')

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
            ticks = np.linspace(np.min(pline),np.max(pline),4)
            plt.xlim(np.min(pline),np.max(pline))
            plt.ylim(np.min(pline),np.max(pline))


            plt.plot(ratesDI[:,k],pred[:,k], 'o', markersize=3)
            plt.ylabel(slf.headersTabVar[k] + r'$\mathregular{_{ET}}$')
            plt.xlabel(slf.headersTabVar[k] + r'$\mathregular{_{full model}}$')
            plt.plot(pline,pline,'k-',linewidth=1)
            plt.plot(pline,pline*0.7,'k--',linewidth=0.5)
            plt.plot(pline,pline*1.3,'k--',linewidth=0.5)
            plt.legend([slf.headersTabVar[k]], loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

            plt.tight_layout()
            plt.savefig(slf.outputDir+'/'+'figures/parity_'+slf.headersTabVar[k]+'.png', dpi=300)

     ## Function which select the position for the newly added points at each iteration
    def findPoints(slf, trainData, pointsToAdd, indexTabVariable):
        """Evaluate the position of the new points based on the discrete function gradient

        Parameters
        ----------
            trainData : np.array
                Matrix consisting of the training data variables and function rate
            pointsToAdd : np.array
                Variables considered for the addition defined based on the variable importance
            indexTabVariable : int
                Index of the tabulation variables considered

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
            # trer2 = []

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

                    # Treatment of input var
                    if (slf.typevarInVar[j] == 'lin') :
                        distance = p_unique[i+1]-p_unique[i]
                    elif (slf.typevarInVar[j] == 'log') :
                        distance = np.log10(p_unique[i+1])-np.log10(p_unique[i])
                    elif (slf.typevarInVar[j] == 'inv') :
                        distance = 1./p_unique[i+1]-1./p_unique[i]

                    for k in range(point_len) :

                        # Treatment of output var
                        if (slf.typevarTabVar[indexTabVariable] == 'lin'):
                            valFi = p_temp_fi[k,slf.numberOfInputVariables]
                            valSi = p_temp_si[k,slf.numberOfInputVariables]
                        elif (slf.typevarTabVar[indexTabVariable] == 'log'):
                            valFi = 10**p_temp_fi[k,slf.numberOfInputVariables]
                            valSi = 10**p_temp_si[k,slf.numberOfInputVariables]

                        # Reverse scaling of tabulation variables
                        valFi = slf.scalerout.inverse_transform(valFi.reshape(1,-1))
                        valSi = slf.scalerout.inverse_transform(valSi.reshape(1,-1))

                        derj = np.abs((valSi - valFi)/distance)

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

                        # Treatment of input var
                        if (slf.typevarInVar[j] == 'lin') :
                            dx1 = p_unique[i]-p_unique[i-1]
                            dx2 = p_unique[i+1]-p_unique[i]
                            dx3 = p_unique[i+1]-p_unique[i-1]
                        elif (slf.typevarInVar[j] == 'log') :
                            dx1 = np.log10(p_unique[i])-np.log10(p_unique[i-1])
                            dx2 = np.log10(p_unique[i+1])-np.log10(p_unique[i])
                            dx3 = np.log10(p_unique[i+1])-np.log10(p_unique[i-1])
                        elif (slf.typevarInVar[j] == 'inv') :
                            dx1 = 1./p_unique[i]-1./p_unique[i-1]
                            dx2 = 1./p_unique[i+1]-1./p_unique[i]
                            dx3 = 1./p_unique[i+1]-1./p_unique[i-1]

                    else :
                        p_temp_im = trainData[trainData[:,j] == p_unique[i]]
                        p_temp_i = trainData[trainData[:,j] == p_unique[i+1]]
                        p_temp_ip = trainData[trainData[:,j] == p_unique[i+2]]

                        # Treatment of input var
                        if (slf.typevarInVar[j] == 'lin') :
                            dx1 = p_unique[i+1]-p_unique[i]
                            dx2 = p_unique[i+2]-p_unique[i+1]
                            dx3 = p_unique[i+2]-p_unique[i]
                        elif (slf.typevarInVar[j] == 'log') :
                            dx1 = np.log10(p_unique[i+1])-np.log10(p_unique[i])
                            dx2 = np.log10(p_unique[i+2])-np.log10(p_unique[i+1])
                            dx3 = np.log10(p_unique[i+2])-np.log10(p_unique[i])
                        elif (slf.typevarInVar[j] == 'inv') :
                            dx1 = 1./p_unique[i+1]-1./p_unique[i]
                            dx2 = 1./p_unique[i+2]-1./p_unique[i+1]
                            dx3 = 1./p_unique[i+2]-1./p_unique[i]

                    #locapp = []
                    locappd2 = []
                    point_len = len(p_temp_i)
                    for k in range(point_len) :
                        # Treatment of output var
                        if (slf.typevarTabVar[indexTabVariable] == 'lin'):
                            a = p_temp_ip[k,slf.numberOfInputVariables]
                            b = p_temp_i[k,slf.numberOfInputVariables]
                            c = p_temp_im[k,slf.numberOfInputVariables]
                        elif (slf.typevarTabVar[indexTabVariable] == 'log'):
                            a = 10**p_temp_ip[k,slf.numberOfInputVariables]
                            b = 10**p_temp_i[k,slf.numberOfInputVariables]
                            c = 10**p_temp_im[k,slf.numberOfInputVariables]

                         # Reverse scaling of tabulation variables
                        a = slf.scalerout.inverse_transform(a.reshape(1,-1))
                        b = slf.scalerout.inverse_transform(b.reshape(1,-1))
                        c = slf.scalerout.inverse_transform(c.reshape(1,-1))


                        der2j = (a*dx1 + c*dx2 - b*dx3)/(0.5*dx1*dx2*dx3);

                        locappd2.append(np.abs(der2j))

                    trer.append(np.max(locappd2))

                # ------------------------------------------------------------------------------------------------------------------
            else :
                der = 0
                trer = 0

            firstDerivative.append(np.abs(der))
            secondDerivative.append(np.abs(trer))

        maxFirst = 0
        maxSecond = 0
        for j in range(slf.numberOfInputVariables):
            if maxFirst < np.max(firstDerivative[j]):
                maxFirst = np.max(firstDerivative[j])
            if maxSecond < np.max(secondDerivative[j]):
                maxSecond = np.max(secondDerivative[j])

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
                countDer1 = 0
                countDer2 = 0

                for i in range(der.shape[0]) :
                    if  np.abs(der[i]/maxFirst) > slf.algorithmParams['dth'] :
                        poi.append(add_p[i])
                        countDer1 += 1
                    elif np.abs(trer[i]/maxSecond) > slf.algorithmParams['d2th']  :
                        poi.append(add_p[i])
                        countDer2 += 1

                if(slf.debug):
                    logger.debug ('      --------------------------- ')
                    logger.debug ('      Point addition in direction: ',slf.headersInVar[j])
                    logger.debug ('      Points added through 1st der: ',countDer1)
                    logger.debug ('      Points added through 2nd der: ',countDer2)
                    logger.debug ('      ---------------------------')

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
        logger = logging.getLogger('output_adp')
        logger.info('\n  * Tabulation Variables: ' + slf.headersTabVar[indexTabVariable])

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

            pointsPerSpec = [equidistantPoints]*(slf.numberOfInputVariables)

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
                logger.info ('\n\n\n * Equidistant points ')

                iterate = False

                logger.info ('   --------------------------- ')
                logger.info ('   Points per species: ' + ' '.join( str(e) for e in pointsPerSpec))
                logger.info ('   ---------------------------')
            else :
                logger.info ('\n    * Iteration: ' + str(count))
                if count > 0:

                    biggestImp  = np.max(importances)

                    pointsToAdd = np.zeros(slf.numberOfInputVariables)
                    locAdd = np.zeros(slf.numberOfInputVariables, dtype=bool)

                    if (slf.useBoruta):
                        # In specific scenarios, Boruta might select no features
                        # especially for datasets with low number of samples, where
                        # the randomized shadow feature might perform better than
                        # the actual features. In case Boruta usage is specified
                        # to TRUE and no features are added, the ADP will
                        # automatically force the usage of VIth for feature
                        # selection in this iteration.
                        selectedFeatures = False
                        # Applying Boruta Filter
                        for t in range(slf.numberOfInputVariables):
                            if (boruta.support_[t] or (boruta.support_weak_[t] and slf.useBorutaWeak)):
                                locAdd[t] = True
                                selectedFeatures = True

                        # If no features are selected, force VIth
                        if not selectedFeatures:
                            locAdd = np.where(importances/biggestImp > slf.algorithmParams['VIth'])
                    else:
                        # Using variable importance
                        locAdd = np.where(importances/biggestImp > slf.algorithmParams['VIth'])

                    pointsToAdd[locAdd] = 1

                    logger.info ('      Normalized variable importance: ' + ' '.join( str(e) for e in (importances/biggestImp)))

                    if (slf.useBoruta):
                        logger.info ('\n      Using Boruta for feature selection.')
                        logger.info ('      Boruta Support: ' + ' '.join(header for index,header in enumerate(slf.headersInVar) if boruta.support_[index]))
                        logger.info ('      Boruta Support Weak: ' + ' '.join(header for index,header in enumerate(slf.headersInVar) if boruta.support_weak_[index]))
                        if not selectedFeatures:
                            logger.info ('      Boruta did not select any feature for improvement in this iteration.')
                            logger.info ('      Forcing feature selection with Variable Importance threshold in this iteration.')
                        logger.info('\n')

                    newPress = slf.findPoints(trainingData[:,:], pointsToAdd, indexTabVariable)
                    pointsPerSpec = [len(j) for j in newPress]

                    p_var = []
                    for i in range(slf.numberOfInputVariables) :
                        p_var.append(np.sort(newPress[i]))

                    logger.info ('      --------------------------- ')
                    logger.info ('      Points per species :' + ' '.join( str(e) for e in pointsPerSpec))
                    logger.info ('      ---------------------------')
                else :
                    logger.info ('      --------------------------- ')
                    logger.info ('      Points per species :' + ' '.join( str(e) for e in pointsPerSpec))
                    logger.info ('      ---------------------------')

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

            logger.info ('      Total number of points: ' + str(trainingData.shape[0]))

            # Compute new training points
            ratesAll = []
            rates = []
            funcEvalTime = time.time()
            if count > 0 :
                logger.info ('      New points            : ' + str(trainingData.shape[0]-trD.shape[0]) + '\n')
                added   = []
                addedra = []

                ratesAll = []
                cont = 0

                # if number of processes is not specified, it uses the number of core
                funcEvalTime = time.time()

                # Get rates
                ratesAll = np.array(slf.approxFunction(trainingData))
                if(ratesAll.shape[1] != slf.numberOfTabVariables) :
                    logger.error ('\nFATAL ERROR: shape of tabulation variable matrix is wrong, obtained: ' + ratesAll.shape[1] + ' expected: ' + slf.numberOfTabVariables)
                    exit()
                rates = ratesAll[:,indexTabVariable]
                rates_plot = rates

                logger.info ('      Function solved in ' + str(time.time()-funcEvalTime))

                for kk, gg in enumerate(trainingData) :
                    # Searches for the indexes where the points existed previously to point addition
                    index = np.where(np.all((trD[:,0:slf.numberOfInputVariables] == gg),axis=1) == True)[0]

                    # If the point is new:
                    if len(index) == 0 :
                        # Register the rate of the new point
                        addedra.append(rates_plot[kk])
                        # Register the point that was added
                        added.append(gg)
                        # Counts new point
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
                    oldForestQuery = slf.scalerout.inverse_transform(10**slf.reg.predict(queryDataVal).reshape(-1,1))
                elif (slf.typevarTabVar[indexTabVariable] == 'lin') :
                    oldForestQuery = slf.scalerout.inverse_transform(slf.reg.predict(queryDataVal).reshape(-1,1))

                # Rescaling the data
                #slf.scalerout.fit(rates.reshape(-1,1))
                rates = slf.scalerout.transform(rates.reshape(-1,1))
                # Store scaled value of rates
                if (slf.typevarTabVar[indexTabVariable] == 'log') :
                    rates = np.log10(rates).ravel()
                elif (slf.typevarTabVar[indexTabVariable] == 'lin') :
                    rates = rates.ravel()


                addedSup = queryDataVal
                addedra = np.array(addedra)

                addedra = np.expand_dims(addedra,1)
                addedSup = np.append(addedSup, addedra, axis=1)

                # Print on file the values of the new training points (just for nice pictures)
                np.savetxt(slf.outputDir+'/'+'train_'+str(count)+'_'+slf.headersTabVar[indexTabVariable]+'.dat',addedSup,header=str(slf.headersInVar),comments='#')

            else:
                if indexTabVariable != 0 :
                    rates=np.loadtxt(slf.outputDir+'/'+'rates.dat',skiprows=1,delimiter=',',usecols=(indexTabVariable))
                    ratesAll = np.loadtxt(slf.outputDir+'/'+'rates.dat',skiprows=1,delimiter=',')
                    rates_plot = rates

                    slf.scalerout.fit(rates.reshape(-1,1))
                    rates=slf.scalerout.transform(rates.reshape(-1,1))
                    if (slf.typevarTabVar[indexTabVariable] == 'log') :
                        rates = np.log10(rates).ravel()
                    elif (slf.typevarTabVar[indexTabVariable] == 'lin') :
                        rates = rates.ravel()

                    logger.info ('\n      Function loaded in ' + str(time.time()-funcEvalTime))

                else :
                    # First species
                    ratesAll = slf.approxFunction(trainingData)

                    if(ratesAll.shape[1] != slf.numberOfTabVariables) :
                        logger.error ('\nFATAL ERROR: shape of tabulation variable matrix is wrong, obtained: ' + ratesAll.shape[1] + ' expected: ' + slf.numberOfTabVariables)
                        exit()

                    ratesAll = np.array(ratesAll)

                    rates = ratesAll[:,indexTabVariable]
                    rates_plot = rates

                    # Finds out the min and max value to scale
                    slf.scalerout.fit(rates.reshape(-1,1))
                    # Transform and scale down
                    rates=slf.scalerout.transform(rates.reshape(-1,1))
                    if (slf.typevarTabVar[indexTabVariable] == 'log') :
                        np.savetxt(slf.outputDir+'/'+'train_'+str(count)+'_'+slf.headersTabVar[indexTabVariable]+'.dat',np.c_[trainingData,slf.scalerout.inverse_transform(rates.reshape(-1,1)).ravel()],header=str(slf.headersInVar),comments='#')
                        rates = np.log10(rates).ravel()
                    elif (slf.typevarTabVar[indexTabVariable] == 'lin') :
                        np.savetxt(slf.outputDir+'/'+'train_'+str(count)+'_'+slf.headersTabVar[indexTabVariable]+'.dat',np.c_[trainingData,slf.scalerout.inverse_transform(rates.reshape(-1,1)).ravel()],header=str(slf.headersInVar),comments='#')
                        rates = rates.ravel()
                    logger.info ('\n      Function solved in ' + str(time.time()-funcEvalTime))


            # Create training data
            rates=np.expand_dims(rates,1)
            rates_plot=np.expand_dims(rates_plot,1)

            trainingDataRaw = trainingData.copy()
            trainingData = np.append(trainingData, rates, axis=1)

            # Create training data for ExtraTrees
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

            # Write training file(for both training datasets)
            np.savetxt(slf.trainingFile,trainingDataSupp,delimiter=',',header=str(slf.headersInVar))

            # Transforming back the rates
            pred = rates.copy()
            if(len(pred.shape) == 1) :
                pred = pred.reshape(-1,1)

            for k in range(slf.numberOfTabVariables) :
                if (slf.typevarTabVar[k] == 'log') :
                    pred[:,k] = 10**pred[:,k]

            pred = slf.scalerout.inverse_transform(pred)
            pred = np.append(trainingDataRaw, pred, axis=1)
            slf.trainingData = pred

            if (count > 0 or equidistantPoints) :
                np.savetxt(slf.outputDir+'/'+'rates.dat',ratesAll,delimiter=',',header=str(slf.headersTabVar))

            # Ranger grow forest (done 10 times to get a less biased OOB especially in the case of small dataset)
            OOB = []
            OOBScore = []
            # Ranger grow forest (done 10 times to get a less biased OOB especially in the case of small dataset)

            for k in range(10) :
                slf.trainExtraTressMISO(trainingDataSupp)

                OOB.append(mean_squared_error(trainingDataSupp[:,-1],slf.reg.oob_prediction_))
                OOBScore.append(slf.reg.oob_score_)

                # Store Variable importance
                imp[k,:] = np.array(permutation_importance(slf.reg,trainingDataSupp[:,0:slf.numberOfInputVariables],trainingDataSupp[:,-1],n_repeats=10,scoring='r2').importances_mean) #np.array(slf.reg.feature_importances_)

            OOB = np.mean(np.array(OOB))
            OOBScore = np.mean(OOBScore)

            # Setting Boruta up, by using the same estimator from procedure
            # We must use a new estimator similar to the one used by the procedure for boruta
            # The issue is that if we use the same estimator object, Boruta will create
            # some new "Shadow Features" if there aren't enough and by requiring new
            # features, the reg object from the procedure will be affected, which leads to
            # errors in predictions later on.
            if (slf.useBoruta):
                boruta = BorutaPy(
                   # estimator = slf.reg,
                   estimator = ExtraTreesRegressor(random_state=slf.randomState, n_estimators=slf.forestParams['Ntree'], max_features=nfeat, bootstrap = True, oob_score = True, max_samples = slf.forestParams['fraction'], min_samples_leaf=slf.forestParams['tps']),
                   n_estimators = 'auto',
                   max_iter = 100 # number of trials to perform
                )

                # Boruta code for Variable Importances
                boruta.fit(np.array(trainingDataSupp[:,0:slf.numberOfInputVariables]), np.array(trainingDataSupp[:,-1]))

            importances=np.mean(imp,axis=0)

            joblib.dump([slf.reg, slf.scalerout], slf.outputDir+'/'+'tmp/rf_'+slf.headersTabVar[indexTabVariable]+'_count'+str(count)+'.pkl')

            # Compute the approximation error
            # To avoid benchmarking, the procedure computes the approximation error iteration per iteration
            # This error tells the quality of the improvement given by the newly added points. If this error is small
            # it means that the new points are useless hence the iterative procedure reached the end
            if count > 0 :
                errA = slf.approximationError(queryDataVal,oldForestQuery,slf.typevarTabVar[indexTabVariable])
                RADspecies.append(errA)  #Add RAD to storage list

            # Add OOB storage list
            OOBspecies.append(OOB)
            slf.OOBScoreEv.append(OOBScore)
            logger.info ('\n      Approximation quality:')
            logger.info ('          Out-Of-Bag error     : ' + str(OOB))
            logger.info ('          Out-Of-Bag score     : ' + str(OOBScore))
            if count > 0 :
                logger.info ('          Iterative approx err : ' + str(errA) + ' %')

            # Load rates from query file as real value
            #ratesDI = np.loadtxt(slf.queryTabVar, skiprows=1,delimiter=',',usecols=(indexTabVariable))

            slf.trainingDataSize.append(trainingData.shape[0])

            if (slf.benchmark) :
                #ratesDI = np.loadtxt(slf.queryTabVar, skiprows=1,delimiter=',',usecols=(indexTabVariable))
                #errMSLE = slf.benchmarkError(indexTabVariable,slf.typevarTabVar[indexTabVariable],count,msle=True, relative = False)
                #errMRE = slf.benchmarkError(indexTabVariable,slf.typevarTabVar[indexTabVariable],count,msle=False, relative = True)
                errMAE = slf.benchmarkError(indexTabVariable,slf.typevarTabVar[indexTabVariable],count,msle=False, relative = False)
                logger.info ('\n      Benchmark calculations:')
                logger.info ('      Currently saving MAE error')
                logger.info ('          Av. Benchmark error (MAE) : ' + str(np.average(errMAE)*100.) + ' %')
                logger.info ('          Max. Benchmark error (MAE): ' + str(np.max(errMAE)*100.) +' %')

                slf.benchmarkErrorEv.append(np.average(errMAE)*100.)
                slf.benchmarkMaxErrorEv.append(np.max(errMAE)*100.)
                if(slf.plot) :
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
                        logger.info ('\n      Maximum size of training data reached in ' + str(count+1) + ' iterations')
                        if slf.debug :
                            logger.debug('          OOB Evolution: ' + ' '.join( str(e) for e in  OOBspecies))
                            logger.debug('          RAD Evolution: ' + ' '.join( str(e) for e in  RADspecies))

                    else :
                        iterate = False
                        slf.normOOB.append(OOBspecies)
                        slf.RAD.append(RADspecies)
                        logger.info ('\n      Accuracy constraints reached in ' + str(count+1) + ' iterations')
                        if slf.debug :
                            logger.debug('          OOB Evolution: ' + ' '.join( str(e) for e in  OOBspecies))
                            logger.debug('          RAD Evolution: ' + ' '.join( str(e) for e in  RADspecies))

                else :
                    if count == 0:
                        maxPreviousOOB = 0.
                        for k in range(indexTabVariable) :
                            maxPreviousOOB = max(maxPreviousOOB,slf.normOOB[k][-1])
                        if slf.debug :
                            logger.debug('          MaxOld OOB: '+ str(maxPreviousOOB))

                        if OOB < slf.algorithmParams['OOBth']:
                            iterate = False
                            slf.normOOB.append(OOBspecies)
                            slf.RAD.append([])
                            logger.info ('\n      Accuracy constraints reached in ' + str(count+1) + ' iterations')
                            if slf.debug :
                                logger.debug('          OOB Evolution: ' + ' '.join( str(e) for e in  OOBspecies))
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
                            logger.info ('\n      Maximum size of training data reached in ' + str(count+1) + ' iterations')
                            if slf.debug :
                                logger.debug('          OOB Evolution: ' + ' '.join( str(e) for e in  OOBspecies))
                                logger.debug('          RAD Evolution: ' + ' '.join( str(e) for e in  RADspecies))
                        else :
                            iterate = False
                            slf.normOOB.append(OOBspecies)
                            slf.RAD.append(RADspecies)
                            logger.info ('\n      Accuracy constraints reached in ' + str(count+1) + ' iterations')
                            if slf.debug :
                                logger.debug('          OOB Evolution: ' + ' '.join( str(e) for e in  OOBspecies))
                                logger.debug('          RAD Evolution: ' + ' '.join( str(e) for e in  RADspecies))


    def createTrainingDataAndML(slf, equidistantPoints = 0):
        """Generate the training set by means of the adaptive design procedure, train and save on forestFile the final ExtraTrees with all the rates and signs. Final ExtraTrees is saved according to forestFile variable in joblib format

        """
        logger = logging.getLogger('output_adp')
        logger.info('\n------------------ Iterative Species Points Addition ------------------')
        # Create the training set by adaptive and iterative refinement
        for indexS in range(slf.numberOfTabVariables):
            slf.addVariables(indexS, equidistantPoints)

        logger.info('\n-------------------- Generating Final ExtraTrees ----------------------')
        # Create final dataset and ExtraTrees
        # Load trainingData and rates
        trainingData=np.loadtxt(slf.trainingFile,skiprows=1,delimiter=',',usecols=np.arange(slf.numberOfInputVariables))
        rates=np.loadtxt(slf.outputDir+'/'+'rates.dat',skiprows=1,delimiter=',')

        if(len(rates.shape) == 1) :
            rates = rates.reshape(-1,1)

        plotData=np.c_[trainingData, rates]
        np.savetxt(slf.outputDir+'/'+'plotDataFinal.dat',plotData,delimiter='    ',header=str(slf.headersInVar))

        slf.scalerout.fit(rates)
        rates=slf.scalerout.transform(rates)
        for k in range(slf.numberOfTabVariables) :
            if (slf.typevarTabVar[k] == 'log') :
                rates[:,k] = np.log10(np.abs(rates[:,k])).ravel()
            elif (slf.typevarTabVar[k] == 'lin') :
                rates[:,k] = np.abs(rates[:,k]).ravel()

        #Construct trainingData for all species
        trainingData=np.c_[trainingData, rates]
        np.savetxt(slf.outputDir+'/'+'trainFinal.dat',trainingData,delimiter=',',header=str(slf.headersInVar))

        #RF training
        for k in range(10) :
            slf.trainExtraTressMIMO(trainingData)
            joblib.dump([slf.reg, slf.scalerout], slf.forestFile[0:slf.forestFile.rfind('.')]+'_'+str(k)+'.pkl',compress=1)

        #Save trees
        joblib.dump([slf.reg, slf.scalerout], slf.forestFile)

        #PredictionVSQuery
        if(slf.benchmark) :
            ratesDI =np.loadtxt(slf.outputDir+'/'+slf.queryTabVar,skiprows=1,delimiter=',')
            if(len(ratesDI.shape) == 1) :
                ratesDI = ratesDI.reshape(-1,1)

            queryData = np.loadtxt(slf.outputDir+'/'+slf.queryFile, skiprows=1, delimiter=',')

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

                err = np.abs(rateDI[idx1]-rateRF[idx1])/np.abs(np.average(rateDI[idx1]))

                logger.info ('\n * Variables: ' + slf.headersTabVar[index])
                logger.info ('    * Av. Benchmark error   : ' + str(np.average(err)*100.) + ' %')
                logger.info ('    * Max. Benchmark error  : ' + str(np.max(err)*100.) + ' %')


        # Produces a PKL with only the variable types to allow a more general approach to the
        # interface code with CFD
        inpVarParam = {
            'quantity': slf.numberOfInputVariables,
            'names': slf.headersInVar,
            'types': slf.typevarInVar,
            'min_range': slf.min_range,
            'max_range': slf.max_range,
            }

        tabVarParam = {
            'quantity': slf.numberOfTabVariables,
            'names': slf.headersTabVar,
            'types': slf.typevarTabVar
            }

        joblib.dump([slf.reg, slf.scalerout, inpVarParam, tabVarParam], slf.forestFileForCFD)
        logger.info('\n----------------------- Model for CFD generated -----------------------\n')

        logger.info('\n--------------------------- Procedure stats ---------------------------\n')

        if(slf.benchmark) :
            logger.info('    * Benchmark error evolution: ' + ' '.join( str(e) for e in slf.benchmarkErrorEv))
            if(slf.plot):
                slf.plotParity()
        logger.info('    * Training data size evolution: ' + ' '.join( str(e) for e in slf.trainingDataSize))
        logger.info('\n--------------------------------- end ---------------------------------')

    def createBenchmarkDataset(slf, num_query):
        """Create a benchmark dataset by repetitively solving the full model

        Parameters
        ----------
            num_query : int
                Number of queries where the full model is solved
        """
        logger = logging.getLogger('output_adp')
        logger.info('\n * Create Benchmark Dataset')
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
        query_val = slf.approxFunction(query_set)

        for i in range(slf.numberOfInputVariables) :
            if (slf.typevarInVar[i] == 'log') :
                query_set[:,i] = np.log10(query_set[:,i])
            elif (slf.typevarInVar[i] == 'inv') :
                query_set[:,i] = 1./query_set[:,i]

        # Store query dataset input
        np.savetxt(slf.outputDir+'/'+slf.queryFile,query_set,delimiter=',',header=str(slf.headersInVar))

        # Store values
        np.savetxt(slf.outputDir+'/'+slf.queryTabVar,query_val,delimiter=',',header=str(slf.headersTabVar))

    def predict( slf, idata ):
        return predict( idata, slf.forestFileForCFD )
