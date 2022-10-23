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


#__all__ = [] # TBD

__author__ = "Mauro Bracconi"
__copyright__ = "Copyright 2020, Mauro Bracconi"
__license__ = "BSD"
__version__ = "1.0.0"
__mail__ = 'mauro.bracconi@polimi.it'
__maintainer__ = __author__
__status__ = "Alpha"

from .adaptiveDesignProcedure import *
