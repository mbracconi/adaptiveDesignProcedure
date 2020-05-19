.. adaptiveDesignTrainingPoints documentation master file, created by
   sphinx-quickstart on Wed Apr  1 09:45:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to adaptiveDesignProcedure's documentation!
========================================================

A design procedure of the training data for Machine Learning algorithms able to iteratively add datapoints according to function discrete gradient.

Reference & How to cite
------------------------
Most of the theoretical aspects behind adaptiveDesignProcedure are reported in:
M. Bracconi and M. Maestri, "Training set design for Machine Learning techniques applied to the approximation of computationally intensive first-principles kinetic models", Chemical Engineering Journal, 2020, DOI:`10.1016/j.cej.2020.125469 <https://doi.org/10.1016/j.cej.2020.125469>`_.

Installation
--------------------
The official distribution is on GitHub, and you can clone the repository using:
::

    git clone https://github.com/mbracconi/adaptiveDesignProcedure.git
    
Change directory:
::

	cd adaptiveDesignProcedure

To install the package type:
::

	python setup.py install


To uninstall the package you have to rerun the installation and record the installed files in order to remove them:
::

	python setup.py install --record installed_files.txt
	pcat installed_files.txt | xargs rm -rf

API Reference
--------------------
.. toctree::
   :maxdepth: 2

   adaptiveDesignProcedure


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
