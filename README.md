# adaptiveDesignProcedure

A design procedure of the training data for Machine Learning algorithms able to iteratively add datapoints according to function discrete gradient.

## Reference & How to cite:
Most of the theoretical aspects behind **adaptiveDesignProcedure** are reported in:

M. Bracconi and M. Maestri, "Training set design for Machine Learning techniques applied to the approximation of computationally intensive first-principles kinetic models", Chemical Engineering Journal, 2020, DOI: [10.1016/j.cej.2020.125469](https://doi.org/10.1016/j.cej.2020.125469)

## Authors:
**adaptiveDesignProcedure** is developed and mantained at the Laboratory of Catalysis and Catalytic Processes of Politecnico di Milano by [Dr. Mauro Bracconi](https://www4.ceda.polimi.it/manifesti/manifesti/controller/ricerche/RicercaPerDocentiPublic.do?EVN_PRODOTTI=evento&idugov=67311)

## Installation:
Clone the repository:
```bash
> git clone https://github.com/mbracconi/adaptiveDesignProcedure.git
```
Change directory:
```bash
> cd adaptiveDesignProcedure
```
To install the package type:
```bash
> python setup.py install
```

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:
```bash
> python setup.py install --record installed_files.txt
> cat installed_files.txt | xargs rm -rf
```

## Documentation :
**adaptiveDesignProcedure** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation. To build the html versions of the docs simply:
```bash
> cd docs
> make html
```

## Example:
As an example, the "Showcase of the procedure" (Section 4.1 - M. Bracconi & M. Maestri, Chemical Engineering Journal, 2020, DOI: 10.1016/j.cej.2020.125469) is provided in this repository.

Open a terminal and go to example directory:
```bash
> cd example
```

Run the example:
```bash
> python example.py
```

## **Requirements:**
* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [scikit-learn](https://scikit-learn.org/stable/)

## **Acknowledgements:**
* [ERC SHAPE](http://www.shape.polimi.it/) project held by [Prof. Matteo Maestri](http://www.shape.polimi.it/people/matteo-maestri/)

