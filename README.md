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
git clone https://github.com/mbracconi/adaptiveDesignProcedure.git
```
Change directory:
```bash
cd adaptiveDesignProcedure
```
The **adaptiveDesignProcedure** class is in the adaptiveDesignProcedure.py file

## Documentation :
**adaptiveDesignProcedure** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation. To build the html versions of the docs simply:

```bash
> cd docs
> make html
```

## Example:
Run in a terminal example.py to reproduce "Showcase of the procedure" (Section 4.1) of M. Bracconi & M. Maestri, Chemical Engineering Journal, 2020, DOI: 10.1016/j.cej.2020.125469

## **Requirements:**
* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [scikit-learn](https://scikit-learn.org/stable/)

## **Acknowledgements:**
* [ERC SHAPE](http://www.shape.polimi.it/) project held by [Prof. Matteo Maestri](http://www.shape.polimi.it/people/matteo-maestri/)

