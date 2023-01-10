from setuptools import setup

description = (
    "A design procedure of the training data for Machine Learning"
    "algorithms able to iteratively add datapoints according to"
    "function discrete gradient"
    "Reference:"
	"Most of the theoretical aspects behind adaptiveDesignProcedure"
	"are reported in:"
	"M. Bracconi and M. Maestri, Training set design for "
	"Machine Learning techniques applied to the approximation "
	"of computationally intensive first-principles kinetic models,"
	"Chemical Engineering Journal, 2020, "
	"DOI: 10.1016/j.cej.2020.125469"
    "\n"
)

setup(name='adaptiveDesignProcedure',
	  version='1.4.0',
	  description='adaptiveDesignProcedure',
	  long_description=description,
	  classifiers=[
	  	'Development Status :: 5 - Production/Stable',
	  	'License :: OSI Approved :: BSD License',
	  	'Programming Language :: Python :: 3.6',
	  	'Intended Audience :: Science/Research',
	  	'Topic :: Scientific/Engineering :: Chemical Engineering'
	  ],
	  keywords='machine learning; computational fluid dyamics',
	  url='https://github.com/mbracconi/adaptiveDesignProcedure',
	  author='Mauro Bracconi',
	  author_email='mauro.bracconi@polimi.it',
	  license='BSD',
	  packages=['adaptiveDesignProcedure'],
	  install_requires=[
			'joblib',
			'numpy==1.23.4'
			'scikit-learn==1.1.2',
			'boruta==0.3',
			'packaging'
	  ],
	  extras_require={ 'docs': ['Sphinx', 'sphinx_rtd_theme'] },
	  test_suite='',
	  tests_require=[''],
	  include_package_data=True,
	  zip_safe=False)
