# BioEM:  Bayesian inference of Electron Microscopy
# 2.0 VERSION: January, 2018

## Build status and test coverage

[![Build status](https://gitlab.mpcdf.mpg.de/MPIBP-Hummer/BioEM/badges/master/build.svg)](https://gitlab.mpcdf.mpg.de/MPIBP-Hummer/BioEM/commits/master)
[![Code coverage](https://gitlab.mpcdf.mpg.de/MPIBP-Hummer/BioEM/badges/master/coverage.svg?job=total_coverage)](http://MPIBP-Hummer.pages.mpcdf.de/BioEM/)
[![Doc](https://readthedocs.org/projects/pip/badge/?version=stable)](http://bioem.readthedocs.io)
[![License: GPL v3][license-badge]](License.txt)


## Contributors

Pilar Cossio, David Rohr, Fabio Baruffa, Markus Rampp, Luka Stanisic, Volker Lindenstruth and Gerhard Hummer

## References

* [Cossio, P and Hummer, G. J Struct Biol. 2013 Dec;184(3):427-37. doi: 10.1016/j.jsb.2013.10.006.](http://www.ncbi.nlm.nih.gov/pubmed/24161733)

* Cossio, P., Rohr, D., Baruffa, F., Rampp, M., Lindenstruth, V. and Hummer, G. "BioEM: GPU-accelerated computing of Bayesian inference of electron microscopy images"  [Computer Physics Communications, 210C, 163-171 (2017)](http://dx.doi.org/10.1016/j.cpc.2016.09.014), [arXiv:1609.06634](https://arxiv.org/abs/1609.06634)

## Description

The BioEM code calculates the posterior probability of a structural model given multiple experimental EM images.
The posterior is calculated by solving a multidimensional integral over many nuisance parameters that account for
the experimental factors in the image formation, such as molecular orientation and interference effects.
The BioEM software computes this integral via numerical grid sampling over a portable CPU/GPU computing platform.
By comparing the BioEM posterior probabilities it is possible to discriminate and rank structural models, allowing to characterize
the variability and dynamics of the biological system.

For a detailed description of the BioEM software see the [BioEM documentation](http://bioem.readthedocs.io), also provided in the doc/ directory.

### Command line input & help is found by just running the compiled executable ./bioEM

      ++++++++++++ FROM COMMAND LINE +++++++++++

	Command line inputs:
	  --Modelfile       arg (Mandatory) Name of model file
	  --Particlesfile   arg (Mandatory) Name of particle-image file
	  --Inputfile       arg (Mandatory) Name of input parameter file
	  --ReadOrientation arg (Optional) Read file name containing orientations
	  --ReadPDB             (Optional) If reading model file in PDB format
	  --ReadMRC             (Optional) If reading particle file in MRC format
	  --ReadMultipleMRC     (Optional) If reading Multiple MRCs
	  --DumpMaps            (Optional) Dump maps after they were read from particle-image file
	  --LoadMapDump         (Optional) Read Maps from dump option
	  --OutputFile      arg (Optional) For changing the outputfile name
	  --help                (Optional) Produce help message

Details for the inputfiles and formats are provided in sections [1](http://bioem.readthedocs.io/en/latest/index.html#the-bioem-software) and [2](http://bioem.readthedocs.io/en/latest/index.html#bioem-input) of the [BioEM documentation](http://bioem.readthedocs.io).

### Output

* Main output file with, default name "Output_Probabilities", provides the posterior probability for each image, as well as the parameters that give a maximum value of the posterior:
     
     RefMap #(number Particle Map) Probability  #(log(P))
     
     RefMap #(number Particle Map) Maximizing Param: #(Euler Angles) #(PSF parameters) #(center displacement)

     **Important: It is recommended to compare log(P) with respect to other Models or to Noise as in [1].

* (Optional) Write the posterior probabilities as a function of the orientations (key word: WRITE_PROB_ANGLES in InputFile, see [documentation](http://bioem.readthedocs.io/en/latest/index.html#std:inpar-WRITE_PROB_ANGLES)).

### Tutorial
 
A directory with example EM particles, models, and input files are provided in the Tutorial_BioEM directory. 
The tutorial is described in [section 4 of the BioEM documentation](http://bioem.readthedocs.io/en/latest/index.html#tutorial). 


### Installation

To build and install bioEM a cmake procedure is used, for example:

```
#clone the repository
git clone ...
cd BioEM
#build the code (CPU version)
mkdir build
cd build
cmake ..
make VERBOSE=1
```

Dependencies and software requirements:

* Compiler: a modern C++ compiler which is OpenMP compliant
              and (optionally) complies with CUDAs nvcc
              (tested with Intel icpc 12-16, GCC 4.7-5.1)
    -> adapt the name of the compiler using ccmake 

    for free software see: https://gcc.gnu.org/

* MPI: the Message Passing Standard library
         (tested with Intel MPI 4.1-5.1, IBM PE 1.3-1.4)
    -> adapt the names of the MPI compiler wrappers using ccmake 

    for free software see: http://www.open-mpi.de/
          
* FFTW: a serial but fully thread-safe fftw3 installation or equivalent (tested with fftw 3.3)
     -> point environment variable $FFTW_ROOT to a FFTW3 installation or use ccmake to specify

    for free software see: http://www.fftw.org 

* CUDA (required to build and run the GPU version) 

    for free software see: https://developer.nvidia.com/cuda-downloads

For details on the installation [section 1 of the BioEM documentation](http://bioem.readthedocs.io/en/latest/index.html#the-bioem-software).


### Performance Variables

The BioEM performance variables enhance or modify the code's computational performance without modifying the numerical results.
They should be tuned for the specific computing node characteristics where bioEM is executed, e.g., select the number of GPUs to use, OpenMP 
threads etc. These are passed via environment variables. See [section 4 of the BioEM documentation](http://bioem.readthedocs.io/en/latest/index.html#performance) for a detailed description.

### License 

The BioEM program is a free software, under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License. 
This program is distributed in the hope that it will be useful, but without any warranty.  See License.txt for more details.

[license-badge]: https://img.shields.io/badge/License-GPL%20v3-blue.svg