.. BioEM documentation master file, created by
   sphinx-quickstart on Tue Dec  5 14:19:56 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

..
   .. role:: math(raw)
      :format: html latex


.. role:: raw-latex(raw)
      :format: latex



#################################
Welcome to BioEM's documentation!
#################################

*A software for Bayesian analysis of EM images*

:Authors:

   P. Cossio, D. Rohr, F. Baruffa, L. Stanisic, M. Rampp, V. Lindestruth, G. Hummer

:Organization:

   *Max Planck Institute of Biophysics*

   *Frankfurt Institute for Advanced Studies*

   *Max Planck Computing and Data Facility*

:Date:

   December 2015

:Preface & Disclaimer:

   This manual is a preliminary guide for installation and use of the
   BioEM software. It is not intended to be complete. As the BioEM
   code is improved and developed, the manual will be updated. For any
   comments or questions please contact:
   *pilar.cossio@biophys.mpg.de*.

:Copyright:

   :math:`<`\ Bayesian inference of Electron Microscopy images\
   :math:`>` **BioEM 1.0** software for Copyright (C) 2016 Pilar Cossio,
   David Rohr, Fabio Baruffa, Markus Rampp, Volker Linderstruth and
   Gerhard Hummer.

   **BioEM 2.0** Copyright (C) 2018 Pilar Cossio, Luka Stanisic,
   Markus Rampp and Gerhard Hummer.

   The BioEM program is a free software, under the terms of the GNU
   General Public License as published by the Free Software
   Foundation, version 3 of the License. This program is distributed
   in the hope that it will be useful, but **without any
   warranty**. See the GNU General Public License for more details.

:Citation:

   Please cite refs. :cite:`CossioHummerJSB_2013,BioEM_server`.

The BioEM software
==================

Introduction
------------

Most biological systems are dynamic, they change conformation with time
and inter-convert between several functional metastable states. These
flexible biomolecules can be characterized using electron microscopy
(EM), a technique that produces frozen images of the sample in a
near-native environment. Each individual image contains information of
the instantaneous configuration of the biomolecule, and, in principle,
each particle can be in a different conformational state. However,
analyzing the images individually is challenging because the
signal-to-noise level is very low. This has so far limited EM to study a
subset of static biomolecules because the reconstruction of high-resolution
density maps requires most particles to be in the same conformation.

Here, we present a computing tool to harness the single-molecule
character of EM for studying dynamic biomolecules. With our method, we
can categorize and classify models of flexible biomolecules from
individual EM images. Bayesian inference of electron microscopy images,
BioEM :cite:`CossioHummerJSB_2013,BioEM_server`, allows us
to compute the posterior probability of a model given experimental data.
The BioEM posterior is calculated by solving a multidimensional integral
over many nuisance parameters that account for the experimental factors
in the image formation, such as molecular orientation and interference
effects. The BioEM software computes this integral via numerical grid
sampling over a portable CPU/GPU computing platform. By comparing the
BioEM posterior probabilities it is possible to discriminate and rank
structural models, allowing to characterize the dynamics
of the biological system.

In this chapter, we briefly describe the mathematical background of the
BioEM method. Then, we present the necessary tools and procedures to
install the BioEM software. We describe the prerequisite programs that
should be preinstalled on the compute node. Then, we explain the BioEM
download files and directories. Lastly, we describe the steps to install
BioEM using the CMake program. The commandline executions are using the
bash scripting language.

.. _theory:

Theoretical background
----------------------

The BioEM method calculates the posterior probability of a model,
:math:`m`, given a set of experimental images,
:math:`\omega \in \Omega`. Its key idea is to create a calculated image,
from the original model, as similar as possible to the experimental
image. The calculated image is generated using nuisance parameters,
:math:`\boldsymbol \theta`, that describe the molecule orientation,
interference effects with the Point Spread Function (PSF), uncertainties
in the particle center, intensity normalization, offset and noise.
:numref:`Fig. %s<likeliCons>` exemplifies how a calculated image from a model,
with a given set of nuisance parameters, is created. Technically, the
model is first rotated to a given orientation, then projected along the
:math:`z`-axis, then it is convoluted with a PSF to cope with imaging
artifacts, next it is shifted by a certain number of pixels to account
for the uncertainties in the particle center. Normalization, and offset
in the intensity, as well as noise, are taken implicitly into account.
The calculated image is compared to an experimental particle-image,
:math:`\omega`, through a likelihood function,
:math:`L(\omega|m,\boldsymbol\theta)`. Eq. 7 of
ref. :cite:`CossioHummerJSB_2013` shows its analytical
formulation.

.. _likeliCons:
.. figure:: ./img/Fig1_10Dec.*

   *Steps in building a realistic image starting from a 3D
   model:* rotation, projection, point spread function convolution,
   center displacement, and integrated-out parameters of normalization,
   offset and noise. The likelihood function establishes the similarity
   between the calculated image and the observed experimental image.

The posterior probability of a model, given an experimental image, is a
weighted integral over the product of prior probabilities and
likelihood, over all nuisance parameters,

.. math::
   :label: pmom

   P_{m\omega} \propto \int
     L(\omega|m,\boldsymbol\theta)p_M(m)p(\boldsymbol\theta)
     d\boldsymbol\theta~,

where :math:`p_M(m)`, :math:`p(\boldsymbol\theta)` are the prior
probabilities of model and parameters, respectively. The BioEM
software is used to perform the integrals in Eq. :eq:`pmom` over
orientation, PSF parameters, and center displacement using numerical
grid sampling. The remaining integrals over the intensity
normalization, offset, and noise are performed analytically following
ref. :cite:`CossioHummerJSB_2013`.

The posterior probability of a single model given a set of images,
:math:`\omega \in \Omega`, becomes

.. math::
   :label: pb2

   P(m|\Omega)  \propto \prod_{\omega=1}^{\Omega}P_{m\omega}~.

The main result of the BioEM software is the computation of Eq.
:eq:`pb2`. This can be used for model comparison and discrimination
(*e.g.*, to rank the best model) or to calculate the posterior
probability of a full set of models, :math:`m \in M`, following Eq. 2 of
ref. :cite:`CossioHummerJSB_2013`.

In this manual, it is assumed that the user has sufficient comprehension
of the BioEM theory. Therefore, it is encouraged to read
refs. :cite:`CossioHummerJSB_2013,BioEM_server` thoroughly.

Installation
------------

Prerequisite programs and libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before installation, there are several programs and libraries that
should be preinstalled on the compute node. First check that the
compiler is a modern C++ compiler which is OpenMP compliant. In the
following, we give a brief explanation of the mandatory, and optional
prerequisite programs.

Mandatory preinstalled libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *FFTW library (minimal version 3.3.3):* is a subroutine library
   for computing the discrete Fourier transform. It is specifically used
   in BioEM, to calculate the convolution of the ideal image with the
   PSF, and the cross-correlation of the calculated image to the
   experimental image. FFTW can be downloaded from the webpage
   https://fftw.org/.

Optional preinstalled programs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The optional but *encouraged* to use programs for an easy compilation,
and optimal performance, are described below:

-  *CMake (minimal version 2.6):* is a cross-platform software for
   managing the build process of software using a compiler-independent
   method (*i.e.*, creating a Makefile). CMake can be downloaded from
   https://cmake.org/.

-  *CUDA (minimal version 5.5):* is a parallel computing platform
   implemented by the graphics processing units (GPUs) that NVIDIA
   produce. Thus, NVIDIA graphics cards are necessary for running BioEM
   with the CUDA implementation. For more information see
   https://nvidia.com/.

-  *MPI:* Message Passing Interface is a standardized and portable
   message-passing system designed to function on a wide variety of
   parallel computers, with and without shared-memory. Any MPI platform
   (either openMPI or MPICH) can be used with BioEM. The minimal version
   of *impi* is 5.0.

-  *Git:* is a system that is used for project development (see
   https://git-scm.com/). Git can be used to clone the BioEM software
   into a local directory.

After these programs are successfully installed on your compute node, it
will be possible to install BioEM.

.. note::

   It is recommended that the same compiler that is used to compile
   the libraries is also used to compile BioEM.

.. _download:

Download
~~~~~~~~

The BioEM software can be cloned using *git* from
https://github.com/bio-phys/BioEM with

.. code-block:: bash

   git clone https://github.com/bio-phys/BioEM

A compressed directory of the BioEM software can be also directly
downloaded from https://github.com/bio-phys/BioEM. After
downloading the *zip* file, uncompress it by executing

.. code-block:: bash

   unzip BioEM.zip

In the **BioEM** directory there are:

-  the source code *.cpp* and *.cu* files.

-  the **include** directory with corresponding header files.

-  the copyright license, and *README.md* file.

-  the *CMakeLists.txt* file that is necessary for installation with
   CMake (see below).

-  the **Tutorial\_BioEM** directory that includes the example files
   used in the tutorial (chapter :ref:`tutorial`). Inside this directory,
   there is also a directory called **MODEL\_COMPARISON**.

-  the **Quaternions** directory that includes files with lists of
   quaternions that sample uniformly the rotational group *SO3* (section
   :ref:`intor`).

Installing BioEM with CMake
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest installation of BioEM is done with the CMake program.
CMake contains all the instructions to generate automatically a
*Makefile* according to the specific architecture of the computing
node, and the desired features of parallelization. CMake uses the
*CMakeLists.txt* file. This file is provided in the uncompressed
**BioEM** directory. The *CMakeLists.txt* has several modifiable
options, that should be enabled/disabled (**ON**/**OFF**,
respectively) according to the desired functionalities. The keywords
for the modifiable options are shown in :numref:`Table %s
<tableCMake>`. These options can be enabled or disabled by executing
cmake with

.. code-block:: bash

   -D<optionname>=ON/OFF

For example, to turn on the compilation with CUDA run

.. code-block:: bash

   cmake -DUSE_CUDA=ON CMakeLists.txt

It is also possible to modify these options directly in the
CMakeLists.txt file. At the beginning of this file, the keywords and
ON/OFF options are presented.

.. _tableCMake:
.. table:: CMake keyword options.

   +-----------------------------+---------------------------------------------------------+
   | **<optionname>**            | **Option**                                              |
   +=============================+=========================================================+
   | ``USE_OPENMP``              | Enable/Disable OpenMP                                   |
   +-----------------------------+---------------------------------------------------------+
   | ``USE_MPI``                 | Enable/Disable MPI                                      |
   +-----------------------------+---------------------------------------------------------+
   | ``USE_CUDA``                | Enable/Disable CUDA                                     |
   +-----------------------------+---------------------------------------------------------+
   | ``PRINT_CMAKE_VARIABLES``   | Printout CMake variables                                |
   +-----------------------------+---------------------------------------------------------+
   | ``CUDA_FORCE_GCC``          | | Force of GCC as host compiler for CUDA part           |
   |                             | | (If standard host compiler is incompatible with CUDA) |
   +-----------------------------+---------------------------------------------------------+


.. note::

   For certain architectures, an *FindFFTW.cmake* may be required to
   find the FFTW libraries. This file is included in the **BioEM**
   directory.

Steps for basic installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Create a build directory in the main **BioEM** directory, and access
   it by

.. code-block:: bash

   mkdir build && cd build

-  Run CMake with the desired options and the *CMakeLists.txt* file

.. code-block:: bash

   cmake -D<optionname1>=ON -D<optionname2>=OFF  ../CMakeLists.txt

-  If this process is successful, a *Makefile* and **CMakeFiles**
   directory should be generated. If this is not the case, enable the
   variable ``PRINT_CMAKE_VARIABLES``, and re-run CMake with verbosity
   to debug.

-  After generating the *Makefile*, execute it

.. code-block:: bash

   make

-  If this process is successful a ``bioEM`` executable should be
   generated.

For a simple test, run the BioEM executable

.. code-block:: bash

   ./bioEM

If the code runs successfully, the output on the terminal screen
should be as shown in :numref:`Listing %s<cmdline>`.

.. .. _tabletest:
.. code-block:: none
   :caption: BioEM commandline input options
   :name: cmdline

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

BioEM Input
===========

In this chapter, we describe the BioEM input commands and keywords.
BioEM has two main sources of input: from the commandline and from the
input-parameter file. In the first section, we describe each
commandline item from :numref:`Listing %s<cmdline>`. In the second
section, we describe the keywords that should be specified in the
input-parameter file. Lastly, we describe the specific formats of the
model, particle-image, and input-parameter files that are used in the
BioEM software.

Commandline input
-----------------

The BioEM software requires a model, a set of experimental images and
a input-parameter file. The names of these files are passed to the
``bioEM`` executable via the commandline, as well as their format
specifications. We now give a detailed description of the commandline
input items shown in :numref:`Listing %s<cmdline>`.

.. _modfile:

Model file
~~~~~~~~~~

.. option:: --Modelfile <arg>

The structural model is represented as spheres in 3-dimensional space.
The position of the center of the sphere should be specified in the
model file, as well as its corresponding radius and number of electrons.
These spheres can represent atoms, coarse-grained residues or
multi-scale blobs. The radius size approximately determines the
resolution of the model. Spheres with radius less than the pixel size
are projected on to a single pixel.

The name of the file containing the model has to be provided in the
commandline when ``bioEM`` is executed:

.. code-block:: bash

   ./bioEM --Modelfile arg

where ``arg`` is the model filename. The possible formats for the model
(*pdb* or text) are described in section :ref:`modformat`.

.. _partimag:

Particle-image file
~~~~~~~~~~~~~~~~~~~

The name of the experimental particle-image file is passed to the BioEM
executable using the commandline:

.. option:: --Particlesfile <arg>

where ``arg`` is the particle-image file name. The possible formats for
the particle-images (*mrc* or text) are described in section
:ref:`imaformat`.

Additional features to read the particle-images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If one has to read the same particle-image set multiple times, the
following options might be useful. The first time the particle-image
file is read, include in the commandline the keyword

.. option:: --DumpMaps

This will writeout a file *maps.dump* containing the particle-images in
binary format, which will be useful for a faster re-reading.

To read the dumped maps in binary format, use

.. option:: --LoadMapDump

Note that the *maps.dump* file should be in the same directory where
the code is executed. Using this last option, it is not necessary to
include :option:`--Particlesfile` in the commandline. See chapter
:ref:`tutorial` for examples.

.. _infile:

Input-parameter file
~~~~~~~~~~~~~~~~~~~~

BioEM has two sets of variables. One set describes the physical problem,
like the number of pixels, and the parameter integration ranges. Another
set describes the runtime configuration, which involves how to
parallelize, whether to use a GPU, and some other algorithmic settings.
The latter set does not change the output, but has a large influence on
the compute performance. The two sets are treated differently, because
the first set is related to the actual problem, while the second set
belongs to the compute node where the problem is processed. For a
detailed description of the performance variables see chapter
:ref:`perfparm`.

The physical parameters are passed via an input-parameter file that
contains specific keywords for the physical constraints, and integration
limits of the algorithm. The name of the input-parameter file is passed
via the commandline:

.. option:: --Inputfile <arg>

where ``arg`` is the filename.

In section :ref:`inparam`, we describe in detail the keywords used in the
input-parameter file.

.. _ortfile:

Orientations from a file
~~~~~~~~~~~~~~~~~~~~~~~~

In BioEM there is an option to read the orientations of a model directly
from a file, instead of calculating them in the code (see also section
:ref:`intor`). This option provides more flexibility to perform the integral
over the orientations.

For this feature use the following commandline keyword

.. option:: --ReadOrientation <arg>

where ``arg`` is the name of the file containing the list of
orientations. The format for the orientations (Euler angles or
quaternions) is described in section :ref:`orform`.

.. _biout:

BioEM output
~~~~~~~~~~~~

By default, the main BioEM output file is called

   .. outpar:: Output_Probabilities
   .. object:: Output_Probabilities

To change the name of the output file use the following commandline
keyword

.. option:: --OutputFile <arg>

where ``arg`` is the desired name of the output file. This file contains
the logarithm of the posterior probability of the model to each
individual experimental image and the parameter set that gives a maximum
of the posterior (see section :ref:`anaout` for its format).

.. _inparam:

Input of physical parameters
----------------------------

Up to now, we have seen several commandline inputs that can be used in
BioEM. We now focus on the input of the physical parameters that are
necessary for the BioEM computation and are read from *inside* the
input-parameter file. These parameters describe the physical constraints
of the algorithm, such as the integration ranges and grid points, and
are passed using specific keywords in the this file (see also section
:ref:`infile`).

Micrograph parameters
~~~~~~~~~~~~~~~~~~~~~

Mandatory inputs for the description of the experimental particle-image
are

  .. inpar:: PIXEL_SIZE
  .. object:: PIXEL_SIZE (float)

     Pixel size in :math:`\AA` of the experimental micrograph.

  .. inpar:: NUMBER_PIXELS
  .. object:: NUMBER_PIXELS (int)

     We assume a square particle-image. Here, ``(int)`` is the number
     of pixels in each dimension, *e.g.*, for a particle-image of 220
     x 220 pixels, then ``(int)= 220``.

In the BioEM calculation, the integration over the model orientations,
PSF parameters, and center displacement are performed numerically. To do
so, one needs to define the integration ranges, and grid spacing for
each parameter. These quantities depend on the experimental conditions,
such as defocus range, and thus should be specified by the user.

.. _intor:

Integration of orientations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to describe the orientation of the model in 3D space:
with the Euler angles or with quaternions.

-  *Euler Angles*. The Euler angles are :math:`\alpha,\beta,\gamma`, and
   represent a sequence of three elemental rotations about the axes of a
   coordinate system. We use the reference rotations
   :math:`Z_1 X_2 Z_3`, such that the first rotation is around the
   :math:`z`-axis by an angle :math:`\alpha`, the second rotation is
   around the :math:`x`-axis by an angle :math:`\beta`, and a last
   rotation is again around the :math:`z`-axis by an angle
   :math:`\gamma`.

-  *Quaternions*. The orientation of a rigid body can also be described
   with quaternions. A set of quaternions is a four-dimensional vector
   over the real numbers (:math:`q_1`, :math:`q_2`, :math:`q_3`,
   :math:`q_4`) each within :math:`[-1,1]` such that
   :math:`1=q_1^2+q_2^2+q_3^2+q_4^2`.

There are several ways to sample the space of Euler angles or
quaternions. We *importantly remark* that not all possibilities sample
uniformly the group of rotations in 3D space (*SO3*), which is crucial
to perform a fast and accurate integration of uniformly distributed
model orientations.

Uniform sampling of SO3
^^^^^^^^^^^^^^^^^^^^^^^

To uniformly sample *SO3*, we recommend using a list of quaternions
generated with the successive orthonormal images method from
ref. :cite:`Yershova2010`. In the directory **Quaternions**, we
provide lists of quaternions that have been generated using this
method. Here, it is necessary to follow section :ref:`ortfile` because
a list of quaternions is read from a separate file. To use quaternions
the keyword :inpar:`USE_QUATERNIONS` in the input-parameter file is
also required.

Non-uniform sampling
^^^^^^^^^^^^^^^^^^^^

It is also possible to have trivial grid-sampling of the Euler angles or
quaternions:

-  *Grid-sampling of the Euler Angles (*\ :math:`\alpha,\beta,\gamma`\
   *):* Sampling of the full Euler angle space within an uniform
   cubic-grid: :math:`\alpha \in [-\pi,\pi]`, :math:`\cos(\beta) \in
   [-1,1]` and :math:`\gamma \in [-\pi,\pi]`. Here one needs to
   provide the number of grid points in :math:`\alpha`, and
   :math:`\cos(\beta)`. By default, the grid spacing of Euler angle
   :math:`\gamma` will be the same as that of :math:`\alpha`. The
   keywords in the parameter file are

   .. inpar:: GRIDPOINTS_ALPHA
   .. object:: GRIDPOINTS_ALPHA (int)

   .. inpar:: GRIDPOINTS_BETA
   .. object:: GRIDPOINTS_BETA (int)

   where ``(int)`` is the number of grid points.

   .. note::

      For an optimal grid spacing, it is recommended that
      ``GRIDPOINTS_ALPHA~ 2*GRIDPOINTS_BETA``.

-  *Grid-sampling of quaternions:* With BioEM it is also possible to
   generate a grid in quaternion space. One should provide the keywords

   .. inpar:: USE_QUATERNIONS
   .. object:: USE_QUATERNIONS

   .. inpar:: GRIDPOINTS_QUATERNION
   .. object:: GRIDPOINTS_QUATERNION (int)

   where ``(int)`` is the grid spacing in each dimension :math:`[-1,1]`.

-  *Non-uniform sampling of orientations from a file:* We note that with
   the option of reading the orientations from a file (section
   :ref:`ortfile`) the user has great freedom to sample also non-uniformly
   the orientation space (for example around a given orientation, see :ref:`modcom`).

Integration of the PSF parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To take into account the interference effects in the experiment, we
convolute the ideal image from the model with the PSF. In practice, we
use its Fourier-space equivalent, which is the multiplication the
contrast transfer function (CTF) and envelope function. An approximate
expression for the CTF is

.. math:: \mathrm{CTF}(s)=-A\cos(as^2/2)-\sqrt{1-A^2}\sin(as^2/2),

where :math:`s` is the radial spatial frequency, and
:math:`a=2\pi \lambda \Delta f` with :math:`\lambda` is the electron
wavelength, and :math:`\Delta f` is the defocus. Parameter
:math:`A \in [0,1]` establishes the contributions of the cosine and sine
components.

The envelope function is

.. math:: \mathrm{Env}(s)=e^{-bs^2/2},

where parameter :math:`b` controls the Gaussian width and modulates the
CTF.

To calculate the BioEM posterior probability, we integrate numerically
the three parameters :math:`\Delta f`, :math:`b` and :math:`A`. To do
so, one should include in the input-parameter file the keyword for each
parameter, its integration limits, and number of grid points:

  *Parameter – (start) – (end) – (gridpoints)*

  .. inpar:: CTF_DEFOCUS
  .. object:: CTF_DEFOCUS (float) (float) (int)

  .. inpar:: CTF_B_ENV
  .. object:: CTF_B_ENV (float) (float) (int)

  .. inpar:: CTF_AMPLITUDE
  .. object:: CTF_AMPLITUDE (float) (float) (int)

The defocus, :math:`\Delta f`, should be in units of :math:`\mu`\ m,
and :math:`b` in Å\ :math:`^2`. The amplitude parameter :math:`A` is
adimensional within :math:`[0,1]`. The default value of the electron
wavelength is 0.019688\ :math:`\AA`, which corresponds to a :math:`300
kV` microscope. To change this value use the keyword

  .. inpar:: ELECTRON_WAVELENGTH
  .. object:: ELECTRON_WAVELENGTH (float)

where ``(float)`` should be in :math:`\AA`.

Integration of center displacement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integration of the particle center is done over a square and uniform
grid. The particle, along both directions, is translated from its center
up to a maximum distance (*max displ.*). Users should provide this
maximum displacement and the grid spacing in units of pixels.

The keyword in parameter file is:

  *Parameter - (max displ.) - (grid-space)*

  .. inpar:: DISPLACE_CENTER
  .. object:: DISPLACE_CENTER (int) (int)

If ``[DISPLACE_CENTER 10 2]``, the integration will be done along
:math:`x` within :math:`[x_c-10,x_c+10]` (where :math:`x_c` is the
center), and :math:`[y_c-10,y_c+10]` along :math:`y`, with sampling
every 2 pixels.

The integration over the *normalization*, *offset* and *noise* are
carried out analytically. See Supplementary Information of
ref. :cite:`CossioHummerJSB_2013`.

.. _priorsec:

Priors
~~~~~~

- *Uniform model prior probability:* To include a uniform model prior
  use the following keyword in the input-parameter file

  .. inpar:: PRIOR_MODEL
  .. object:: PRIOR_MODEL (float)

  where ``(float)`` is the value of the model’s prior.

- *Prior for orientations:* It is possible to assign prior
  probabilities for each orientation. The keyword

  .. inpar:: PRIOR_ANGLES
  .. object:: PRIOR_ANGLES

  allows to read the prior of each orientation from the input file of
  orientations (see section :ref:`ortfile`). An extra column of format
  “%12.6f” should be added in the orientations-file, which indicates
  the value of the prior probability for each orientation.

- *Prior for* :math:`b` *envelope parameter:* To avoid full loss of
  the high-frequency components in Fourier space, the code utilizes a
  Gaussian prior on the :math:`b` envelope parameter

  .. math:: p(b)=\frac{1}{2\sqrt{2\pi}\sigma_b}e^{-b^2/2\sigma_b^2},

  where :math:`\sigma_b` is the Gaussian width. By default the
  Gaussian prior is centered at zero, and :math:`\sigma_b=100\AA`, to
  modify the width include in the input-parameter file the keyword

  .. inpar:: SIGMA_PRIOR_B_CTF
  .. object:: SIGMA_PRIOR_B_CTF (float)

  where ``(float)`` is the desired :math:`\sigma_b`. See also the
  supporting information of ref. :cite:`BioEM_server`.

- *Prior for* :math:`\Delta f` *defocus parameter:* BioEM implements a
  Gaussian prior on the :math:`\Delta f` defocus parameter

  .. math:: p(\Delta f)=\frac{1}{\sqrt{2\pi}\sigma_{\Delta f}}e^{-(\Delta f - \Delta f_c)^2/2\sigma_{\Delta f}^2},

  where :math:`\sigma_{\Delta f}` is the Gaussian width and
  :math:`\Delta f_c` is the Gaussian center. By default
  :math:`\sigma_{\Delta f}=1.0\mu`\ m, and :math:`\Delta
  f_c=3.0\mu`\ m. To modify these values include in the
  input-parameter file the keyword

  .. inpar:: SIGMA_PRIOR_DEFOCUS
  .. object:: SIGMA_PRIOR_DEFOCUS (float)

  where ``(float)`` is the desired :math:`\sigma_{\Delta f}`, and

  .. inpar:: PRIOR_DEFOCUS_CENTER
  .. object:: PRIOR_DEFOCUS_CENTER (float)

  to change the Gaussian center :math:`\Delta f_c`. See also the
  supporting information of ref. :cite:`BioEM_server`.

- *Prior for* :math:`A` *amplitude parameter:* BioEM implements a
  Gaussian prior on the :math:`A` amplitude parameter

  .. math:: p(A)=\frac{1}{\sqrt{2\pi}\sigma_{A}}e^{-(A - A_c)^2/2\sigma_{A}^2},

  where :math:`\sigma_{A}` is the Gaussian width and :math:`A_c` is
  the Gaussian center. By default :math:`\sigma_{A}=0.3`, and
  :math:`A_c=0`. To modify these values include in the input-parameter
  file the keyword

  .. inpar:: SIGMA_PRIOR_AMP_CTF
  .. object:: SIGMA_PRIOR_AMP_CTF (float)

  where ``(float)`` is the desired :math:`\sigma_{A}`, and

  .. inpar:: PRIOR_AMP_CTF_CENTER
  .. object:: PRIOR_AMP_CTF_CENTER (float)

  to change the Gaussian center :math:`A_c`.

.. _angprob:

Posterior probability as a function of orientations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One can write out the log-posterior as a function of each orientation.
In this case, the integration is performed over the CTF parameters,
particle-center, normalization, offset and noise, but not over the
orientations. The keyword in parameter file is

  .. inpar:: WRITE_PROB_ANGLES
  .. object:: WRITE_PROB_ANGLES (int)

With this feature there is an additional output file
:outpar:`ANG_PROB` where ``(int)`` orientations with highest posterior
are written. The orientations in this file are sorted in decreasing log-posterior
order.

Overview of keywords for the input-parameter file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following, we provide a list of the possible keywords read from
the input-parameter.

BioEM posterior probability computation:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  :inpar:`PIXEL_SIZE` ``(float)``: Micrograph pixel size in Å.

-  :inpar:`NUMBER_PIXELS` ``(int)``: Assuming a square particle-image,
   it is the number of pixels along an axis. This should coincide with
   the number of pixels read from the micrograph.

-  :inpar:`CTF_DEFOCUS` ``(float) (float) (int)``: (CTF integration)
   Grid sampling of CTF defocus, :math:`\Delta f`. Units of
   micro-meters.  ``(float) (float)`` are the starting and ending
   limits, respectively, and ``(int)`` is the number of grid points.

-  :inpar:`CTF_B_ENV` ``(float) (float) (int)``: (CTF integration)
   Grid sampling of envelope parameter :math:`b`. Units of Å\
   :math:`^2`.  ``(float) (float)`` are the starting and ending
   limits, respectively, and ``(int)`` is the number of grid points.

-  :inpar:`CTF_AMPLITUDE` ``(float) (float) (int)``: (CTF integration)
   Grid sampling of the CTF amplitude, :math:`A` (adimensional
   :math:`\in [0,1]`). ``(float) (float)`` are the starting and ending
   limits, respectively, and ``(int)`` is the number of grid points.

-  :inpar:`DISPLACE_CENTER` ``(int) (int)``: (Integration of particle
   center displacement) Sampling within a square grid. Units of
   pixels.  ``(int) (int)`` are the maximum displacement from the
   center in both directions, and the grid spacing, respectively.

Optional keywords:
^^^^^^^^^^^^^^^^^^

-  :inpar:`GRIDPOINTS_ALPHA` ``(int)``: (Integration of orientations,
   mandatory if quaterionions or `--ReadOrientation` are not used)
   Number of grid points used in the integration over Euler angle
   :math:`\alpha \in [-\pi,\pi]`. Here a cubic grid in Euler angle
   space is performed. The integral over Euler angle :math:`\gamma` is
   identical to that of :math:`\alpha`.

-  :inpar:`GRIDPOINTS_BETA` ``(int)``: (Integration of orientations,
   mandatory if quaterionions or `--ReadOrientation` are not used)
   Number of grid points used in the integration over
   :math:`\cos(\beta) \in [-1,1]`.

-  :inpar:`USE_QUATERNIONS`: (Integration of Orientations) If using
   quaternions to the describe the orientations. *Recommended* for
   uniformly sampling of :math:`SO3` with the quaternions lists
   available in the **Quaternions** directory.

-  :inpar:`GRIDPOINTS_QUATERNION` ``(int)``: (Integration of
   Orientations) For a hypercubic grid quaternion sampling. Each
   quaternion is within :math:`[-1,1]`. ``(int)`` is the number of
   grid points per dimension.

-  :inpar:`ELECTRON_WAVELENGTH` ``(float)``: To change the default
   value of the electron wavelength ``(float)`` used to calculate the
   CTF phase with the defocus. Default 0.019688 :math:`\AA`.

-  :inpar:`PRIOR_MODEL` ``(float)``: Prior probability of
   model. **Default** 1.

-  :inpar:`PRIOR_ANGLES`: To read the prior of each orientation in the input
   file of orientations.

-  :inpar:`SIGMA_PRIOR_B_CTF` ``(float)``: To change the Gaussian width
   of the prior probability of the CTF envelope parameter :math:`b`
   (section :ref:`priorsec`). **Default** 100 Å.

-  :inpar:`SIGMA_PRIOR_DEFOCUS` ``(float)``: To change the Gaussian
   width of the prior of the defocus :math:`\sigma_{\Delta f}`
   (section :ref:`priorsec`).  **Default** 1 :math:`\mu` m.

-  :inpar:`PRIOR_DEFOCUS_CENTER` ``(float)``: To change the Gaussian
   center of the prior of the defocus :math:`\Delta f_c` (section
   :ref:`priorsec`).  **Default** 3 :math:`\mu` m.

-  :inpar:`SIGMA_PRIOR_AMP_CTF` ``(float)``: To change the Gaussian
   width of the prior of the amplitude :math:`\sigma_{A}` (section
   :ref:`priorsec`).  **Default** 0.3.

-  :inpar:`PRIOR_AMP_CTF_CENTER` ``(float)``: To change the Gaussian
   center of the prior of the amplitude :math:`A_c` (section
   :ref:`priorsec`).  **Default** 0.

-  :inpar:`NO_MAP_NORM`: Condition to not normalize to zero mean and unit
   variance the input maps.

-  :inpar:`WRITE_PROB_ANGLES` ``(int)``: To write out the posterior as
   a function of the best ``(int)`` orientation.

File formats
------------

.. _modformat:

Formats for the model file
~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two types of model file formats that are read by BioEM:

-  *Text file:* A simple text file with format “%f %f %f %f %f”. The
   first three columns are the coordinates of the sphere centers in
   :math:`\AA`, the fourth column is the radius in :math:`\AA`, and the
   last column is the corresponding number of electrons (which can be non-integer).

   (Format: ``x — y — z — radius — number electrons``).

   This format is useful for all atom, mixed or coarse-grained
   representations of the density maps.

-  *pdb file:* BioEM reads the C\ :math:`_\alpha` atom positions with
   their corresponding residue type from standard *pdb* files. A
   residue is modeled as a sphere, centered at the C\ :math:`_\alpha`,
   with van-der-Waals radii and number of electrons corresponding to
   the specific amino acid type (as in
   ref. :cite:`CossioHummerJSB_2013`). To read pdb files the following
   commandline keyword is needed (related to section :ref:`modfile`):

   .. option:: --ReadPDB

.. _imaformat:

Formats for the particle-images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two format options are allowed for the the particle-image file:

.. inpar:: PARTICLE

-  *Text file:* Data are formatted as “%8d%8d%16.8f” where the first
   two columns are the pixel indexes, and the third column is the
   intensity at that pixel. Multiple particles are read in the same
   file with the separator :inpar:`PARTICLE`. Pixel indexes should
   start at 0, and all pixels should be included.

-  *.mrc file:* BioEM also reads standard *.mrc* particle-image files.
   To do so, the additional commandline keyword is needed:

   .. option:: --ReadMRC

-  If reading multiple *mrc* files, the name of the file containing the
   *list* of all the *mrc* files should be provided. The additional
   command is required:

   .. option:: --ReadMultipleMRC

   *Example:*

   .. code-block:: bash

      --Particlesfile LIST --ReadMRC --ReadMultipleMRC

   ``LIST`` is the name of the file containing the list of names of the
   multiple *mrc* files.

   .. inpar:: NO_MAP_NORM
   .. note::

      When *mrc* particles are read, by default the intensities are
      normalized to zero average and unit standard deviation. Use the
      keyword ``NO_MAP_NORM`` in the input-parameter file to unset
      this default.

.. _orform:

Formats for the orientations file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Related to sections :ref:`ortfile` and :ref:`intor`. The format for the
orientations file is described in the following:

-  The first row of the file should have ``(int)`` equal to the total
   number of orientations.

-  The orientations can be described with Euler angles, or with
   quaternions:

   -  *Euler angles*. These are Euler angles :math:`\alpha,\beta,\gamma`
      in radians, which representing three rotations about axis
      :math:`Z_1X_2Z_3`. The format for the file containing the Euler
      angles is “%12.6f%12.6f%12.6f”, ordered as
      :math:`\alpha,\beta,\gamma`, respectively.

   -  *Quaternions*. A set of quaternions is a four-dimensional vector
      over the real numbers (:math:`q_1`, :math:`q_2`, :math:`q_3`,
      :math:`q_4`) each within :math:`[-1,1]`. The format for this
      file containing the quaternions should be
      “%12.6f%12.6f%12.6f%12.6f”, ordered as :math:`q_1`, :math:`q_2`,
      :math:`q_3`, and :math:`q_4`, respectively. To use quaternions
      the keyword :inpar:`USE_QUATERNIONS` should be placed in the
      input-parameter file.

-  **Prior for orientations.** Its possible to assign prior
   probabilities to each orientation. To do so, one should add at the
   end of each line an extra column (of format “%12.6f”) that indicates
   the value of the prior probability for each orientation.

.. _anaout:

Output format
~~~~~~~~~~~~~

The main BioEM output file is called :outpar:`Output_Probabilities` by
default. Its name can be changed using the commandline
:option:`--OutputFile` as described in section :ref:`biout`. This file
contains the logarithm of the posterior probability of the model to
each individual experimental image.

.. code-block:: bash

   RefMap [number Particle Map] LogProb  [ln(P)]

It also reports the parameter grid values that give a maximum value of
the posterior probability.

.. code-block:: bash

   RefMap [number Particle Map] Maximizing Param: [Orientation] [PSF parameters] [center displacement] [norm] [offset]

*Important remark:* The posterior probability is not normalized. Thus,
it is always recommended to compare :math:`\ln (P)` of different
models or relative to noise as in ref. :cite:`CossioHummerJSB_2013`
(see also section :ref:`modcom`).

Before executing a production run, it is recommended to check that the
values of the log-posterior are finite, and the parameters that give a
maximum of the posterior are in a reasonable range (*e.g.*, not at the
borders of the integration limits).

The output file

   .. outpar:: COORDREAD
   .. object:: COORDREAD

is always generated. It is good to check
that the model coordinates, radius and density are read correctly.

Optional outputs
^^^^^^^^^^^^^^^^

The optional output files for BioEM are:

   .. outpar:: ANG_PROB
   .. object:: ANG_PROB

     Related to section :ref:`angprob`. This file has the posterior
     probabilities for each orientation, which was specified with the
     keyword :inpar:`WRITE_PROB_ANGLES` in the parameter inputfile.
     For the Euler angles, the format of the output file is

   .. code-block:: bash

      [Map number -- alpha -- beta -- gamma -- log Probability]

   For the quaternions, its format is

   .. code-block:: bash

      [Map number -- q1 -- q2 -- q3 -- q4 -- log Probability]

.. _perfparm:

Performance
===========

The BioEM performance variables enhance or modify the code’s
computational performance without modifying the numerical results. They
should be tuned for the specific computing node characteristics where
BioEM is executed. They are passed via environment variables using the
bash scripting language.

In the following chapter, we explain the types of parallelization used
within the BioEM software, list all relevant environment variables, and
provide some suggestions for runtime configurations in different
situations.

.. _wayparallel:

Ways of parallelization
-----------------------

BioEM compares various projections of a model to a set of reference
particle-images. As explained in section :ref:`theory` the model is first
projected along a given angular orientation, then it is convoluted with
the PSF, next it is shifted by a certain number of pixels to account for
the center displacement, and finally this modified projection is
compared to a reference particle-image.

From a computational complexity perspective, the performance depends
mostly on the number of angular orientations relative to the number
experimental images. If there are many experimental images and many
orientations then the comparison of the calculated projection to all the
experimental images is by far the most time consuming part. However, if
there are few experimental images and many orientations, the comparison
part is not the time-limiting step.

**BioEM 2.0** has been optimized for both CPU and GPU performance
according to two different scenarios:

-  **Many orientations versus** *many* **experimental images**

-  **Many orientations versus** *few* **experimental images**

Because the optimal parallelization scheme changes depending on the
previous conditions, we address each item separately.

.. _multiorvsmultiim:

Many orientations vs. many experimental images
----------------------------------------------

BioEM facilitates the comparison of many orientations to many
experimental images using an all model projections to an all
particle-image comparison through a nested loop.

For this case, the following external variable modulates the BioEM
optimization algorithm:

.. code-block:: bash

   export BIOEM_ALGO=1

As shown in Fig. 2 of ref. :cite:`BioEM_server`, in the
:envvar:`BIOEM_ALGO`\ ``=1`` the outermost loop is over the
orientations and the inner most loop iterates over all particle-images
and center displacements.

Parallelization
~~~~~~~~~~~~~~~

There are multiple dimensions for parallelization:

-  *MPI:* BioEM uses MPI to parallelize over the orientations in the
   outermost loop. In this case the probabilities for all
   particle-images / PSF kernels / center displacements are calculated
   for a certain subset of orientations by each MPI process. Afterward,
   the probabilities computed by every MPI process are reduced to the
   final probabilities. If started via ``mpirun``, BioEM will
   automatically distribute the orientations evenly among all MPI
   processes.

-  *OpenMP:* BioEM can use OpenMP to parallelize over the particle
   images in the innermost loop. As processing of these particle-images
   is totally independent, there is no synchronization required at all.
   BioEM will automatically multithread over the particle-images. The
   number of employed threads can be controlled with the standard

   .. code-block:: bash

      export OMP_NUM_THREADS=[x]

   environment variable for OpenMP, where ``[x]`` is the number of
   OpenMP threads.

-  *Graphics Processing Units (GPUs):* BioEM can use GPUs to speed up
   the processing. In this case, the innermost loop over all
   particle-images, and with all center displacements, is processed by
   the GPU. The projections and the PSF convolutions are still processed
   by the CPU. This process is pipelined such that the CPU prepares the
   next projections, and PSF convolutions while the GPU calculates the
   probabilities to all particle-images for the previous calculated
   projections. Hence, this is a horizontal parallelization layer among
   the particle images with an additional vertical layer through the
   pipeline. Usage of GPUs must be enabled with the

   .. code-block:: bash

      export GPU=1

   environment variable. One BioEM process will always only use one GPU,
   by default the fastest one. A GPU device can be explicitly configured
   with the environment variable:

   .. code-block:: bash

      export GPUDEVICE=[x]

   Multiple GPUs can be used through MPI. In this case, every GPU will
   process all particle-images but calculate the probabilities only for
   a subset of the orientations (see description of MPI above).
   Selection of GPU devices for each process must be carried out by

   .. code-block:: bash

      export GPUDEVICE=-1

   In this case the MPI process with rank N on a system with G GPUs will
   take the GPU with ID (N % G). This option is mandatory when using
   MPI.

-  *GPU / CPU combined processing:* Besides the pipeline approach
   described in the previous point, which employs the CPU for creating
   the calculated image, and the GPU for calculating the likelihood to
   all particle-images, there is also the possibility to split the set
   of particle-images among the CPU and the GPU. This is facilitated by
   the environment variable

   .. code-block:: bash

      export GPUWORKLOAD=-1

   that automatically sets the percentage of particle-images processed
   by the GPU.

   It is also possible to not use this autotuning option but to set a
   static value provided by the user

   .. code-block:: bash

      export GPUWORKLOAD=[x]

   where :math:`0\le x \le100` provides the x% of particles processed by
   the GPU. However, the autotuning option is set by default.

   In an optimal situation the CPU will:

   -  Issue a GPU kernel call such that the GPU calculates the
      probabilities for x% of the particle-images for the current
      orientation and convolution.

   -  Process its own fraction of (100-x)% of the particle-images in
      parallel to the GPU.

   -  Afterward, finish the preparation of the next orientation and PSF
      convolution before the GPU has finished calculating the
      probabilities for the current orientation and PSF convolution.

-  *Multiple Projections/Convolutions at once via OpenMP:* BioEM can
   prepare the projections of multiple orientations and convolutions at
   once using OpenMP. The benefit compared to the pure OpenMP
   parallelization over the particle images, however, is tiny, while the
   memory requirements are drastically increased. This is relevant if
   MPI is not used, OpenMP is used, GPU is not used, and if the number
   of reference particle-image is small. The number of projections at
   once is determined by the environment variable

   .. code-block:: bash

      export BIOEM_PROJ_CONV_AT_ONCE=[x]

   where ``[x]`` is the number of projections that will be calculated
   simultaneously.

-  *Fourier-algorithm to process all center displacements in parallel:*
   BioEM uses as default the Fourier-algorithm to calculate the
   cross-correlation. The Fourier-algorithm automatically takes all
   displacements into account without having to loop over them. Hence,
   its runtime is almost independent from the number of center
   displacements (see ref. :cite:`BioEM_server`).

Parallelization on only CPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For parallelization over the CPU cores:

-  One can use MPI with as many MPI processes as there are CPU cores
   :math:`\times` nodes, and with :envvar:`OMP_NUM_THREADS`\ ``=1``.
   In this case, the parallelization is done only over the
   orientations.

-  On a single node, one can use OpenMP to parallelize over the
   particle images, and optionally using the environmental variable
   :envvar:`BIOEM_PROJ_CONV_AT_ONCE`\ ``=[x]`` to increase number of
   projections/convolutions processed in parallel.

-  One can combine both MPI and OpenMP, as shown in
   ref. :cite:`BioEM_server`. For instance, on a single node,
   :envvar:`OMP_NUM_THREADS`\ ``=[x]`` can be set to ``x = 1/4 N``,
   where ``N`` is the number of CPU cores on the system, and BioEM can
   be called with ``mpirun``, and 4 MPI processes. In this
   case, four orientations are processed in parallel using MPI, and
   ``x`` particle-images are processed in parallel using OpenMP.

-  If multiple nodes are used MPI is mandatory, and should be combined
   with OpenMP. Optimal work distribution will depend on the number of
   orientations (parallelization with MPI) compared to the number of
   particle-images (parallelization with OpenMP).


   .. note::

      To find the optimal performance setup for only CPUs, it is
      recommended to try both BioEM algorithms :envvar:`BIOEM_ALGO`\
      ``=1`` and :envvar:`BIOEM_ALGO`\ ``=2`` with different
      combinations of the options described.

Parallelization on CPUs and GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Naturally, different methods of parallelization can be combined with
the GPU:

-  One can combine MPI with the GPU algorithm to use multiple GPUs at
   once. The number of MPI processes has to be equal to the number of
   available GPUs.

-  One can use GPUs and CPU cores jointly to calculate the
   probabilities for all particle-images with OpenMP and the
   :envvar:`GPUWORKLOAD`\ ``=-1`` autotunning variable. For more than
   one GPU, MPI must be employed. In this case, the number of MPI
   processes must match the number of GPUs.  So it is important to
   combine MPI, and OpenMP inside one node in order to use all CPU
   cores.

Examples of possible ways of parallelization are shown in Fig. 5 and 6
of ref. :cite:`BioEM_server` for the FRH protein complex
system.

Many orientations vs. few experimental images
---------------------------------------------

**BioEM2.0** has been optimized to treat many orientations and few
experimental images using GPUs and CPUs. For this case, the following
external variable modulates the BioEM algorithm:

.. code-block:: bash

   export BIOEM_ALGO=2

In this algorithm, the parallelization for GPU is now done on a lower
level: the GPU (or OpenMP for the only CPU case) processes the center
displacements, whilst the CPU with MPI processes the orientations and
with OpenMP the projections and convolutions. Hence, there is more
parallelism and better performance for the GPU for this case.

Parallelization
~~~~~~~~~~~~~~~

We present the different parallelization options when using the
:envvar:`BIOEM_ALGO`\ ``=2``:

-  *MPI:* Similarly as with :envvar:`BIOEM_ALGO`\ ``=1`` (section
   :ref:`multiorvsmultiim`) MPI is used to parallelize over the
   orientations in the outermost loop.

-  *OpenMP:* With :envvar:`BIOEM_ALGO`\ ``=2`` the
   :envvar:`BIOEM_PROJ_CONV_AT_ONCE` is by default equal to
   :envvar:`OMP_NUM_THREADS`. However,
   :envvar:`BIOEM_PROJ_CONV_AT_ONCE` can also be modified as described
   above. Importantly, for :envvar:`BIOEM_ALGO`\ ``=2`` the contribution of
   :envvar:`BIOEM_PROJ_CONV_AT_ONCE` is significant. These
   OMP threads are used to work in parallel on the projections, the
   convolutions, and if GPU is disabled on the center displacements
   and comparisons.

-  *Graphics Processing Units (GPUs):* For :envvar:`BIOEM_ALGO`\
   ``=2`` the loop over center displacements can be processed by the
   GPU. The projections and convolutions are still processed by the
   CPU. The GPU environment variables are :envvar:`GPU`\ ``=1`` to use
   the GPU and :envvar:`GPUDEVICE`\ ``=[x]`` to select the GPU
   device. With :envvar:`GPUDEVICE`\ ``=-1`` the GPU is automatically
   selected. Note that :envvar:`GPUWORKLOAD` is always ``100``,
   meaning that all center displacements are always processed by GPU.

-  *Fourier-algorithm to process all center displacements in parallel:*
   For :envvar:`BIOEM_ALGO`\ ``=2``, the Fourier-algorithm is also
   default and always used.

Parallelization on only CPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For :envvar:`BIOEM_ALGO`\ ``=2`` and only CPUs:

-  One can use MPI with as many MPI processes as there are CPU cores
   :math:`\times` nodes and with :envvar:`OMP_NUM_THREADS`\ ``=1``. In
   this case, the parallelization is done only over the orientations .

-  On a single node, one can use OpenMP with :envvar:`OMP_NUM_THREADS`\
   ``=[x]`` to parallelize over the projections, convolutions and
   center displacements (by default using also
   :envvar:`BIOEM_PROJ_CONV_AT_ONCE`).

-  One can combine both MPI and OpenMP where MPI runs over the
   orientations and OpenMP over the projections, convolutions and
   center displacements. For instance, on a single node,
   :envvar:`OMP_NUM_THREADS`\ ``=[x]`` can be set to ``x = 1/4 N``,
   where ``N`` is the number of CPU cores on the system, and BioEM can
   be called with ``mpirun``, and 4 MPI processes.

-  If multiple nodes are used MPI is mandatory, and should be combined
   with OpenMP. Optimal work distribution will depend on the
   specifications of the nodes, and the number of orientations
   compared to the number of particle-images.

.. note::

   To find the optimal performance setup for only CPUs, it is
   recommended to try both BioEM algorithms :envvar:`BIOEM_ALGO`\
   ``=1`` and :envvar:`BIOEM_ALGO`\ ``=2`` with different combinations
   of the options described.

Parallelization on CPUs and GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For :envvar:`BIOEM_ALGO`\ ``=2``, different methods of GPU and CPU
parallelization can be combined:

-  One can combine MPI with the GPU algorithm to use multiple GPUs at
   once. The number of MPI processes has to be equal to the number of
   available GPUs.

-  One can use GPUs and CPU cores jointly. MPI will parallelize over the
   orientations, OpenMP can parallelize over the projections and the GPU
   over the convolutions and center displacements. The number of MPI
   processes must match the number of GPUs. So it is important to
   combine MPI and OpenMP inside one node in order to use all CPU cores.

Note on the numerical results
-----------------------------

**BioEM2.0** combines float and double-precision variables. Float
precision is used for most variables within the code, which
significantly speeds-up the calculations (see
:cite:`BioEM_server`). By contrary, the posterior
probability is handled in double precision to maintain a high numerical
accuracy. Nonetheless, we note that there could be a minimal numerical
difference in the computed probabilities, depending whether CPUs, GPUs
or a combination of both is used. This is coming from the different
results and rounding errors on different hardware and different
underlying libraries, thus it is hard to avoid it. However, in all
practical cases this minimal discrepancies can be considered negligible;
much smaller than the uncertainties of the numerical integrations.

List of environment variables
-----------------------------

.. envvar:: BIOEM_ALGO

   (Default: 1) Set to 1 to enable the BioEM algorithm optimized for
   many orientations versus *many* experiment images computations. Set
   to 2 to enable the BioEM algorithm optimized for many orientations
   versus *few* experiment images computations.

.. envvar:: GPU

   (Default: 0) Set to 1 to enable GPU usage, set to 0 to use only the
   CPU.

.. envvar:: GPUDEVICE

   (Default: fastest) Only relevant if :envvar:`GPU`\ ``=1``.

     - If this is not set, BioEM will autodetect the fastest GPU. Only
       possible if MPI is not used.

     - If ``x >= 0``, BioEM will use GPU number ``x``. Only possible
       if MPI is not used.

     - If ``x = -1``, BioEM runs with ``N`` MPI threads, and the
       system has ``G`` GPUs, then BioEM will use GPU with number (``N
       % G``).  The idea is that one can place multiple MPI processes
       on one node, and each will use a different GPU. For a
       multi-node configuration, one must make sure that consecutive
       MPI ranks are placed on the same node, *i.e.*, four processes
       on two nodes (node0 and node1) must be placed as (node0, node0,
       node1, node1) and not as (node0, node1, node0, node1), because
       in the latter case only 1 GPU per node will be used (by two MPI
       processes each).

.. envvar:: GPUWORKLOAD

   (Default: -1 for :envvar:`BIOEM_ALGO`\ ``=1`` and fixed to 100 for
   :envvar:`BIOEM_ALGO`\ ``=2``) Only relevant if :envvar:`GPU`\
   ``=1``. This defines the fraction of the workload in percent. To be
   precise: the fraction of the number of particle-images processed by
   the GPU. The remaining particle-images will be processed by the
   CPU. For :envvar:`BIOEM_ALGO`\ ``=1``, if set to -1 the autotuning
   option will automatically select the ideal % of particles processed
   by the GPU. For :envvar:`BIOEM_ALGO`\ ``=2`` it is fixed to
   :envvar:`GPUWORKLOAD`\ ``=100``.

.. envvar:: GPUASYNC

   (Default: 1) Only relevant if :envvar:`GPU`\ ``=1``. This uses a
   pipeline to overlap the processing on the GPU, the preparation of
   projections and convolutions on the CPU, and the DMA
   transfer. There is no reason to disable this except for debugging
   purposes.

.. envvar:: GPUDUALSTREAM

   (Default: 1) Only relevant if :envvar:`GPU`\ ``=1``. If this is set
   to 1, the GPU will use two streams in parallel. This can help to
   improve the GPU utilization. Benchmarks have shown that there is a
   very little positive effect by this setting, as utilization of GPU
   is already high.

.. envvar:: BIOEM_CUDA_THREAD_COUNT

   (Default: 256 for :envvar:`BIOEM_ALGO`\ ``=1`` and 512 for
   :envvar:`BIOEM_ALGO`\ ``=2``) Only relevant if :envvar:`GPU`\
   ``=1``. This variable can explicitly select the number of CUDA
   threads. Different inputs and algorithms might need different
   number of threads for an optimized performance, but also to respect
   hardware (memory) limits of a GPU device.

.. envvar:: OMP_NUM_THREADS

   (Default: Number of CPU cores) This is the
   standard OpenMP environment variable to define the number of OpenMP
   threads. It can be used for profiling purposes to analyze the
   scaling. It can be set to ``x=1`` to use MPI exclusively or to other
   values for a mixed MPI / OpenMP configuration.

.. envvar:: BIOEM_PROJ_CONV_AT_ONCE

   (Default: 1 for :envvar:`BIOEM_ALGO`\ ``=1`` and ``=``\
   :envvar:`OMP_NUM_THREADS` for :envvar:`BIOEM_ALGO`\ ``=2``) This
   defines the number of projections and convolutions prepared at
   once. OpenMP threads (whose number is defined by
   :envvar:`OMP_NUM_THREADS` environment variable) are used to prepare
   these projections and convolutions in parallel.  For
   :envvar:`BIOEM_ALGO`\ ``=1`` :envvar:`BIOEM_PROJ_CONV_AT_ONCE`\
   ``=[x]`` is mostly relevant, if OpenMP is used, no GPU is used,
   and/or the number of reference particle-image is very small. For
   :envvar:`BIOEM_ALGO`\ ``=2`` its contribution is important.

.. envvar:: BIOEM_DEBUG_BREAK

   (Default: deactivated) This is a debugging
   option. It will reduce the number of projection and PSF convolutions
   to a maximum of ``x`` both. It can be used for profiling to analyze
   scaling, and for fast sanity tests.

.. envvar:: BIOEM_DEBUG_NMAPS

   (Default: deactivated) As :envvar:`BIOEM_DEBUG_BREAK`, with the
   difference that this limits the number of reference particle-images
   to a maximum of ``x``.

.. envvar:: BIOEM_DEBUG_OUTPUT

   (Default: 0) Change the verbosity of the output. Higher means more
   output, lower means less output.

     - ``x=0``: Stands for no debug output.

     - ``x=1``: Limited timing output.

     - ``x=2``: Standard timing output showing durations of
       projection, convolution, and cross-correlation comparison. This
       adds successively more extensive output.

Default environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With **BioEM2.0** the Fourier-algorithm :cite:`BioEM_server`
is always used. This implies that the GPU algorithm is by default
``GPUALGO=2`` (defined in BioEM1.0). It has been shown that for
realistic cases, where the particle center is an unknown parameter, the
Fourier-algorithm outperforms all other algorithms. Because of this, we
have selected it to be permanently default.

.. _performsugg:

Suggestions for runtime configurations
--------------------------------------

Default Settings
~~~~~~~~~~~~~~~~

It is recommended that the following settings should be left at theirs
defaults: :envvar:`GPUASYNC` (Default 1), :envvar:`GPUDUALSTREAM`
(Default 1).

Profiling
~~~~~~~~~

For profiling one can limit the number of orientations, projections
and particle-images for example using :envvar:`BIOEM_DEBUG_BREAK` and
:envvar:`BIOEM_DEBUG_NMAPS`. However, for accurate estimations, it is
recommended to keep the proportion of orientations to particle-images
the same as in the actual application. Also a good choice is
:envvar:`BIOEM_DEBUG_OUTPUT`\ ``=2`` to get the timing of each
projection, convolution and comparison. For a larger number of
particle-images it might make sense to switch to
:envvar:`BIOEM_DEBUG_OUTPUT`\ ``=1``.

Production run: *Many orientations vs. many experimental images*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On only CPUs
^^^^^^^^^^^^

-  :envvar:`BIOEM_ALGO`\ ``=1`` to select the BioEM algorithm 1 that
   optimizes the computation of many orientations to many particle
   images.

-  :envvar:`BIOEM_DEBUG_OUTPUT`\ ``=0`` can reduce the size of the text
   output.

-  :envvar:`BIOEM_PROJ_CONV_AT_ONCE`\ ``=[x]`` may have a positive
   effect. The memory footprint increases with ``x``, so it should not
   be too large.  For best performance, choose a multiple of the
   number of OpenMP threads.

-  On a single node, one should use OpenMP parallelization for many
   particle-images and few orientations; and MPI parallelization for
   few particle-images and many orientations. Assume a system with
   ``N`` CPU cores, the command for the first would be

   ``BIOEM_PROJ_CONV_AT_ONCE=[4*N] OMP_NUM_THREADS=[N]``

   and for the second

   ``OMP_NUM_THREADS=1 ; mpirun -n [N]``

-  For a medium number of particle-images and orientations, a combined
   MPI / OpenMP configuration can be better.

   *Example:* Assume 20 CPU cores, possible options would be (among
   others):

   -  20 MPI processes with 1 OMP thread each:

      ``OMP_NUM_THREADS=1 mpirun -n 20``

   -  10 MPI processes with 2 OMP threads each:

      ``OMP_NUM_THREADS=2 mpirun -n 10``

   -  4 MPI processes with 5 OMP threads each:

      ``OMP_NUM_THREADS=5 mpirun -n 4``

   -  2 MPI processes with 10 OMP threads each:

      ``OMP_NUM_THREADS=10 mpirun -n 2``

   The best configuration has to be checked by the user. But in any
   case, one should make sure that the number of MPI processes times
   the number of OMP threads per process equals the number of
   (virtual) CPU cores. *Importantly*, one should also compare the
   timings from :envvar:`BIOEM_ALGO`\ ``=1`` or :envvar:`BIOEM_ALGO`\
   ``=2`` with the different configurations.

On combined CPUs and GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^

-  :envvar:`BIOEM_ALGO`\ ``=1`` to select the BioEM algorithm 1 that
   optimizes the computation of many orientations to many particle
   images.

-  :envvar:`BIOEM_DEBUG_OUTPUT`\ ``=0`` can reduce the size of the text
   output.

-  :envvar:`BIOEM_PROJ_CONV_AT_ONCE`\ ``=[x]`` may have a positive
   effect. However, the memory footprint increases with ``x``, it this
   can be a limiting factor for GPUs. Therefore, it is usually enough
   to keep the default :envvar:`BIOEM_PROJ_CONV_AT_ONCE`\ ``=1``,
   unless the number of particle images is small (in which case one
   should consider the :envvar:`BIOEM_ALGO`\ ``=2`` algorithm anyway).

- :envvar:`GPU`\ ``=1`` should be used if a GPU is
   available. Performance wise, one Titan GPU corresponds roughly to
   20 cores at 3 GHz.

-  :envvar:`GPUWORKLOAD`\ ``=-1`` for autotuning of the optimal
   workload balance.

-  If a system offers multiple GPUs, all GPUs should be used. This must
   be accomplished via MPI. In this case, the number of MPI processes
   per node must match the number of GPUs per node. There are
   different ways to make sure every MPI process uses a different GPU
   (as discussed in the GPU paragraph of section
   :ref:`wayparallel`). Assuming the MPI processes are placed such, that
   consecutive MPI ranks are placed on one node, one can use the
   :envvar:`GPUDEVICE`\ ``=-1`` setting. This is assumed here. Let us
   assume an example of ``N`` nodes with ``C`` CPU cores each and
   ``G`` GPUs each. The following command will use all GPUs, and
   ignore the CPUs:

   ``OMP_NUM_THREADS=1 GPU=1 GPUDEVICE=-1 mpirun -n [N*G]``

-  One can use all the CPU cores as well as the GPUs. A combined MPI /
   OpenMP setting as discussed previously must be used, under the
   constraint that the number of MPI processes matches the number of
   GPUs:

   ``OMP_NUM_THREADS=[C/G] GPU=1 GPUDEVICE=-1 mpirun -n [N*G]``

Production run: *Many orientations vs. few experimental images*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On only CPUs
^^^^^^^^^^^^

-  :envvar:`BIOEM_ALGO`\ ``=2`` to select the BioEM algorithm 2 that
   optimizes the computation of many orientations to few particle
   images.

-  :envvar:`BIOEM_DEBUG_OUTPUT`\ ``=0`` can reduce the size of the text
   output.

-  One should use a combination of OpenMP and MPI. Assume 20 CPU cores,
   possible options would be (among others):

   -  20 MPI processes with 1 OMP thread each:

      ``OMP_NUM_THREADS=1 mpirun -n 20``

   -  10 MPI processes with 2 OMP threads each:

      ``OMP_NUM_THREADS=2 mpirun -n 10``

   -  4 MPI processes with 5 OMP threads each:

      ``OMP_NUM_THREADS=5 mpirun -n 4``

   -  2 MPI processes with 10 OMP threads each:

      ``OMP_NUM_THREADS=10 mpirun -n 2``

   The best configuration has to be checked by the user. But in any
   case, one should make sure that the number of MPI processes times
   the number of OMP threads per process equals the number of
   (virtual) CPU cores. *Importantly*, one should also compare the
   timings from :envvar:`BIOEM_ALGO`\ ``=1`` or :envvar:`BIOEM_ALGO`\
   ``=2`` with the different configurations.

On combined CPUs and GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^

-  :envvar:`BIOEM_ALGO`\ ``=2`` to select the BioEM algorithm 2 that
   optimizes the computation of many orientations to few particle
   images.

-  :envvar:`GPU`\ ``=1`` should be used if a GPU is available.

-  For multiple GPUs, MPI has to be used, with number of MPI processes
   equal to the number of GPUs. Additionally, if there are ``x`` CPU
   cores per MPI process use :envvar:`OMP_NUM_THREADS`\ ``=[x]``.

-  Consider increasing the value of :envvar:`BIOEM_PROJ_CONV_AT_ONCE`
   to increase the parallelism, or decreasing the value of
   :envvar:`BIOEM_PROJ_CONV_AT_ONCE` to decrease GPU memory
   requirements.

-  Keep the other environment variables as default.

.. _tutorial:

Tutorial
========

In this chapter, we provide a short tutorial to perform BioEM
calculations. First, we explain the commandline executions, and
inputfile options, to calculate the posterior probability of a model
given a particle-image set. Then, we show examples of the additional
calculations that can be performed with the BioEM code. Lastly, we
give full example of how to do model comparison using BioEM.

All files mentioned in this chapter are provided in the
**Tutorial\_BioEM** directory that comes with the BioEM package (see
section :ref:`download`).

Posterior probability using BioEM
---------------------------------

We now show examples of the different commandline options and inputfile
formats used to calculate the BioEM posterior probability. Here, we only
describe the input setups related to the physical problem. For computing
node performance setups see section :ref:`performsugg`.

Commandline input and execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  *Text Model - Text Image:* To calculate the BioEM posterior
   probability of a model in text format given particle images also in
   text format.

   **Files:**

   -  Model file: *Model\_Text*

   -  Parameter input file: *Param\_Input*

   -  Particle-image file: *Text\_Image\_Form*

   **Commandline execution:**

   ``bioEM`` :option:`--Inputfile` Param_Input :option:`--Modelfile`
   Model_Text :option:`--Particlesfile` Text_Image_Form

   **Outputfile:** *Output\_Probabilities*.

   .. note::

      1. Check coordinates in the output :outpar:`COORDREAD` file to
      verify that the model is correct.

      2. The *txt* particle-image file can contain multiple particles
      that are distinguished by the separator :inpar:`PARTICLE` (see
      section :ref:`partimag`).

      3. The *Param\_Input* file is an example for a debug run. It has
      very few grid points to perform the integrations
      numerically. See section :ref:`Prorun`, for suggestions on
      input-parameter configurations for a production run.

-  *PDB Model - Text Image:* To perform the BioEM calculation with a
   model in *pdb* format.

   **New Command:** :option:`--ReadPDB`

   **Files:**

   -  Model file: *Model.pdb*

   -  Parameter file: *Param\_Input*

   -  Particle-image file: *Text\_Image\_Form*

   **Commandline execution:**

   ``bioEM`` :option:`--Inputfile` Param_Input :option:`--Modelfile`
   Model.pdb :option:`--ReadPDB`  :option:`--Particlesfile`
   Text_Image_Form

   **Outputfile:** *Output\_Probabilities*.

-  *PDB Model - One MRC Image:* To perform the BioEM calculation for a
   single *.mrc* particle-image file.

   **New Command:** :option:`--ReadMRC`

   **Files:**

   -  Model file: *Model.pdb*

   -  Parameter file: *Param\_Input*

   -  Particle-image file: *OneImage.mrc*

   **Commandline execution:**

   ``bioEM`` :option:`--Inputfile` Param_Input :option:`--Modelfile`
   Model.pdb :option:`--ReadPDB`  :option:`--Particlesfile`
   OneImage.mrc :option:`--ReadMRC`

   **Outputfile:** *Output\_Probabilities*.

-  *PDB Model - Multiple MRCs:* To perform the BioEM calculation, when
   multiple *mrc* files are read. In this case, the file name containing
   the list of all *mrc* filenames should be provided.

   **New Command:** :option:`--ReadMultipleMRC`

   **Files:**

   -  Model file: *Model.pdb*

   -  Parameter file: *Param\_Input*

   -  File with names of MRC files : *ListMRC*

   .. note::

      The file *ListMRC* contains the names of files *OneImage.mrc*
      and *TwoImages.mrc* that are provided in the **Tutorial\_BioEM**
      directory.

   **Commandline execution:**

   ``bioEM`` :option:`--Inputfile` Param_Input :option:`--Modelfile`
   Model.pdb :option:`--ReadPDB`  :option:`--Particlesfile`
   ListMRC  :option:`--ReadMRC`  :option:`--ReadMultipleMRC`

   **Example outputfile:** *Output\_Probabilities.*

   .. note::

      Both commands :option:`--ReadMRC` :option:`--ReadMultipleMRC`
      are required.

-  *Read Euler angles from file:* Related to section :ref:`intor`. With
   this feature the Euler angles are read from an input orientations
   file.

   **New Command:** :option:`--ReadOrientation`

   **Files:**

   -  Model file: *Model.pdb*

   -  Parameter file: *Param\_Input*

   -  Particle image file: *Text\_Image\_Form*

   -  EulerAngle File: *Euler\_Angle\_List*

   **Commandline execution:**

   ``bioEM`` :option:`--Inputfile` Param_Input :option:`--Modelfile`
   Model.pdb :option:`--ReadPDB`  :option:`--Particlesfile`
   Text_Image_Form :option:`--ReadOrientation`  Euler_Angle_List

   **Outputfile:** *Output\_Probabilities*

   .. note::

      If the command :option:`--ReadOrientation` is used then the code
      will disregard the Euler angle grid-sampling stated in the
      *Param\_Input* file. This means that reading the orientations
      from a file prevails over the option of calculating cubic-grids
      directly inside the code.

-  *Read quaternions from file:* Related to section :ref:`intor`. With
   this feature the quaternions are read from an input orientations
   file.

   **New Command:** :option:`--ReadOrientation`

   **Important!:** in the input-parameter file one has to add the
   keyword:

   :inpar:`USE_QUATERNIONS`

   **Files:**

   -  Model file: *Model.pdb*

   -  Parameter file: *Param\_Input\_Quat*

   -  Particle image file: *Text\_Image\_Form*

   -  Quaternion File: *Quat\_list\_Small*

   **Commandline execution:**

   ``bioEM`` :option:`--Inputfile` Param_Input_Quat
   :option:`--Modelfile` Model.pdb :option:`--ReadPDB`
   :option:`--Particlesfile` Text_Image_Form
   :option:`--ReadOrientation` Quat_list_Small

   **Outputfile:** *Output\_Probabilities*

   .. note::

      In the directory **Quaternions**, there are several quaternion
      lists that sample uniformly the rotational group in 3D space,
      *SO3*. These files are strongly *recommended* to use.

.. _Prorun:

Input-parameter suggestions for a production run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We strongly recommend to use all the prior information of the system
that is available, *e.g.*, if the orientations, defocus, etc. are known,
one should use this information to reduce the sampling time in the BioEM
algorithm. If few prior information is available, we provide the file
*Param\_ProRun* as a tentative setup for a production run that is shown
in :numref:`Table %s <tableParamPro>`.

**Commandline execution:**

``bioEM`` :option:`--Inputfile` Param_Input_Quat
:option:`--Modelfile` Model.pdb :option:`--ReadPDB`
:option:`--Particlesfile` Text_Image_Form
:option:`--ReadOrientation` List_Quat_ProRun

**Outputfile:** *Output\_Probabilities*

.. _tableParamPro:
.. table:: Input-parameter suggestions for a production run, if negligible prior information is available.

   +---------------------------------------------+
   | ``USE_QUATERNIONS``                         |
   +---------------------------------------------+
   | ``CTF_B_ENV``               2.0   300.0   4 |
   +---------------------------------------------+
   | ``CTF_DEFOCUS``             0.5     4.5   8 |
   +---------------------------------------------+
   | ``CTF_AMPLITUDE``          0.01   0.601   5 |
   +---------------------------------------------+
   | ``SIGMA_PRIOR_B_CTF``       50.             |
   +---------------------------------------------+
   | ``SIGMA_PRIOR_DEFOCUS``     0.4             |
   +---------------------------------------------+
   | ``PRIOR_DEFOCUS_CENTER``    2.8             |
   +---------------------------------------------+
   | ``DISPLACE_CENTER``          40       1     |
   +---------------------------------------------+

To note are:

-  The Gaussian prior on the :math:`b` envelope parameter, has a width
   of 50\ :math:`\AA`.

-  The Gaussian prior on the CTF defocus :math:`\Delta f` parameter, has
   a width of 0.4\ :math:`\mu`\ m, and it is centered at
   2.8\ :math:`\mu`\ m.

-  Quaternions are used to describe the orientations. The quaternions
   should be read from a file that samples uniformly :math:`SO3`. See
   for example *List\_Quat\_ProRun*, with :math:`> 4000` orientations.

-  The grid spacing of the particle-center displacement can be very fine
   if the FFT algorithm is used (see section :ref:`wayparallel`).

Additional commandline options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several additional features using the commandline are available with
BioEM:

-  *Dump particle-images:* This feature writes out the particle-images
   in binary format. This allows a faster to readout in a further BioEM
   execution.

   **New Command:** :option:`--DumpMaps`

   **Files:**

   -  Model file: *Model.pdb*

   -  Parameter file: *Param\_Input*

   -  File with names of MRC files : *ListMRC*

   **Commandline execution:**

   ``bioEM`` :option:`--Inputfile` Param_Input :option:`--Modelfile`
   Model.pdb :option:`--ReadPDB`  :option:`--Particlesfile` ListMRC
   :option:`--ReadMRC`  :option:`--ReadMultipleMRC`
   :option:`--DumpMaps`

   **Outputfiles:** \ *Output\_Probabilities* and *maps.dump*.

-  *Load particle-images:* This feature reads in the particle images in
   binary format from file *maps.dump* (see above). In this case, no
   particle-image file is necessary, but the *maps.dump* file should be
   in the current directory.

   **New Command:** :option:`--LoadMapDump`

   **Files:**

   -  Model file: *Model.pdb*

   -  Parameter file: *Param\_Input*

   -  Dumped Mapfile: *maps.dump*

   **Commandline execution:**

   ``bioEM`` :option:`--Inputfile` Param_Input :option:`--Modelfile`
   Model.pdb :option:`--ReadPDB`  :option:`--LoadMapDump`

   **Outputfile:** *Output\_Probabilities*

-  *Including prior probabilities*: To include the prior probabilities
   both for the model and orientations see the *Param\_Input\_Priors*
   file. The prior probabilities for the orientations should be included
   in an additional file (*e.g.*, see *Euler\_Angle\_List\_Prior*). An
   example is:

   **Files:**

   -  Model file: *Model\_Text*

   -  Parameter file: *Param\_Input\_Priors*

   -  Particle image file: *Text\_Image\_Form*

   -  EulerAngle File: *Euler\_Angle\_List\_Prior*

   **Commandline execution:**

   ``bioEM`` :option:`--Modelfile` Model_Text
   :option:`--Particlesfile` Text_Image_Form :option:`--Inputfile`
   Param_Input_Priors :option:`--ReadOrientation`
   Euler_Angle_List_Prior

   **Outputfile:** *Output\_Probabilities*

-  *Posterior as a function of orientations:*

   This option prints out the posterior probabilities of the model as
   a function of the orientations. In this case, all integrals in Eq.
   Eq. :eq:`pmom` are performed apart from that over the
   orientations. The keyword in the parameter file is

   :inpar:`WRITE_PROB_ANGLES` ``x``

   an additional outputfile :outpar:`ANG_PROB` is generated with the
   best ``x`` orientations. An example of the parameter input is
   provide in the *Param\_Input\_WritePAng* file.

.. _modcom:

Example: model comparison using BioEM
-------------------------------------

BioEM should be used for model comparison and ranking. Here, we provide
a complete example of how to analyze the output files of BioEM to
discriminate between structural models with two subsequent rounds of
assessment. In the first round, the orientation sampling is done
uniformly over :math:`SO3` using the BioEM algorithm 1 (*e.g.*, an
all-orientations to all-particles comparison). In the second round, the
posterior for each particle is calculated independently for a subset of
orientations that are close to the best orientation from the previous
round.

The relevant files are found in the **MODEL\_COMPARISON** directory that
is inside the **Tutorial\_BioEM** directory. There you will find:

-  *MODEL\_1*: First model in text format.

-  *MODEL\_2*: Second model in text format.

-  *Param\_Input\_ModelComparision*: example of parameter input.

-  *Quaternion\_List*: List of quaternions to sample uniformly *SO3*.

-  *20\_ParticleImages*: Stack of the particle images in text format.

-  *Particles*: Folder with the individual 20 particles files in text
   format.

-  *create\_gridOr.sh*: Bash script to create a refined grid over the
   best orientation from the previous round. This script uses python3.3
   with the file *multiply\_quat.py* and the quaternion grid file
   *smallGrid\_125*.

-  *multiply\_quat.py*: Python3.3 script that multiplies the best
   quaternion from the previous round with the quaternions from
   *smallGrid\_125* to generate a new list of quaternions that samples
   homogeneously near the best quaternion.

-  *smallGrid\_125*: Grid of quaternions around the north pole.

-  *subtract\_LogP.sh*: Bash script to calculate the difference in log
   posterior from the outputfiles.

-  *bioem\_array\_sge.sh*: Example launch script for BioEM round 2 (see
   below) for a high-performance computing (HPC) platform with the sge
   job scheduling system.

-  *bioem\_array\_slurm.sh*: Example launch script for BioEM round 2
   (see below) for a HPC platform with the slurm job scheduling system.

Before we begin it is recommended to link the ``bioEM`` executable into
the working model-comparison directory.

Round 1: Model comparison with uniform sampling for all particle images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we will calculate the BioEM probability over a uniform grid of
orientations on :math:`SO3` for all particle images using
:envvar:`BIOEM_ALGO`\ ``=1``. To compare the models, one needs to run the BioEM
program for each:

-  Model 1

   .. code:: bash

      BIOEM_DEBUG_OUTPUT=0 BIOEM_ALGO=1 ./bioEM --Modelfile
      MODEL_1 --Particlesfile 20_ParticleImages --Inputfile
      Param_Input_ModelComparision --ReadOrientation
      Quaternion_List --OutputFile Output_MODEL_1

-  Model 2:

   .. code:: bash

      BIOEM_DEBUG_OUTPUT=0 BIOEM_ALGO=1 ./bioEM --Modelfile
      MODEL_2 --Particlesfile 20_ParticleImages --Inputfile
      Param_Input_ModelComparision --ReadOrientation
      Quaternion_List --OutputFile Output_MODEL_2

Here, two output files containing the posterior probabilities of each
model (``Output_MODEL_1`` and ``Output_MODEL_2``) are generated. Since
the input-parameter and particle-image files are the same, then the
output files should only differ in the specific numerical results.

To calculate the difference in log-posterior of *Model 1* with respect
to *Model 2*, one can simply run in terminal the bash script
*subtract\_LogP.sh*:

.. code:: bash

   ./subtract_LogP.sh Output_MODEL_1 Output_MODEL_2 > Results-Round1

This script prints out the particle number, log-posterior of *Model
1*, log-posterior of *Model 2*, difference in log-posteriors (*Model
1- Model 2*), and cumulative difference.

Round 2: Model comparison with different orientations for each particle image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we can refine the BioEM probability by sampling near to the best
orientation from **Round 1** using :envvar:`BIOEM_ALGO`\ ``=2``.

We can extract the best orientations for each particle from the
*Output\_MODEL\_1* and *Output\_MODEL\_2* files, and we can generate a
new orientation grid for each particle around the best orientation. To
do so, one can use the script *create\_gridOr.sh* that creates a new
list of quaternions for each particle image.

For Model 1, the commandline instructions are

.. code:: bash

   ./create_gridOr.sh Output_MODEL_1 M1

This script takes as first column the OutputFile from round 1, and as
second column a variable that assigns a name to the new lists (for
simplicity, we have chosen ``M1``). This scripts generates twenty
individual orientation files *Quaternion\_List\_M1\_P$x* with *$x* from
1 to 20 (*i.e.*, a file per particle). All files are stored in
*/tmp/Quaternion\_Lists\_M1* folder. Note that when working with large
number of input/output files, it is very important to keep them in a
well structured, hierarchical manner and at the most appropriate
location. The optimal configuration depends on the number and size of
files, as well as the file system of the machine. As an example, we have
stored the generated folders and files in the */tmp/* directory.

Similarly, for model 2

.. code:: bash

   ./create_gridOr.sh Output_MODEL_2 M2

we obtain 20 individual files *Quaternion\_List\_M2\_P$x*, stored
inside */tmp/Quaternion\_Lists\_M2* folder.

Because each image now has a different list of orientations, one has to
launch each BioEM analysis individually within a nested loop. For
example:

-  Model 1:

   .. code:: bash

      mkdir -p /tmp/Outputs_M1_Round2
      numim=20
      for((x=1;x<${numim}+1; x++))
      do

        BIOEM_DEBUG_OUTPUT=0 BIOEM_ALGO=2 ./bioEM --Modelfile
        MODEL_1 --Particlesfile Particles/Particle_$x --Inputfile
        Param_Input_ModelComparision --ReadOrientation
        /tmp/Quaternion_Lists_M1/Quaternion_List_M1_P$x --OutputFile
        /tmp/Outputs_M1_Round2/Output_MODEL_1_P$x

        tail -2 /tmp/Outputs_M1_Round2/Output_MODEL_1_P$x >>
        Output_MODEL_1_Round2

      done

   where ``numim`` is the total number of particle images and ``x``
   indexes the particle number.

-  Model 2:

   .. code:: bash

      mkdir -p /tmp/Outputs_M2_Round2
      numim=20
      for((x=1;x<${numim}+1; x++))
      do

        BIOEM_DEBUG_OUTPUT=0 BIOEM_ALGO=2 ./bioEM --Modelfile
        MODEL_2 --Particlesfile Particles/Particle_$x --Inputfile
        Param_Input_ModelComparision --ReadOrientation
        /tmp/Quaternion_Lists_M2/Quaternion_List_M2_P$x --OutputFile
        /tmp/Outputs_M2_Round2/Output_MODEL_2_P$x

        tail -2 /tmp/Outputs_M2_Round2/Output_MODEL_2_P$x >>
        Output_MODEL_2_Round2

      done

This loop can be treated in an easier way using job arrays of sge (with
``#$ -t 1:X`` option) or slurm (with ``#SBATCH –array=1:X`` option) on
HPC platforms. In the **MODEL\_COMPARISON** directory, the files
*bioem\_array\_sge.sh* and *bioem\_array\_slurm.sh* show example launch
scripts for the procedure previously described for sge and slurm,
respectively.

To compare the resulting probabilities from round 2, one can use the
same script (*subtract\_LogP.sh*) with the new output files:

.. code:: bash

   ./subtract_LogP.sh Output_MODEL_1_Round2 Output_MODEL_2_Round2 >
   Results-Round2

In :numref:`Fig. %s<ModelComp>`, we compare the results of the BioEM
model comparison from round 1 (red) and round 2 (blue). In
:numref:`Fig. %s<ModelComp>` (top) we plot the the BioEM log-posterior
of *Model 1* versus *Model 2* (output columns 4 and 6 from the script
execution) for both rounds. These results show that the blue dots
(those from round 2) are more shifted to the left and to the top,
indicating that by refining the sampling around the best orientation
there is an increase of the posterior probability. In
:numref:`Fig. %s<ModelComp>` (bottom), we show the cumulative
difference of (*Model 1 - Model 2*) (column 10) as a function of the
image number for both rounds. From these results, one can conclude
that *Model 1* is more probable. Importantly, the discriminating power
also increases by refining around the best orientation (as shown also
in ref. :cite:`BioEM_cring`).

Lastly, we note that the analysis of the posterior probabilities can be
done in many different manners, as in
ref. :cite:`CossioHummerJSB_2013,BioEM_cring`. The final
interpretation of the results is left to the individual user.

.. _ModelComp:
.. figure:: ./img/ModComp.*

   *Example of model comparison using BioEM.* (**top**) Natural
   logarithm of the BioEM Posterior probability of *Model 1* versus
   *Model 2* for 20 particle-images for round 1 (red) of BioEM
   refinement with uniformly distributed equal orientations for all
   particles and round 2 (blue) of BioEM refinement around the best
   orientation from round 1. (**bottom**) Cumulative difference of
   *Model 1 - Model 2* as a function of the image number for round 1
   (red) and 2 (blue). Example files of models, particle-images and
   input-parameters are in the **MODEL\_COMPARISON** directory.

.. Bibliography
.. ============

.. only:: html

  .. rubric:: References

.. bibliography:: bib_manual.bib
   :style: unsrt

.. Indices and tables
.. ==================

.. only:: html

  .. rubric:: Indices and tables

  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`
