/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   < BioEM software for Bayesian inference of Electron Microscopy images>
   Copyright (C) 2017 Pilar Cossio, David Rohr, Fabio Baruffa, Markus Rampp,
        Luka Stanisic, Volker Lindenstruth and Gerhard Hummer.
   Max Planck Institute of Biophysics, Frankfurt, Germany.
   Frankfurt Institute for Advanced Studies, Goethe University Frankfurt,
   Germany.
   Max Planck Computing and Data Facility, Garching, Germany.

   Released under the GNU Public License, v3.
   See license statement for terms of distribution.

   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include <cstring>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "map.h"
#include "param.h"

using namespace std;

bioem_param::bioem_param()
{

  //**************** Initializing Variables and defaults ****************

  // Number of Pixels
  param_device.NumberPixels = 0;
  param_device.NumberFFTPixels1D = 0;
  // Euler angle grid spacing
  angleGridPointsAlpha = 0;
  angleGridPointsBeta = 0;
  // Envelop function paramters
  numberGridPointsEnvelop = 0;
  // Contrast transfer function paramters
  numberGridPointsCTF_amp = 0;
  numberGridPointsCTF_phase = 0;

  // ****center displacement paramters Equal in both directions***
  param_device.maxDisplaceCenter = 0;
  numberGridPointsDisplaceCenter = 0;

  fft_plans_created = 0;

  refCTF = NULL;
  CtfParam = NULL;
  angles = NULL;
  angprior = NULL;

  printModel = false;
  BestmapCalcCC = false;
}

int bioem_param::readParameters(const char *fileinput)
{ // **************************************************************************************
  // ***************************** Reading Input Parameters
  // ******************************
  // **************************************************************************************

  // Control for Parameters
  bool yesPixSi = false;
  bool yesNumPix = false;
  bool yesGPal = false;
  bool yesGPbe = false;
  bool yesMDC = false;
  bool yesBFact = false;
  bool yesDefocus = false;
  bool yesAMP = false;
  bool yesPSFenv = false;
  bool yesPSFpha = false;
  bool yesquatgrid = false;

  //***************** Default VALUES
  param_device.tousepsf = false;
  writeCTF = false;
  elecwavel = 0.019866;
  doquater = false;
  nocentermass = false;
  printrotmod = false;
  readquatlist = false;
  doaaradius = true;
  notnormmap = false;
  usepsf = false;
  yespriorAngles = false;
  printrotmod = false;
  ignorePDB = false;

  NotUn_angles = 0;
  priorMod = 1; // Default
  shiftX = 0;
  shiftY = 0;
  param_device.sigmaPriorbctf = 100.;
  param_device.sigmaPriordefo = 2.0;
  param_device.Priordefcent = 3.0;
  param_device.sigmaPrioramp = 0.5;
  param_device.Priorampcent = 0.;

  ifstream input(fileinput);
  if (!input.good())
  {
    myError("Opening file: %s", fileinput);
  }

  char line[512] = {0};
  char saveline[512];

  cout << "\n +++++++++++++++++++++++++++++++++++++++++ \n";
  cout << "\n   READING BioEM PARAMETERS             \n\n";
  cout << " +++++++++++++++++++++++++++++++++++++++++ \n";

  while (input.getline(line, 512))
  {
    strcpy(saveline, line);
    char *token = strtok(line, " ");

    if (token == NULL || line[0] == '#' || strlen(token) == 0)
    {
      // comment or blank line
    }
    else if (strcmp(token, "PIXEL_SIZE") == 0)
    {
      token = strtok(NULL, " ");
      pixelSize = atof(token);
      if (pixelSize < 0)
      {
        myError("Negative pixel size");
      }
      cout << "Pixel Size " << pixelSize << "\n";
      yesPixSi = true;
    }
    else if (strcmp(token, "NUMBER_PIXELS") == 0)
    {
      token = strtok(NULL, " ");
      param_device.NumberPixels = int(atoi(token));
      if (param_device.NumberPixels < 0)
      {
        myError("Negative Number of Pixels");
      }
      cout << "Number of Pixels " << param_device.NumberPixels << "\n";
      yesNumPix = true;
    }
    else if (strcmp(token, "GRIDPOINTS_ALPHA") == 0)
    {
      token = strtok(NULL, " ");
      angleGridPointsAlpha = int(atoi(token));
      if (angleGridPointsAlpha < 0)
      {
        myError("Negative GRIDPOINTS_ALPHA");
      }
      cout << "Grid points alpha " << angleGridPointsAlpha << "\n";
      yesGPal = true;
    }
    else if (strcmp(token, "GRIDPOINTS_BETA") == 0)
    {
      token = strtok(NULL, " ");
      angleGridPointsBeta = int(atoi(token));
      if (angleGridPointsBeta < 0)
      {
        myError("Negative GRIDPOINTS_BETA");
      }
      cout << "Grid points in Cosine ( beta ) " << angleGridPointsBeta << "\n";
      yesGPbe = true;
    }
    else if (strcmp(token, "USE_QUATERNIONS") == 0)
    //        else if (token=="USE_QUATERNIONS")
    {
      cout << "Orientations with Quaternions. \n";
      doquater = true;
    }
    else if (strcmp(token, "GRIDPOINTS_QUATERNION") == 0)
    {
      if (!notuniformangles)
      {
        token = strtok(NULL, " ");
        GridPointsQuatern = int(atoi(token));
        cout << "Gridpoints Quaternions " << GridPointsQuatern << "\n";
      }
      else
      {
        myError("Inconsistent input: grid or list with quaternions?");
      }
      yesquatgrid = true;
      doquater = true;
    }
    // CTF PARAMETERS
    else if (strcmp(token, "CTF_B_ENV") == 0)
    {
      token = strtok(NULL, " ");
      startBfactor = atof(token);
      if (startBfactor < 0)
      {
        myError("Negative start B Env.");
      }
      token = strtok(NULL, " ");
      endBfactor = atof(token);
      if (endBfactor < 0)
      {
        myError("Negative end B Env.");
      }
      token = strtok(NULL, " ");
      numberGridPointsEnvelop = int(atoi(token));
      if (numberGridPointsEnvelop < 0)
      {
        myError("Negative number of grid points B Env.");
      }
      cout << "Grid CTF B-ENV: " << startBfactor << " " << endBfactor << " "
           << numberGridPointsEnvelop << "\n";
      if (startBfactor > endBfactor)
      {
        myError("Grid ill defined end > start");
      }
      yesBFact = true;
    }
    else if (strcmp(token, "CTF_DEFOCUS") == 0)
    {
      token = strtok(NULL, " ");
      startDefocus = atof(token);
      if (startDefocus < 0)
      {
        myError("Negative start defocus");
      }
      token = strtok(NULL, " ");
      endDefocus = atof(token);
      if (endDefocus < 0)
      {
        myError("Negative end defocus");
      }
      token = strtok(NULL, " ");
      numberGridPointsCTF_phase = int(atoi(token));
      if (numberGridPointsCTF_phase < 0)
      {
        myError("Negative number of grid points defocus");
      }
      cout << "Grid CTF Defocus: " << startDefocus << " " << endDefocus << " "
           << numberGridPointsCTF_phase << "\n";
      if (startDefocus > endDefocus)
      {
        myError("Grid ill defined end > start");
      };
      if (endDefocus > 8.)
      {
        myError("Defocus beyond 8micro-m range is not allowed");
      }
      yesDefocus = true;
    }
    else if (strcmp(token, "CTF_AMPLITUDE") == 0)
    {
      token = strtok(NULL, " ");
      startGridCTF_amp = atof(token);
      if (startGridCTF_amp < 0)
      {
        myError("Negative start amplitude");
      }
      token = strtok(NULL, " ");
      endGridCTF_amp = atof(token);
      if (endGridCTF_amp < 0)
      {
        myError("Negative end amplitude");
      }
      token = strtok(NULL, " ");
      numberGridPointsCTF_amp = int(atoi(token));
      if (numberGridPointsCTF_amp < 0)
      {
        myError("Negative number of grid points amplitude");
      }
      cout << "Grid Amplitude: " << startGridCTF_amp << " " << endGridCTF_amp
           << " " << numberGridPointsCTF_amp << "\n";
      if (startGridCTF_amp > endGridCTF_amp)
      {
        myError("Grid ill defined end > start");
      };
      yesAMP = true;
    }
    else if (strcmp(token, "ELECTRON_WAVELENGTH") == 0)
    {
      token = strtok(NULL, " ");
      elecwavel = atof(token);
      if (elecwavel < 0.0150)
      {
        myError("Wrong electron wave length %lf. "
                "Has to be in Angstrom (A)",
                elecwavel);
      }
      cout << "Electron wave length in (A) is: " << elecwavel << "\n";
    }
    // PSF PARAMETERS
    else if (strcmp(token, "USE_PSF") == 0)
    {
      usepsf = true;
      param_device.tousepsf = true;
      cout << "Important: Using Point Spread Function. Thus, all parameters "
              "are in Real Space. \n";
    }
    else if (strcmp(token, "PSF_AMPLITUDE") == 0)
    {
      token = strtok(NULL, " ");
      startGridCTF_amp = atof(token);
      if (startGridCTF_amp < 0)
      {
        myError("Negative start amplitude");
      }
      token = strtok(NULL, " ");
      endGridCTF_amp = atof(token);
      if (endGridCTF_amp < 0)
      {
        myError("Negative end amplitude");
      }
      token = strtok(NULL, " ");
      numberGridPointsCTF_amp = int(atoi(token));
      if (numberGridPointsCTF_amp < 0)
      {
        myError("Negative number of grid points amplitude");
      }
      cout << "Grid Amplitude: " << startGridCTF_amp << " " << endGridCTF_amp
           << " " << numberGridPointsCTF_amp << "\n";
      if (startGridCTF_amp > endGridCTF_amp)
      {
        myError("Grid ill defined end > start");
      };
      yesAMP = true;
    }
    else if (strcmp(token, "PSF_ENVELOPE") == 0)
    {
      token = strtok(NULL, " ");
      startGridEnvelop = atof(token);
      if (startGridEnvelop < 0)
      {
        myError("Negative start PSF Env.");
      }
      token = strtok(NULL, " ");
      endGridEnvelop = atof(token);
      if (endGridEnvelop < 0)
      {
        myError("Negative end PSF Env.");
      }
      token = strtok(NULL, " ");
      numberGridPointsEnvelop = int(atoi(token));
      if (numberGridPointsEnvelop < 0)
      {
        myError("Negative number of grid points PSF Env.");
      }
      cout << "Grid PSF Envelope: " << startGridEnvelop << " " << endGridEnvelop
           << " " << numberGridPointsEnvelop << "\n";
      if (startGridEnvelop > endGridEnvelop)
      {
        myError("Grid ill defined end > start");
      }
      yesPSFenv = true;
    }
    else if (strcmp(token, "PSF_PHASE") == 0)
    {
      token = strtok(NULL, " ");
      startGridCTF_phase = atof(token);
      if (startGridCTF_phase < 0)
      {
        myError("Negative start amplitude");
      }
      token = strtok(NULL, " ");
      endGridCTF_phase = atof(token);
      if (endGridCTF_phase < 0)
      {
        myError("Negative end amplitude");
      }
      token = strtok(NULL, " ");
      numberGridPointsCTF_phase = int(atoi(token));
      if (numberGridPointsCTF_phase < 0)
      {
        myError("Negative number of grid points amplitude");
      }
      cout << "Grid PSF phase: " << startGridCTF_phase << " "
           << endGridCTF_phase << " " << numberGridPointsCTF_phase << "\n";
      if (startGridCTF_phase > endGridCTF_phase)
      {
        myError("Grid ill defined end > start");
      }
      yesPSFpha = true;
    }
    else if (strcmp(token, "DISPLACE_CENTER") == 0)
    {
      token = strtok(NULL, " ");
      param_device.maxDisplaceCenter = int(atoi(token));
      if (param_device.maxDisplaceCenter < 0)
      {
        myError("Negative MAX_D_CENTER");
      }
      cout << "Maximum displacement Center " << param_device.maxDisplaceCenter
           << "\n";
      token = strtok(NULL, " ");
      param_device.GridSpaceCenter = int(atoi(token));
      if (param_device.GridSpaceCenter < 0)
      {
        myError("Negative PIXEL_GRID_CENTER");
      }
      cout << "Grid space displacement center " << param_device.GridSpaceCenter
           << "\n";
      yesMDC = true;
    }
    else if (strcmp(token, "WRITE_PROB_ANGLES") ==
             0) // Key word if writing down each angle probabilities
    {
      token = strtok(NULL, " ");
      param_device.writeAngles = int(atoi(token));
      if (param_device.writeAngles < 0)
      {
        myError("Negative WRITE_PROB_ANGLES");
      }
      cout << "Writing " << param_device.writeAngles
           << " Probabilies of each angle \n";
    }
    else if (strcmp(token, "IGNORE_PDB") == 0) // Ignore PDB extension
    {
      ignorePDB = true;
      cout << "Ignoring PDB extension in model file \n";
    }
    else if (strcmp(token, "NO_PROJECT_RADIUS") ==
             0) // If projecting CA with amino-acid radius
    {
      doaaradius = false;
      cout << "Not Projecting corresponding radius \n";
    }
    else if (strcmp(token, "WRITE_CTF_PARAM") == 0) // Number of Euler angle
                                                    // tripplets in non uniform
                                                    // Euler angle sampling
    {
      writeCTF = true;
      token = strtok(NULL, " ");
      cout << "Writing CTF parameters from PSF parameters that maximize the "
              "posterior. \n";
    }
    else if (strcmp(token, "NO_CENTEROFMASS") == 0) // Number of Euler angle
                                                    // tripplets in non uniform
                                                    // Euler angle sampling
    {
      nocentermass = true;
      cout << "BE CAREFUL CENTER OF MASS IS NOT REMOVED \n Calculated images "
              "might be out of range \n";
    }
    else if (strcmp(token, "PRINT_ROTATED_MODELS") == 0) // Number of Euler
                                                         // angle tripplets in
                                                         // non uniform Euler
                                                         // angle sampling
    {
      printrotmod = true;
      cout << "PRINTING out rotatted models (best for debugging)\n";
    }
    else if (strcmp(token, "NO_MAP_NORM") == 0)
    {
      notnormmap = true;
      cout << "NOT NORMALIZING MAP\n";
    }
    else if (strcmp(token, "PRIOR_MODEL") == 0)
    {
      token = strtok(NULL, " ");
      priorMod = atof(token);
      cout << "MODEL PRIOR Probability " << priorMod << "\n";
    }
    else if (strcmp(token, "PRIOR_ANGLES") == 0)
    {
      yespriorAngles = true;
      cout << "READING Priors for Orientations in additonal orientation file\n";
    }
    else if (strcmp(token, "SHIFT_X") == 0)
    {
      token = strtok(NULL, " ");
      shiftX = atoi(token);
      cout << "Shifting initial model X by " << shiftX << "\n";
    }
    else if (strcmp(token, "SHIFT_Y") == 0)
    {
      token = strtok(NULL, " ");
      shiftY = atoi(token);
      cout << "Shifting initial model Y by " << shiftY << "\n";
    }
    else if (strcmp(token, "SIGMA_PRIOR_B_CTF") == 0)
    {
      token = strtok(NULL, " ");
      param_device.sigmaPriorbctf = atof(token);
      cout << "Chainging  Gaussian width in Prior of Envelope b parameter: "
           << param_device.sigmaPriorbctf << "\n";
    }
    else if (strcmp(token, "SIGMA_PRIOR_DEFOCUS") == 0)
    {
      token = strtok(NULL, " ");
      param_device.sigmaPriordefo = atof(token);
      cout << "Gaussian Width in Prior of defocus parameter: "
           << param_device.sigmaPriordefo << "\n";
    }
    else if (strcmp(token, "PRIOR_DEFOCUS_CENTER") == 0)
    {
      token = strtok(NULL, " ");
      param_device.Priordefcent = atof(token);
      cout << "Gaussian Center in Prior of defocus parameter: "
           << param_device.Priordefcent << "\n";
    }
    else if (strcmp(token, "SIGMA_PRIOR_AMP_CTF") == 0)
    {
      token = strtok(NULL, " ");
      param_device.sigmaPrioramp = atof(token);
      cout << "Gaussian Width in Prior of defocus parameter: "
           << param_device.sigmaPriordefo << "\n";
    }
    else if (strcmp(token, "PRIOR_AMP_CTF_CENTER") == 0)
    {
      token = strtok(NULL, " ");
      param_device.Priorampcent = atof(token);
      cout << "Gaussian Center in Prior of defocus parameter: "
           << param_device.Priordefcent << "\n";
    }
    else if (strcmp(token, "PRINT_ROTATED_MODELS") == 0) // Number of Euler
                                                         // angle tripplets in
                                                         // non uniform Euler
                                                         // angle sampling
    {
      printrotmod = true;
      cout << "Printing out rotated models (best for debugging)\n";
    }
  }
  input.close();

  //************** Checks/Controlls for INPUT

  if (not(yesPixSi))
  {
    myError("Input missing: please provide PIXEL_SIZE");
  }
  if (not(yesNumPix))
  {
    myError("Input missing: please provide NUMBER_PIXELS");
  }
  if (!notuniformangles)
  {
    if (!doquater)
    {
      if (not(yesGPal))
      {
        myError("Input missing: please provide GRIDPOINTS_ALPHA");
      }
      if (not(yesGPbe))
      {
        myError("Input missing: please provide GRIDPOINTS_BETA");
      }
    }
    else if (!yesquatgrid)
    {
      myError("Input missing: please provide GRIDPOINTS_QUATERNION");
    }
  }
  if (not(yesMDC))
  {
    myError("Input missing: please provide grid displacement CENTER");
  }

  cout << "To verify input of Priors:\n";
  cout << "Sigma Prior B-Env: " << param_device.sigmaPriorbctf << "\n";
  cout << "Sigma Prior Defocus: " << param_device.sigmaPriordefo << "\n";
  cout << "Center Prior Defocus: " << param_device.Priordefcent << "\n";

  // PSF or CTF Checks and asigments
  if (usepsf)
  {
    if (not(yesPSFpha))
    {
      myError("Input missing: please provide grid PSF PHASE");
    }
    if (not(yesPSFenv))
    {
      myError("Input missing: please provide grid PSF ENVELOPE");
    }
    if (not(yesAMP))
    {
      myError("Input missing: please provide grid PSF AMPLITUD");
    }
  }
  else
  {
    // cout << "Note - Calculation using CTF values (not PSF). If this is not
    // correct then key word: USE_PSF missing in inputfile**\n";
    if (not(yesBFact))
    {
      myError("Input missing: please provide grid CTF B Env.");
    }
    if (not(yesDefocus))
    {
      myError("Input missing: please provide grid CTF defocus");
    }
    if (not(yesAMP))
    {
      myError("Input missing: please provide grid CTF amplitude");
    }
    // Asigning values of phase according to defocus
    startGridCTF_phase = startDefocus * M_PI * 2.f * 10000 * elecwavel;
    endGridCTF_phase = endDefocus * M_PI * 2.f * 10000 * elecwavel;
    // Asigning values of envelope according to b-envelope (not b-factor)
    startGridEnvelop = startBfactor; // 2.f;
    endGridEnvelop = endBfactor;     // / 2.f;
    param_device.Priordefcent *= M_PI * 2.f * 10000 * elecwavel;
    param_device.sigmaPriordefo *= M_PI * 2.f * 10000 * elecwavel;
  }

  if (elecwavel == 0.019688)
    cout << "Using default electron wave length: 0.019688 (A) of 300kV "
            "microscope\n";

  param_device.NumberFFTPixels1D = param_device.NumberPixels / 2 + 1;
  FFTMapSize = param_device.NumberPixels * param_device.NumberFFTPixels1D;

  nTotParallelMaps = CUDA_FFTS_AT_ONCE;

  if (writeCTF && !usepsf)
  {
    myError("Writing CTF is only valid when integrating over the PSF");
  }

  cout << " +++++++++++++++++++++++++++++++++++++++++ \n";

  return (0);
}

int bioem_param::forprintBest(const char *fileinput)
{
  // **************************************************************************************
  // **********Alternative parameter routine for only printing out a map
  // ******************

  ifstream input(fileinput);
  withnoise = false;
  showrotatemod = false;

  writeCTF = false;
  elecwavel = 0.019866;
  doquater = false;
  nocentermass = false;
  printrotmod = false;
  readquatlist = false;
  doaaradius = true;
  shiftX = 0;
  shiftY = 0;
  stnoise = 1;

  //**** Different keywords! For printing MAP ************
  if (!input.good())
  {
    myError("Opening best parameter file: %s", fileinput);
  }

  delete[] angles;
  angles = new myfloat3_t[1]; // Only best orientation

  char line[512] = {0};
  char saveline[512];
  bool ctfparam = false;

  usepsf = false;

  cout << "\n +++++++++++++++++++++++++++++++++++++++++ \n";
  cout << "\n     ONLY READING BEST PARAMETERS \n";
  cout << "\n     FOR PRINTING MAXIMIZED MAP \n";
  cout << " +++++++++++++++++++++++++++++++++++++++++ \n";
  while (input.getline(line, 512))
  {
    strcpy(saveline, line);
    char *token = strtok(line, " ");

    if (token == NULL || line[0] == '#' || strlen(token) == 0)
    {
      // comment or blank line
    }
    else if (strcmp(token, "PIXEL_SIZE") == 0)
    {
      token = strtok(NULL, " ");
      pixelSize = atof(token);
      if (pixelSize < 0)
      {
        myError("Negative pixel size");
      }
      cout << "Pixel Size " << pixelSize << "\n";
    }
    else if (strcmp(token, "NUMBER_PIXELS") == 0)
    {
      token = strtok(NULL, " ");
      param_device.NumberPixels = int(atoi(token));
      if (param_device.NumberPixels < 0)
      {
        myError("Negative number of pixels");
      }
      cout << "Number of Pixels " << param_device.NumberPixels << "\n";
    }
    else if (strcmp(token, "BEST_ALPHA") == 0)
    {
      token = strtok(NULL, " ");
      angles[0].pos[0] = atof(token);
      cout << "Best Alpha " << angles[0].pos[0] << "\n";
    }
    else if (strcmp(token, "BEST_BETA") == 0)
    {
      token = strtok(NULL, " ");
      angles[0].pos[1] = atof(token);
      cout << "Best beta " << angles[0].pos[1] << "\n";
    }
    else if (strcmp(token, "BEST_GAMMA") == 0)
    {
      token = strtok(NULL, " ");
      angles[0].pos[2] = atof(token);
      cout << "Best Gamma " << angles[0].pos[2] << "\n";
    }
    else if (strcmp(token, "USE_QUATERNIONS") == 0)
    {
      cout << "Orientations with Quaternions. \n";
      doquater = true;
    }
    else if (strcmp(token, "BEST_Q1") == 0)
    {
      token = strtok(NULL, " ");
      angles[0].pos[0] = atof(token);
      cout << "Best q1 " << angles[0].pos[0] << "\n";
    }
    else if (strcmp(token, "BEST_Q2") == 0)
    {
      token = strtok(NULL, " ");
      angles[0].pos[1] = atof(token);
      cout << "Best q2 " << angles[0].pos[1] << "\n";
    }
    else if (strcmp(token, "BEST_Q3") == 0)
    {
      token = strtok(NULL, " ");
      angles[0].pos[2] = atof(token);
      cout << "Best Q3 " << angles[0].pos[2] << "\n";
    }
    else if (strcmp(token, "BEST_Q4") == 0)
    {
      token = strtok(NULL, " ");
      angles[0].quat4 = atof(token);
      cout << "Best Q3 " << angles[0].quat4 << "\n";
    }
    else if (strcmp(token, "USE_PSF") == 0)
    {
      usepsf = true;
      cout << "Important: Using Point Spread Function. Thus, all parameters "
              "are in Real Space. \n";
    }
    else if (strcmp(token, "BEST_PSF_ENVELOPE") == 0)
    {
      token = strtok(NULL, " ");
      startGridEnvelop = atof(token);
      if (startGridEnvelop < 0)
      {
        myError("Negative START_ENVELOPE");
      }
      cout << "Best Envelope PSF " << startGridEnvelop << "\n";
    }
    else if (strcmp(token, "BEST_PSF_PHASE") == 0)
    {
      token = strtok(NULL, " ");
      startGridCTF_phase = atof(token);
      cout << "Best Phase PSF " << startGridCTF_phase << "\n";
    }
    else if (strcmp(token, "BEST_PSF_AMP") == 0)
    {
      token = strtok(NULL, " ");
      startGridCTF_amp = atof(token);
      if (startGridCTF_amp < 0)
      {
        myError("Negative amplitude");
      }
      cout << "Best Amplitude PSF " << startGridCTF_amp << "\n";
    }
    else if (strcmp(token, "BEST_CTF_B_ENV") == 0)
    {
      token = strtok(NULL, " ");
      startGridEnvelop = atof(token); // / 2.f;
      if (startGridEnvelop < 0)
      {
        myError("Negative start B Env.");
      }
      cout << "Best B- Env " << startGridEnvelop << "\n";
      ctfparam = true;
    }
    else if (strcmp(token, "BEST_CTF_DEFOCUS") == 0)
    {
      token = strtok(NULL, " ");
      startGridCTF_phase = atof(token) * M_PI * 2.f * 10000 * elecwavel;
      cout << "Best Defocus " << startGridCTF_phase << "\n";
      ctfparam = true;
    }
    else if (strcmp(token, "BEST_CTF_AMP") == 0)
    {
      token = strtok(NULL, " ");
      startGridCTF_amp = atof(token);
      if (startGridCTF_amp < 0)
      {
        myError("Negative amplitude");
      }
      cout << "Best Amplitude " << startGridCTF_amp << "\n";
      ctfparam = true;
    }
    else if (strcmp(token, "BEST_DX") == 0)
    {
      token = strtok(NULL, " ");
      ddx = atoi(token);
      cout << "Best dx " << ddx << "\n";
    }
    else if (strcmp(token, "BEST_DY") == 0)
    {
      token = strtok(NULL, " ");
      ddy = atoi(token);
      cout << "Best dy " << ddy << "\n";
    }
    else if (strcmp(token, "BEST_NORM") == 0)
    {
      token = strtok(NULL, " ");
      bestnorm = atof(token);
      cout << "Best norm " << bestnorm << "\n";
    }
    else if (strcmp(token, "BEST_OFFSET") == 0)
    {
      token = strtok(NULL, " ");
      bestoff = atof(token);
      cout << "Best offset " << bestoff << "\n";
    }
    else if (strcmp(token, "WITHNOISE") == 0)
    {
      token = strtok(NULL, " ");
      stnoise = atof(token);
      withnoise = true;
      cout << "Including noise with standard deviation " << stnoise << "\n";
    }
    else if (strcmp(token, "NO_PROJECT_RADIUS") ==
             0) // If projecting CA with amino-acid radius
    {
      doaaradius = false;
      cout << "Not projecting corresponding radius \n";
    }
    else if (strcmp(token, "PRINT_ROTATED_MODELS") == 0) // Number of Euler
                                                         // angle tripplets in
                                                         // non uniform Euler
                                                         // angle sampling
    {
      printrotmod = true;
      cout << "Printing out rotated models (best for debugging)\n";
    }
    else if (strcmp(token, "SHIFT_X") == 0)
    {
      token = strtok(NULL, " ");
      shiftX = atoi(token);
      cout << "Shifting initial model X by " << shiftX << "\n";
    }
    else if (strcmp(token, "SHIFT_Y") == 0)
    {
      token = strtok(NULL, " ");
      shiftY = atoi(token);
      cout << "Shifting initial model Y by " << shiftY << "\n";
    }
  }

  if (doquater)
  {
    if (angles[0].quat4 * angles[0].quat4 > 1)
    {
      myError("Quaternion %lf", angles[0].quat4);
    }
    if (angles[0].pos[0] * angles[0].pos[0] > 1)
    {
      myError("Quaternion %lf", angles[0].pos[0]);
    }
    if (angles[0].pos[1] * angles[0].pos[1] > 1)
    {
      myError("Quaternion %lf", angles[0].pos[1]);
    }
    if (angles[0].pos[2] * angles[0].pos[2] > 1)
    {
      myError("Quaternion %lf", angles[0].pos[2]);
    }
  }

  input.close();

  if (usepsf && ctfparam)
  {
    myError("Inconsitent input: using both PSF and CTF?");
  }

  // Automatic definitions
  numberGridPointsCTF_amp = 1;
  gridCTF_amp = startGridCTF_amp;
  numberGridPointsCTF_phase = 1;
  gridCTF_phase = startGridCTF_phase;
  numberGridPointsEnvelop = 1;
  gridEnvelop = startGridEnvelop;
  // doquater = false;

  param_device.NumberFFTPixels1D = param_device.NumberPixels / 2 + 1;
  FFTMapSize = param_device.NumberPixels * param_device.NumberFFTPixels1D;

  nTotParallelMaps = CUDA_FFTS_AT_ONCE;

  return 0;
}

void bioem_param::PrepareFFTs()
{
  //********** PREPARING THE PLANS FOR THE FFTS ******************
  if (mpi_rank == 0)
    cout << "Preparing FFTs\n";
  releaseFFTPlans();
  mycomplex_t *tmp_map, *tmp_map2;
  tmp_map = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) *
                                          param_device.NumberPixels *
                                          param_device.NumberPixels);
  tmp_map2 = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) *
                                           param_device.NumberPixels *
                                           param_device.NumberPixels);
  Alignment = 64;

  fft_plan_c2c_forward = myfftw_plan_dft_2d(
      param_device.NumberPixels, param_device.NumberPixels, tmp_map, tmp_map2,
      FFTW_FORWARD, FFTW_MEASURE | FFTW_DESTROY_INPUT);
  fft_plan_c2c_backward = myfftw_plan_dft_2d(
      param_device.NumberPixels, param_device.NumberPixels, tmp_map, tmp_map2,
      FFTW_BACKWARD, FFTW_MEASURE | FFTW_DESTROY_INPUT);
  fft_plan_r2c_forward = myfftw_plan_dft_r2c_2d(
      param_device.NumberPixels, param_device.NumberPixels,
      (myfloat_t *) tmp_map, tmp_map2, FFTW_MEASURE | FFTW_DESTROY_INPUT);
  fft_plan_c2r_backward = myfftw_plan_dft_c2r_2d(
      param_device.NumberPixels, param_device.NumberPixels, tmp_map,
      (myfloat_t *) tmp_map2, FFTW_MEASURE | FFTW_DESTROY_INPUT);

  if (fft_plan_c2c_forward == 0 || fft_plan_c2c_backward == 0 ||
      fft_plan_r2c_forward == 0 || fft_plan_c2r_backward == 0)
  {
    myError("Planning FFTs");
  }

  myfftw_free(tmp_map);
  myfftw_free(tmp_map2);

  const int count = omp_get_max_threads();
  fft_scratch_complex = new mycomplex_t *[count];
  fft_scratch_real = new myfloat_t *[count];
#pragma omp parallel
  {
#pragma omp critical
    {
      const int i = omp_get_thread_num();
      fft_scratch_complex[i] = (mycomplex_t *) myfftw_malloc(
          sizeof(mycomplex_t) * param_device.NumberPixels *
          param_device.NumberFFTPixels1D);
      fft_scratch_real[i] = (myfloat_t *) myfftw_malloc(
          sizeof(myfloat_t) * param_device.NumberPixels *
          param_device.NumberPixels);
    }
  }

  fft_plans_created = 1;
}

void bioem_param::releaseFFTPlans()
{
  if (fft_plans_created)
  {
    const int count = omp_get_max_threads();
    for (int i = 0; i < count; i++)
    {
      myfftw_free(fft_scratch_complex[i]);
      myfftw_free(fft_scratch_real[i]);
    }
    delete[] fft_scratch_complex;
    delete[] fft_scratch_real;

    myfftw_destroy_plan(fft_plan_c2c_forward);
    myfftw_destroy_plan(fft_plan_c2c_backward);
    myfftw_destroy_plan(fft_plan_r2c_forward);
    myfftw_destroy_plan(fft_plan_c2r_backward);
    myfftw_cleanup();
  }
  fft_plans_created = 0;
}

int bioem_param::CalculateGridsParam(
    const char *fileangles) // TO DO FOR QUATERNIONS
{
  // **************************************************************************************
  // **************** Routine that pre-calculates Orientation
  // Grids**********************
  // ************************************************************************************
  // check if no angles will be output and prompt the user if this was their intention. 
  if ( param_device.writeAngles == 0) 
  {
      printf ("Important, if you want BioEM to output angle probabilities after it has finished, you need to set WRITE_PROB_ANGLES to an integer in the parameter file. These probilities are needed for subsequent rounds of BioEM.\n");
  }
  printf ("The number of angle probabilities written will be %d.", param_device.writeAngles);

  if (!doquater)
  {
    //*********** With Euler angles *******************
    cout << "Analysis Using Default Euler Angles\n";
    if (!notuniformangles)
    {

      if (yespriorAngles)
      {
        myError("This option is not valid with prior for orientations."
                "Please provide separate file with orientations and priors");
      }

      cout << "Calculating Grids in Euler Angles\n";

      myfloat_t grid_alpha, cos_grid_beta;
      int n = 0;

      // alpha and gamma are uniform in -PI,PI
      grid_alpha = 2.f * M_PI / (myfloat_t) angleGridPointsAlpha;

      // cosine beta is uniform in -1,1
      cos_grid_beta = 2.f / (myfloat_t) angleGridPointsBeta;

      // Euler Angle Array
      angles =
          (myfloat3_t *) mallocchk(angleGridPointsAlpha * angleGridPointsBeta *
                                   angleGridPointsAlpha * sizeof(myfloat3_t));

      for (int ialpha = 0; ialpha < angleGridPointsAlpha; ialpha++)
      {
        for (int ibeta = 0; ibeta < angleGridPointsBeta; ibeta++)
        {
          for (int igamma = 0; igamma < angleGridPointsAlpha; igamma++)
          {
            angles[n].pos[0] =
                (myfloat_t) ialpha * grid_alpha - M_PI +
                grid_alpha * 0.5f; // ALPHA centered in the middle
            angles[n].pos[1] =
                acos((myfloat_t) ibeta * cos_grid_beta - 1 +
                     cos_grid_beta * 0.5f); // BETA centered in the middle
            angles[n].pos[2] =
                (myfloat_t) igamma * grid_alpha - M_PI +
                grid_alpha * 0.5f; // GAMMA centered in the middle
            angles[n].quat4 = 0.0;
            n++;
          }
        }
      }
      nTotGridAngles = n;
      voluang = grid_alpha * grid_alpha * cos_grid_beta / (2.f * M_PI) /
                (2.f * M_PI) / 2.f * priorMod;
    }
    else
    {
      //************ Reading Euler Angles From File **************************
      ifstream input(fileangles);

      if (!input.good())
      {
        myError("Euler angle file failed to open file %s", fileangles);
      }

      char line[512] = {0};

      // First line tells the number of rows
      input.getline(line, 512);

      char tmpVals[36] = {0};

      strncpy(tmpVals, line, 12);
      mySscanf(1, tmpVals, "%d", &NotUn_angles);
      cout << "Number of Euler angles " << NotUn_angles << "\n";

      if (NotUn_angles < 1)
      {
        myError("Euler angles not defined in input file");
        //      cout << "Use key word: NOT_UNIFORM_TOTAL_ANGS\n";
      }

      angles = (myfloat3_t *) mallocchk(NotUn_angles * sizeof(myfloat3_t));
      if (yespriorAngles)
      {
        delete[] angprior;
        angprior = new myfloat_t[NotUn_angles];
      }

      int n = 0;
      while (input.getline(line, 512))
      {
        float a, b, g;
        char tmpVals[60] = {0};

        strncpy(tmpVals, line, 12);
        mySscanf(1, tmpVals, "%f", &a);

        strncpy(tmpVals, line + 12, 12);
        mySscanf(1, tmpVals, "%f", &b);

        strncpy(tmpVals, line + 24, 12);
        mySscanf(1, tmpVals, "%f", &g);

        if (yespriorAngles)
        {
          float pp;
          strncpy(tmpVals, line + 36, 12);
          mySscanf(1, tmpVals, "%f", &pp);
          if (pp < 0.0000001)
            cout << "Sure your input is correct? Very small prior.\n";
          angprior[n] = (myfloat_t) pp;
        }

        angles[n].pos[0] = (myfloat_t) a;
        angles[n].pos[1] = (myfloat_t) b;
        angles[n].pos[2] = (myfloat_t) g;
        angles[n].quat4 = 0.0; // just to be sure */
#ifdef DEBUG
        //  if(yespriorAngles)
        cout << "check orient: " << n << " "
             << " " << angles[n].pos[0] << " " << angles[n].pos[1] << " "
             << angles[n].pos[2] << " prior:\n "; // << angprior[n]<< "\n";
#endif
        n++;
        if (NotUn_angles < n)
        {
          myError("Not properly defined total Euler angles %d instead of %d", n,
                  NotUn_angles);
        }
      }
      if (NotUn_angles > n)
      {
        myError("Less quaternions than expected in header %d instead of %d", n,
                NotUn_angles);
      }
      nTotGridAngles = NotUn_angles;
      voluang = 1. / (myfloat_t) NotUn_angles * priorMod;
      input.close();
    }
  }
  else
  {
    //************** Analysis with Quaternions
    if (!notuniformangles)
    {
      //************* Grid of Quaternions *******************
      cout << "Calculating Grids in Quaterions\n ";

      if (yespriorAngles)
      {
        myError("This option is not valid with prior for orientations. "
                "It is necessary to provide a separate file with the "
                "angles and the priors");
      }

      if (GridPointsQuatern < 0)
      {
        myError("Missing gridpoints quaternions. After QUATERNIONS "
                "(int). (int)=Number of gridpoins per dimension");
      }

      myfloat_t dgridq, q1, q2, q3;
      int n = 0;

      dgridq = 2.f / (myfloat_t)(GridPointsQuatern + 1);

      // loop to calculate the number ofpoints in the quaternion shpere  rad < 1
      for (int ialpha = 0; ialpha < GridPointsQuatern + 1; ialpha++)
      {
        q1 = (myfloat_t) ialpha * dgridq - 1.f + 0.5 * dgridq;
        for (int ibeta = 0; ibeta < GridPointsQuatern + 1; ibeta++)
        {
          q2 = (myfloat_t) ibeta * dgridq - 1.f + 0.5 * dgridq;
          for (int igamma = 0; igamma < GridPointsQuatern + 1; igamma++)
          {
            q3 = (myfloat_t) igamma * dgridq - 1.f + 0.5 * dgridq;
            if (q1 * q1 + q2 * q2 + q3 * q3 <= 1.f)
              n = n + 2;
          }
        }
      }

      // allocating angles
      nTotGridAngles = n;
      angles = (myfloat3_t *) mallocchk(nTotGridAngles * sizeof(myfloat3_t));
      voluang = dgridq * dgridq * dgridq * priorMod;
      n = 0;

      // assigning values
      for (int ialpha = 0; ialpha < GridPointsQuatern + 1; ialpha++)
      {
        q1 = (myfloat_t) ialpha * dgridq - 1.f + 0.5 * dgridq;
        for (int ibeta = 0; ibeta < GridPointsQuatern + 1; ibeta++)
        {
          q2 = (myfloat_t) ibeta * dgridq - 1.f + 0.5 * dgridq;
          for (int igamma = 0; igamma < GridPointsQuatern + 1; igamma++)
          {
            q3 = (myfloat_t) igamma * dgridq - 1.f + 0.5 * dgridq;
            if (q1 * q1 + q2 * q2 + q3 * q3 <= 1.f)
            {
              angles[n].pos[0] = q1;
              angles[n].pos[1] = q2;
              angles[n].pos[2] = q3;
              angles[n].quat4 = sqrt(1.f - q1 * q1 - q2 * q2 - q3 * q3);
              n++;
              // Adding the negative
              angles[n].pos[0] = q1;
              angles[n].pos[1] = q2;
              angles[n].pos[2] = q3;
              angles[n].quat4 = -sqrt(1.f - q1 * q1 - q2 * q2 - q3 * q3);
              n++;
            }
          }
        }
      }
    }
    else
    {
      //******** Reading Quaternions From a File ***************************
      ifstream input(fileangles);

      if (!input.good())
      {
        myError("Quaterion list file %s", fileangles);
      }

      char line[512] = {0};

      // First line tels the number of rows
      input.getline(line, 512);
      int ntotquat;
      char tmpVals[60] = {0};

      strncpy(tmpVals, line, 12);
      mySscanf(1, tmpVals, "%d", &ntotquat);
      if (ntotquat < 1)
      {
        myError("Invalid number of quaternions %d", ntotquat);
      }
      else
      {
        cout << "Number of quaternions " << ntotquat << "\n";
      }
      angles = (myfloat3_t *) mallocchk(ntotquat * sizeof(myfloat3_t));
      //    delete[] angles;
      //    angles = new myfloat3_t[ ntotquat] ;

      if (yespriorAngles)
      {
        delete[] angprior;
        angprior = new myfloat_t[ntotquat];
      }

      int n = 0;
      while (input.getline(line, 512))
      {
        float q1, q2, q3, q4;
        char tmpVals[60] = {0};

        strncpy(tmpVals, line, 12);
        mySscanf(1, tmpVals, "%f", &q1);

        strncpy(tmpVals, line + 12, 12);
        mySscanf(1, tmpVals, "%f", &q2);

        strncpy(tmpVals, line + 24, 12);
        mySscanf(1, tmpVals, "%f", &q3);

        strncpy(tmpVals, line + 36, 12);
        mySscanf(1, tmpVals, "%f", &q4);

        angles[n].pos[0] = q1;
        angles[n].pos[1] = q2;
        angles[n].pos[2] = q3;
        angles[n].quat4 = q4;

        if (q1 < -1 || q1 > 1)
        {
          myError("Reading quaterions from list. "
                  "Value out of range %lf row %d",
                  q1, n);
        }
        if (q2 < -1 || q2 > 1)
        {
          myError("Reading quaterions from list. "
                  "Value out of range %lf row %d",
                  q2, n);
        }
        if (q3 < -1 || q3 > 1)
        {
          myError("Reading quaterions from list. "
                  "Value out of range %lf row %d",
                  q3, n);
        }
        if (q4 < -1 || q4 > 1)
        {
          myError("Reading quaterions from list. "
                  "Value out of range %lf row %d",
                  q4, n);
        }

        if (yespriorAngles)
        {
          float pp;
          strncpy(tmpVals, line + 48, 12);
          mySscanf(1, tmpVals, "%f", &pp);
          if (pp < 0.0000001)
            cout << "Sure your input is correct? Very small prior.\n";
          angprior[n] = pp;
        }
#ifdef DEBUG
        //    if(yespriorAngles)
        cout << "check orient: " << n << " " << angles[n].pos[0] << " "
             << angles[n].pos[1] << " " << angles[n].pos[2]
             << " prior: " << angles[n].quat4 << "\n";
#endif
        n++;
        if (ntotquat < n)
        {
          myError("More quaternions than expected in header %d instead of %d",
                  n, NotUn_angles);
        }
      }
      if (ntotquat > n)
      {
        myError("Less quaternions than expected in header %d instead of %d", n,
                NotUn_angles);
      }
      nTotGridAngles = ntotquat;
      voluang = 1. / (myfloat_t) ntotquat * priorMod;

      input.close();
    }

    cout << "Analysis with Quaternions. Total number of quaternions "
         << nTotGridAngles << "\n";
  }

  return (0);
}

int bioem_param::CalculateRefCTF()
{
  // **************************************************************************************
  // ********** Routine that pre-calculates Kernels for Convolution
  // **********************
  // ************************************************************************************

  myfloat_t amp, env, phase, ctf, radsq;
  myfloat_t *localCTF;
  mycomplex_t *localout;
  int nctfmax = param_device.NumberPixels / 2;
  int n = 0;

  localCTF = (myfloat_t *) myfftw_malloc(sizeof(myfloat_t) *
                                         param_device.NumberPixels *
                                         param_device.NumberPixels);
  localout = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) *
                                           param_device.NumberPixels *
                                           param_device.NumberFFTPixels1D);

  nTotCTFs = numberGridPointsCTF_amp * numberGridPointsCTF_phase *
             numberGridPointsEnvelop;
  delete[] refCTF;
  refCTF = new mycomplex_t[getRefCtfCount()];
  delete[] CtfParam;
  CtfParam = new myfloat3_t[getCtfParamCount()];

  myfloat_t normctf;

  gridCTF_amp =
      (endGridCTF_amp - startGridCTF_amp) / (myfloat_t) numberGridPointsCTF_amp;
  gridCTF_phase = (endGridCTF_phase - startGridCTF_phase) /
                  (myfloat_t) numberGridPointsCTF_phase;
  gridEnvelop =
      (endGridEnvelop - startGridEnvelop) / (myfloat_t) numberGridPointsEnvelop;

  // if only one grid point for PSF kernel:
  if ((myfloat_t) numberGridPointsCTF_amp == 1)
  {
    gridCTF_amp = startGridCTF_amp;
  }
  else if ((endGridCTF_amp - startGridCTF_amp) < 0.)
  {
    myError("Interval of amplitude in CTF/PSF negative");
  }
  if ((myfloat_t) numberGridPointsCTF_phase == 1)
  {
    gridCTF_phase = startGridCTF_phase;
  }
  else if ((endGridCTF_phase - startGridCTF_phase) < 0.)
  {
    myError("Interval of phase in CTF/PSF is negative");
  }
  if ((myfloat_t) numberGridPointsEnvelop == 1)
  {
    gridEnvelop = startGridEnvelop;
  }
  else if ((endGridEnvelop - startGridEnvelop) < 0.)
  {
    myError("Interval of envelope in CTF/PSF is negative");
  }

  // More checks with input parameters
  // Envelope should not have a standard deviation greater than Npix/2
  if (sqrt(1. / ((myfloat_t) numberGridPointsEnvelop * gridEnvelop +
                 startGridEnvelop)) > float(param_device.NumberPixels) / 2.0 &&
      usepsf)
  {
    myError("MAX standard deviation of envelope is larger than allowed "
            "KERNEL length");
  }
  // Envelope param should be positive
  if (!printModel && (startGridCTF_amp < 0 || endGridCTF_amp > 1))
  {
    myError("PSF amplitude should be between 0 and 1. "
            "start: %lf end: %lf",
            startGridCTF_amp, endGridCTF_amp);
  }

  if (!printModel && endGridCTF_amp < startGridCTF_amp)
  {

    myError("Values of amplitude starting is larger than ending points."
            "start: %lf end: %lf",
            startGridCTF_amp, endGridCTF_amp);
  }

  for (int iamp = 0; iamp < numberGridPointsCTF_amp;
       iamp++) // Loop over amplitud
  {
    amp = (myfloat_t) iamp * gridCTF_amp + startGridCTF_amp;

    for (int iphase = 0; iphase < numberGridPointsCTF_phase;
         iphase++) // Loop over phase
    {
      phase = (myfloat_t) iphase * gridCTF_phase + startGridCTF_phase;

      for (int ienv = 0; ienv < numberGridPointsEnvelop;
           ienv++) // Loop over envelope
      {
        env = (myfloat_t) ienv * gridEnvelop + startGridEnvelop;

        memset(localCTF, 0,
               param_device.NumberPixels * param_device.NumberPixels *
                   sizeof(myfloat_t));

        normctf = 0.0;

        //     cout <<"values " << amp << " " << phase << " " << env
        //<<"\n";
        // Complex CTF
        mycomplex_t *curRef = &refCTF[n * FFTMapSize];

        // Initialzing everything to zero just to be sure
        for (int i = 0;
             i < param_device.NumberPixels * param_device.NumberFFTPixels1D;
             i++)
        {
          curRef[i][0] = 0.f;
          curRef[i][1] = 0.f;
        }

        for (int i = 0; i < param_device.NumberPixels; i++)
        {
          for (int j = 0; j < param_device.NumberPixels; j++)
          {
            localCTF[i * param_device.NumberPixels + j] = 0.f;
          }
        }

        if (usepsf)
        {
          normctf = 0.0;

          for (int i = 0; i < param_device.NumberPixels; i++)
          {
            for (int j = 0; j < param_device.NumberPixels; j++)
            {
              int ri = 0, rj = 0;

              // Calculating the distance from the periodic center at 0,0

              if (i < nctfmax + 1)
              {
                ri = i;
              }
              else
              {
                ri = param_device.NumberPixels - i;
              };
              if (j < nctfmax + 1)
              {
                rj = j;
              }
              else
              {
                rj = param_device.NumberPixels - j;
              };
              radsq = (myfloat_t)((ri) * (ri) + (rj) * (rj)) * pixelSize *
                      pixelSize;

              ctf = exp(-radsq * env / 2.0) *
                    (-amp * cos(radsq * phase / 2.0) -
                     sqrt((1 - amp * amp)) * sin(radsq * phase / 2.0));

              localCTF[i * param_device.NumberPixels + j] = (myfloat_t) ctf;

              normctf += localCTF[i * param_device.NumberPixels + j];

              //  cout << "TT " << i << " " << j << " " << localCTF[i
              //* param_device.NumberPixels + j]  << "\n";
            }
          }

          // Normalization
          for (int i = 0; i < param_device.NumberPixels; i++)
          {
            for (int j = 0; j < param_device.NumberPixels; j++)
            {
              localCTF[i * param_device.NumberPixels + j] =
                  localCTF[i * param_device.NumberPixels + j] / normctf;
            }
          }

          // Calling FFT_Forward
          myfftw_execute_dft_r2c(fft_plan_r2c_forward, localCTF, localout);

          // Saving the Reference PSFs

          for (int i = 0;
               i < param_device.NumberPixels * param_device.NumberFFTPixels1D;
               i++)
          {
            curRef[i][0] = localout[i][0];
            curRef[i][1] = localout[i][1];
            // cout << "PSFFOU " << i << " " << curRef[i][0] << " " <<
            // curRef[i][1] << " " << param_device.NumberFFTPixels1D << " " <<
            // FFTMapSize <<"\n";
          }
        }
        else
        {

          //*******CTF*************
          normctf = 0.0;

          if (amp < 0.0000000001)
          {
            myError("CTF normalization AMP less than threshold < 10^-10");
          }

          // Directly calculating CTF IN FOURIER SPACE
          for (int i = 0; i < param_device.NumberFFTPixels1D; i++)
          {
            for (int j = 0; j < param_device.NumberFFTPixels1D; j++)
            {
              radsq = (myfloat_t)(i * i + j * j) / param_device.NumberPixels /
                      param_device.NumberPixels / pixelSize / pixelSize;
              ctf = exp(-env * radsq / 2.) *
                    (-amp * cos(phase * radsq / 2.) -
                     sqrt((1 - amp * amp)) * sin(phase * radsq / 2.));
              if (i == 0 && j == 0)
                normctf =
                    (myfloat_t) ctf; // component 0 0 should be the norm in 1d
              curRef[i * param_device.NumberFFTPixels1D + j][0] = ctf / normctf;
              curRef[i * param_device.NumberFFTPixels1D + j][1] = 0;
              // On symmetric side
              curRef[(param_device.NumberPixels - i - 1) *
                         param_device.NumberFFTPixels1D +
                     j][0] = ctf / normctf;
              curRef[(param_device.NumberPixels - i - 1) *
                         param_device.NumberFFTPixels1D +
                     j][1] = 0;
            }
          }

          // for(int i = 0; i < param_device.NumberPixels *
          // param_device.NumberFFTPixels1D; i++ )curRef[i][0]/= normctf;
        }

        CtfParam[n].pos[0] = amp;
        CtfParam[n].pos[1] = phase;
        CtfParam[n].pos[2] = env;
        n++;
        // Exit(1);
      }
    }
  }

  myfftw_free(localCTF);
  myfftw_free(localout);
  if (nTotCTFs != n)
  {
    myError("Internal during CTF preparation");
  }

  // ********** Calculating normalized volume element *********

  if (!printModel)
  {
    // All priors (uniform or not) normalized to 1
    // The volume is the grid-spacing of the parameter / normalization
    // the order is angles, displacement, ctf amplitud (all uniform) then env b
    // & phase (non uniform) the sqrt(2) cancel out (see SI)
    param_device.volu =
        voluang * (myfloat_t) param_device.GridSpaceCenter * pixelSize *
        (myfloat_t) param_device.GridSpaceCenter * pixelSize /
        ((2.f * (myfloat_t) param_device.maxDisplaceCenter + 1.)) /
        (2.f * (myfloat_t)(param_device.maxDisplaceCenter + 1.)) /
        (myfloat_t) numberGridPointsCTF_amp * gridEnvelop * gridCTF_phase /
        4.f / M_PI / sqrt(2.f * M_PI) / param_device.sigmaPriorbctf /
        param_device.sigmaPriordefo / param_device.sigmaPrioramp;

    //  cout << "VOLU " << param_device.volu  << " " << gridCTF_amp << "\n";
    // *** Number of total pixels***

    param_device.Ntotpi =
        (myfloat_t)(param_device.NumberPixels * param_device.NumberPixels);
    param_device.NxDisp = 2 * (int) (param_device.maxDisplaceCenter /
                                     param_device.GridSpaceCenter) +
                          1;
    param_device.NtotDisp = param_device.NxDisp * param_device.NxDisp;
  }
  return (0);
}

bioem_param::~bioem_param()
{
  releaseFFTPlans();
  param_device.NumberPixels = 0;
  angleGridPointsAlpha = 0;
  angleGridPointsBeta = 0;
  numberGridPointsEnvelop = 0;
  numberGridPointsCTF_amp = 0;
  numberGridPointsCTF_phase = 0;
  param_device.maxDisplaceCenter = 0;
  numberGridPointsDisplaceCenter = 0;
  if (refCTF)
    delete[] refCTF;
  if (CtfParam)
    delete[] CtfParam;
  if (angles)
    free(angles);
  if (angprior)
    delete[] angprior;
  refCTF = NULL;
  CtfParam = NULL;
  angles = NULL;
  angprior = NULL;
}
