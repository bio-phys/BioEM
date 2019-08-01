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

   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   Note: This program contains subroutine "read_MRC" of the Situs 2.7.2 program.
   Ref: Willy Wriggers. Using Situs for the Integration of Multi-Resolution
   Structures.
   Biophysical Reviews, 2010, Vol. 2, pp. 21-27.
   with a GPL lisences version 3.

   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include <cstring>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "bioem.h"
#include "map.h"
#include "mrc.h"
#include "param.h"

using namespace std;

//************** Loading Map from Binary file *******
void bioem_RefMap::readBinaryMaps()
{
  FILE *fp = fopen(FILE_MAPS_DUMP, "rb");
  if (fp == NULL)
  {
    myError("Opening dump file");
  }
  size_t elements_read;
  elements_read = fread(&ntotRefMap, sizeof(ntotRefMap), 1, fp);
  if (elements_read != 1)
  {
    myError("Reading file");
  }
  maps = (myfloat_t *) mallocchk(ntotRefMap * refMapSize * sizeof(myfloat_t));
  elements_read = fread(maps, sizeof(myfloat_t) * refMapSize, ntotRefMap, fp);
  if (elements_read != (size_t) ntotRefMap)
  {
    myError("Reading file");
  }
  fclose(fp);
  cout << "Particle Maps read from Map Dump\n";
}

//************* Dumping Maps *********************
void bioem_RefMap::writeBinaryMaps()
{
  FILE *fp = fopen(FILE_MAPS_DUMP, "w+b");
  if (fp == NULL)
  {
    myError("Opening dump file");
  }
  fwrite(&ntotRefMap, sizeof(ntotRefMap), 1, fp);
  fwrite(maps, sizeof(myfloat_t) * refMapSize, ntotRefMap, fp);
  fclose(fp);
}

//************** Reading MRC file *******
void bioem_RefMap::readMRCMaps(bioem_param &param, const char *filemap)
{
  ntotRefMap = 0;

  if (READ_PARALLEL &&
      readMultMRC) // reading MRC files in parallel (using the new routine)
  {
    //************** Getting list of multiple MRC files *************
    cout << "Opening File with MRC list names: " << filemap << "\n";
    ifstream input(filemap);
    if (!input.good())
    {
      myError("Failed to open file contaning MRC names: %s", filemap);
    }

    char line[512] = {0};
    char mapname[100];
    char tmpm[10] = {0};
    std::vector<string> fileNames;
    int nFiles = 0;

    while (input.getline(line, 512))
    {
      char tmpVals[100] = {0};

      string strline(line);

      // Check if filename ends with .mrc
      size_t foundpos = strline.find("mrc");
      size_t endpos = strline.find_last_not_of(" \t");
      if (foundpos > endpos)
      {
        myWarning("MRC extension NOT detected in file name: %s. "
                  "Are you sure you want to read an MRC?",
                  filemap);
      }

      strncpy(tmpVals, line, 99);
      mySscanf(1, tmpVals, "%99c", mapname);

      // Check for last line
      strncpy(tmpm, mapname, 3);

      if (strcmp(tmpm, "XXX") != 0)
      {
        // Added for parallel read
        fileNames.push_back(strline);
        nFiles++;
      }

      for (int i = 0; i < 3; i++)
        mapname[i] = 'X';
      for (int i = 3; i < 100; i++)
        mapname[i] = 0;
    }

    //************** Reading multiple MRC files *************
    printf("\n+++++++++++++++++++++++++++++++++++++++++++\n");
    // Allocate data structures for each file
    int numMaps = 0;
    int *swap = (int *) mallocchk(sizeof(int) * nFiles);
    int *nc = (int *) mallocchk(sizeof(int) * nFiles);
    int *nr = (int *) mallocchk(sizeof(int) * nFiles);
    int *ns = (int *) mallocchk(sizeof(int) * nFiles);
    int *nsymbt = (int *) mallocchk(sizeof(int) * nFiles);
    int *offsets = (int *) mallocchk(sizeof(int) * nFiles);
    // Parallel read
#pragma omp parallel
    {
      const int num = omp_get_thread_num();
// Get the number of maps in each file
#pragma omp for
      for (int i = 0; i < nFiles; i++)
      {
        printf("Reading Information from MRC: %s (thread %d)\n",
               fileNames[i].c_str(), num);
        check_one_MRC(fileNames[i].c_str(), &swap[i], &nc[i], &nr[i], &ns[i],
                      &nsymbt[i]);
      }
// Allocate maps and compute starting position for each thread
#pragma omp single
      {
        for (int i = 0; i < nFiles; i++)
        {
          offsets[i] = numMaps;
          numMaps += ns[i];
        }
        maps =
            (myfloat_t *) mallocchk(refMapSize * sizeof(myfloat_t) * numMaps);
      }
// Get the number of maps in each file
#pragma omp for
      for (int i = 0; i < nFiles; i++)
      {
        read_one_MRC(fileNames[i].c_str(), param, offsets[i], swap[i], nc[i],
                     nr[i], ns[i], nsymbt[i]);
      }
    }

    // Cleanup
    fileNames.clear();
    free(swap);
    free(nc);
    free(nr);
    free(ns);
    free(nsymbt);
    free(offsets);
    ntotRefMap = numMaps;

    cout << "\n+++++++++++++++++++++++++++++++++++++++++++ \n";
    cout << "Particle Maps read from MULTIPLE MRC Files in: " << filemap
         << "\n";
  }
  else // reading MRC files in sequentially (using the old routine)
  {
    if (readMultMRC)
    {
      //************** Reading multiple MRC files *************
      cout << "Opening File with MRC list names: " << filemap << "\n";
      ifstream input(filemap);

      if (!input.good())
      {
        myError("Failed to open file contaning MRC names: %s", filemap);
      }

      char line[512] = {0};
      char mapname[100];
      char tmpm[10] = {0};
      const char *indifile;

      while (input.getline(line, 512))
      {
        char tmpVals[100] = {0};

        string strline(line);

        // cout << "MRC File name:" << strline << "\n";

        strncpy(tmpVals, line, 99);
        mySscanf(1, tmpVals, "%99c", mapname);

        // Check for last line
        strncpy(tmpm, mapname, 3);

        if (strcmp(tmpm, "XXX") != 0)
        {
          indifile = strline.c_str();

          //   size_t foundpos= strline.find("mrc");
          //   size_t endpos = strline.find_last_not_of(" \t");

          // Reading multiple MRC
          read_MRC(indifile, param);
        }
        for (int i = 0; i < 3; i++)
          mapname[i] = 'X';
        for (int i = 3; i < 100; i++)
          mapname[i] = 0;
      }
      cout << "\n+++++++++++++++++++++++++++++++++++++++++++ \n";
      cout << "Particle Maps read from MULTIPLE MRC Files in: " << filemap
           << "\n";
    }
    else
    {

      string strfilename(filemap);

      size_t foundpos = strfilename.find("mrc");
      size_t endpos = strfilename.find_last_not_of(" \t");

      if (foundpos > endpos)
      {
        myWarning("MRC extension NOT detected in file name: %s. "
                  "Are you sure you want to read an MRC?",
                  filemap);
      }

      read_MRC(filemap, param);
      cout << "\n++++++++++++++++++++++++++++++++++++++++++ \n";
      cout << "Particle Maps read from ONE MRC File: " << filemap << "\n";
    }
  }
}

//************** Reading Text file *************
void bioem_RefMap::readTextMaps(bioem_param &param, const char *filemap)
{
  if (READ_PARALLEL) // reading textual file in parallel
  {
    FILE *file = fopen(filemap, "r");
    if (file == NULL)
    {
      myError("Opening file: %s", filemap);
    }

    long lSize, size;
    char *buffer;

    // Obtain file size
    fseek(file, 0, SEEK_END);
    lSize = ftell(file);
    rewind(file);

    // Allocate memory to contain the whole file
    buffer = (char *) mallocchk(sizeof(char) * lSize);
    if (buffer == NULL)
    {
      myError("Memory error");
    }

    // Copy the file into the buffer
    size = fread(buffer, 1, lSize, file);
    if (size != lSize)
    {
      myError("Reading error")
    }

    // Checking that the file starts with "PARTICLE"
    char firstToken[9] = {0};
    strncpy(firstToken, buffer, 8);
    if (strcmp(firstToken, "PARTICLE") != 0)
    {
      myError("Missing correct standard map format: PARTICLE HEADER");
    }

    // Parallel read
    int nthreads;
    std::vector<long> *particleStarts;
    int *nlines;
    int *offsets;
    int nummap = 0;
#pragma omp parallel
    {
      nthreads = omp_get_max_threads();
      const int num = omp_get_thread_num();
// Allocate local data structures
#pragma omp single
      {
        particleStarts = new vector<long>[nthreads];
        nlines = (int *) callocchk(sizeof(int) * nthreads);
        offsets = (int *) mallocchk(sizeof(int) * nthreads);
      }
// Get the number of lines and end of line position
#pragma omp for
      for (long i = 0; i < size - 1; i++)
      {
        if (buffer[i] == 'P')
        {
          char token[9] = {0};
          strncpy(token, buffer + i, 8);
          if (strcmp(token, "PARTICLE") == 0)
          {
            nlines[num]++;
            particleStarts[num].push_back(i);
          }
        }
      }
// Allocate points and compute starting position for each thread
#pragma omp single
      {
        for (int i = 0; i < nthreads; i++)
        {
          offsets[i] = nummap;
          nummap += nlines[i];
        }
        maps = (myfloat_t *) mallocchk(refMapSize * sizeof(myfloat_t) * nummap);
      }
// Parallel parsing of the input file
#pragma omp for
      for (int i = 0; i < nthreads; i++)
      {
        int lMap = offsets[i];
        for (std::vector<long>::const_iterator j = particleStarts[i].begin();
             j != particleStarts[i].end(); j++)
        {
          // start of the next line
          long k = *j + 8;
          while (buffer[k] != '\n')
            k++;
          k++;
          // map variables
          char tmpVals[33] = {0};
          int i_coor = 0;
          int j_coor = 0;
          double z = 0.0;
          int countpix = 0;
          // Parse single particle
          while (k < size && buffer[k] != 'P')
          {
            strncpy(tmpVals, buffer + k, 32);
            k += 33;
            mySscanf(3, tmpVals, "%d %d %lf", &i_coor, &j_coor, &z);
            // checking for Map limits
            if (i_coor > -1 && i_coor < numPixels && j_coor > -1 &&
                j_coor < numPixels)
            {
              countpix++;
              maps[lMap * refMapSize + i_coor * numPixels + j_coor] =
                  (myfloat_t) z;
            }
            else
            {
              myError("Reading map (Map number %d, i %d, j %d)", lMap, i_coor,
                      j_coor);
            }
          }
          // Verifying input consistency
          if (i_coor != numPixels - 1 || j_coor != numPixels - 1 ||
              countpix != refMapSize)
          {
            myError("Inconsistent number of pixels in maps and inputfile "
                    "( %d, i %d, j %d)",
                    countpix, i_coor, j_coor);
          }
          // Occasionally print which map is being processed
          if (lMap % 128 == 0)
          {
            printf("...%d (thread %d)\n", lMap, num);
          }
          lMap++;
        }
      }
    }
    // Cleanup
    free(buffer);
    delete[] particleStarts;
    free(nlines);
    free(offsets);

    fclose(file);
    ntotRefMap = nummap;
  }
  else // reading textual file sequentially
  {
    ifstream input(filemap);
    if (!input.good())
    {
      myError("Particle maps failed to open file");
    }

    char line[512] = {0};
    char tmpLine[512] = {0};
    bool first = true;

    int nummap = -1;
    int lasti = 0;
    int lastj = 0;
    int countpix = 0;
    int allocsize = 0;

    while (input.getline(line, 512))
    {
      strncpy(tmpLine, line, strlen(line));
      char *token = strtok(tmpLine, " ");

      if (first)
      {
        if (strcmp(token, "PARTICLE") != 0)
        {
          myError("Missing correct standard map format: PARTICLE HEADER");
        }
        first = false;
      }

      if (strcmp(token, "PARTICLE") == 0) // to count the number of maps
      {
        nummap++;
        countpix = 0;
        if (allocsize == 0)
        {
          allocsize = 64;
          maps = (myfloat_t *) mallocchk(refMapSize * sizeof(myfloat_t) *
                                         allocsize);
        }
        else if (nummap + 1 >= allocsize)
        {
          allocsize *= 2;
          maps = (myfloat_t *) reallocchk(maps, refMapSize * sizeof(myfloat_t) *
                                                    allocsize);
        }
        if (nummap % 128 == 0)
        {
          cout << "..." << nummap << "\n";
        }
        if (lasti + 1 != numPixels && lastj + 1 != numPixels && nummap > 0)
        {
          myError("Inconsistent number of pixels in maps and inputfile "
                  "( %d, i %d, j %d)",
                  numPixels, lasti, lastj);
        }
      }
      else
      {
        int i, j;
        double z;

        char tmpVals[36] = {0};

        strncpy(tmpVals, line, 8);
        mySscanf(1, tmpVals, "%d", &i);

        strncpy(tmpVals, line + 8, 8);
        mySscanf(1, tmpVals, "%d", &j);

        strncpy(tmpVals, line + 16, 16);
        mySscanf(1, tmpVals, "%lf", &z);
        // checking for Map limits
        if (i > -1 && i < numPixels && j > -1 && j < numPixels)
        {
          countpix++;
          maps[nummap * refMapSize + i * numPixels + j] = (myfloat_t) z;
          lasti = i;
          lastj = j;
        }
        else
        {
          myError("Reading map (Map number %d, i %d, j %d)", nummap, i, j);
        }
      }
    }

    if (lasti != numPixels - 1 || lastj != numPixels - 1 ||
        countpix != refMapSize)
    {
      myError("Inconsistent number of pixels in maps and inputfile "
              "( %d, i %d, j %d)",
              countpix, lasti, lastj);
    }
    ntotRefMap = nummap + 1;
    maps = (myfloat_t *) reallocchk(maps, refMapSize * sizeof(myfloat_t) *
                                              ntotRefMap);
  }

  cout << ".";
  cout << "Particle Maps read from Standard File: " << ntotRefMap << "\n";
}

int bioem_RefMap::readRefMaps(bioem_param &param, const char *filemap)
{
  // *************** Reading reference Particle Maps ************
  numPixels = param.param_device.NumberPixels;
  refMapSize = numPixels * numPixels;

  if (param.loadMap)
  {
    readBinaryMaps();
  }
  else if (readMRC)
  {
    readMRCMaps(param, filemap);
  }
  else
  {
    readTextMaps(param, filemap);
  }

  if (param.dumpMap)
  {
    writeBinaryMaps();
  }

  //*********** To Debug with few Maps ********************
  if (getenv("BIOEM_DEBUG_NMAPS"))
  {
    ntotRefMap = atoi(getenv("BIOEM_DEBUG_NMAPS"));
  }
  param.nTotParallelMaps = min(CUDA_FFTS_AT_ONCE, ntotRefMap);

  cout << "Total Number of particles: " << ntotRefMap;
  cout << "\n+++++++++++++++++++++++++++++++++++++++++++ \n";

  return (0);
}

int bioem_RefMap::PreCalculateMapsFFT(bioem_param &param)
{
  // **************************************************************************************
  // ********** Routine that pre-calculates Reference maps FFT for Convolution/
  // Comparison **********************
  // ************************************************************************************

  RefMapsFFT = new mycomplex_t[ntotRefMap * param.FFTMapSize];

#pragma omp parallel for
  for (int iRefMap = 0; iRefMap < ntotRefMap; iRefMap++)
  {
    const int num = omp_get_thread_num();
    myfloat_t *localMap = param.fft_scratch_real[num];
    mycomplex_t *localout = param.fft_scratch_complex[num];

    // Assigning localMap values to padded Map
    for (int i = 0; i < param.param_device.NumberPixels; i++)
    {
      for (int j = 0; j < param.param_device.NumberPixels; j++)
      {
        localMap[i * param.param_device.NumberPixels + j] =
            maps[iRefMap * refMapSize + i * param.param_device.NumberPixels +
                 j];
      }
    }

    // Calling FFT_Forward
    myfftw_execute_dft_r2c(param.fft_plan_r2c_forward, localMap, localout);

    // Saving the Reference CTFs (RefMap array possibly has incorrect alignment,
    // so we copy here. Stupid but in fact does not matter.)
    mycomplex_t *RefMap = &RefMapsFFT[iRefMap * param.FFTMapSize];

    for (int i = 0; i < param.param_device.NumberPixels *
                            param.param_device.NumberFFTPixels1D;
         i++)
    {
      RefMap[i][0] = localout[i][0];
      RefMap[i][1] = localout[i][1];
    }
  }

  return (0);
}

int bioem_RefMap::precalculate(bioem_param &param, bioem &bio)
{
  // **************************************************************************************
  // *******************************Precalculating Routine for
  // Maps************************
  // **************************************************************************************

  sum_RefMap = (myfloat_t *) mallocchk(sizeof(myfloat_t) * ntotRefMap);
  sumsquare_RefMap = (myfloat_t *) mallocchk(sizeof(myfloat_t) * ntotRefMap);

// Precalculating cross-correlations of maps
#pragma omp parallel for
  for (int iRefMap = 0; iRefMap < ntotRefMap; iRefMap++)
  {
    myfloat_t sum, sumsquare;
    bio.calcross_cor(getmap(iRefMap), sum, sumsquare);
    // Storing Crosscorrelations in Map class
    sum_RefMap[iRefMap] = sum;
    sumsquare_RefMap[iRefMap] = sumsquare;
  }

  // Precalculating Maps in Fourier space
  PreCalculateMapsFFT(param);
  free(maps);
  maps = NULL;

  return (0);
}

void bioem_Probability::init(size_t maps, size_t angles, bioem &bio)
{
  //********** Initializing pointers *******************
  nMaps = maps;
  nAngles = angles;
  ptr = bio.malloc_device_host(
      get_size(maps, angles, bio.param.param_device.writeAngles));
  if (bio.DebugOutput >= 1)
    cout << "Allocation #Maps " << maps << " #Angles " << angles << "\n";
  set_pointers();
}

void bioem_Probability::copyFrom(bioem_Probability *from, bioem &bio)
{

  bioem_Probability_map &pProbMap = getProbMap(0);
  bioem_Probability_map &pProbMapFrom = from->getProbMap(0);
  memcpy(&pProbMap, &pProbMapFrom, from->nMaps * sizeof(bioem_Probability_map));

  if (bio.param.param_device.writeAngles)
  {
    for (int iOrient = 0; iOrient < nAngles; iOrient++)
    {
      bioem_Probability_angle &pProbAngle = getProbAngle(0, iOrient);
      bioem_Probability_angle &pProbAngleFrom = from->getProbAngle(0, iOrient);
      memcpy(&pProbAngle, &pProbAngleFrom,
             from->nMaps * sizeof(bioem_Probability_angle));
    }
  }
}

int bioem_RefMap::read_MRC(const char *filename, bioem_param &param)
{

  /* 	 subroutine "read_MRC" of the Situs 2.7.2 program.
         Ref: Willy Wriggers. Using Situs for the Integration of
     Multi-Resolution Structures.
         Biophysical Reviews, 2010, Vol. 2, pp. 21-27.*/

  myfloat_t st, st2;
  unsigned long count;
  FILE *fin;
  float currfloat;
  int nc, nr, ns, swap, header_ok = 1;
  float xlen, ylen, zlen;
  int mode, ncstart, nrstart, nsstart, ispg, nsymbt, lskflg;
  float a_tmp, b_tmp, g_tmp;
  int mx, my, mz, mapc, mapr, maps_local;
  float dmin, dmax, dmean;
  int n_range_viol0, n_range_viol1;

  fin = fopen(filename, "rb");
  if (fin == NULL)
  {
    myError("Opening MRC: %s", filename);
  }
  n_range_viol0 = test_mrc(filename, 0);
  n_range_viol1 = test_mrc(filename, 1);

  if (n_range_viol0 < n_range_viol1)
  { //* guess endianism
    swap = 0;
    if (n_range_viol0 > 0)
    {
      myWarning("%i header field range violations detected in file %s",
                n_range_viol0, filename);
    }
  }
  else
  {
    swap = 1;
    if (n_range_viol1 > 0)
    {
      myWarning("%i header field range violations detected in file %s",
                n_range_viol1, filename);
    }
  }
  printf("\n+++++++++++++++++++++++++++++++++++++++++++\n");
  printf("Reading Information from MRC: %s \n", filename);
  header_ok *= read_int(&nc, fin, swap);
  header_ok *= read_int(&nr, fin, swap);
  header_ok *= read_int(&ns, fin, swap);
  header_ok *= read_int(&mode, fin, swap);
  header_ok *= read_int(&ncstart, fin, swap);
  header_ok *= read_int(&nrstart, fin, swap);
  header_ok *= read_int(&nsstart, fin, swap);
  header_ok *= read_int(&mx, fin, swap);
  header_ok *= read_int(&my, fin, swap);
  header_ok *= read_int(&mz, fin, swap);
  header_ok *= read_float(&xlen, fin, swap);
  header_ok *= read_float(&ylen, fin, swap);
  header_ok *= read_float(&zlen, fin, swap);
  header_ok *= read_float(&a_tmp, fin, swap);
  header_ok *= read_float(&b_tmp, fin, swap);
  header_ok *= read_float(&g_tmp, fin, swap);
  header_ok *= read_int(&mapc, fin, swap);
  header_ok *= read_int(&mapr, fin, swap);
  header_ok *= read_int(&maps_local, fin, swap);
  header_ok *= read_float(&dmin, fin, swap);
  header_ok *= read_float(&dmax, fin, swap);
  header_ok *= read_float(&dmean, fin, swap);
  header_ok *= read_int(&ispg, fin, swap);
  header_ok *= read_int(&nsymbt, fin, swap);
  header_ok *= read_int(&lskflg, fin, swap);

  printf("Number Columns  = %8d \n", nc);
  printf("Number Rows     = %8d \n", nr);
  printf("Number Sections = %8d \n", ns);
  printf("MODE = %4d (only data type mode 2: 32-bit)\n", mode);
  printf("NSYMBT = %4d (# bytes symmetry operators)\n", nsymbt);

  /* printf("  NCSTART = %8d  (index of first column, counting from
     0)\n",ncstart);
     printf(">  NRSTART = %8d  (index of first row, counting from
     0)\n",nrstart);
     printf("  NSSTART = %8d  (index of first section, counting from
     0)\n",nsstart);
     printf("       MX = %8d  (# of X intervals in unit cell)\n",mx);
     printf("       MY = %8d  (# of Y intervals in unit cell)\n",my);
     printf("       MZ = %8d  (# of Z intervals in unit cell)\n",mz);
     printf(" X length = %8.3f  (unit cell dimension)\n",xlen);
     printf(" Y length = %8.3f  (unit cell dimension)\n",ylen);
     printf(" Z length = %8.3f  (unit cell dimension)\n",zlen);
     printf("    Alpha = %8.3f  (unit cell angle)\n",a_tmp);
     printf("     Beta = %8.3f  (unit cell angle)\n",b_tmp);
     printf("    Gamma = %8.3f  (unit cell angle)\n",g_tmp);
     printf("     MAPC = %8d  (columns axis: 1=X,2=Y,3=Z)\n",mapc);
     printf("     MAPR = %8d  (rows axis: 1=X,2=Y,3=Z)\n",mapr);
     printf("     MAPS = %8d  (sections axis: 1=X,2=Y,3=Z)\n",maps_local);
     printf("     DMIN = %8.3f  (minimum density value - ignored)\n",dmin);
     printf("     DMAX = %8.3f  (maximum density value - ignored)\n",dmax);
     printf("    DMEAN = %8.3f  (mean density value - ignored)\n",dmean);
     printf("     ISPG = %8d  (space group number - ignored)\n",ispg);
     printf("   NSYMBT = %8d  (# bytes storing symmetry operators)\n",nsymbt);
     printf("   LSKFLG = %8d  (skew matrix flag: 0:none,
     1:follows)\n",lskflg);*/

  if (header_ok == 0)
  {
    myError("Reading MRC header: %s", filename);
  }

  if (nr != param.param_device.NumberPixels ||
      nc != param.param_device.NumberPixels)
  {
    myError("Inconsistent number of pixels in maps and inputfile "
            "( %d, i %d, j %d)",
            param.param_device.NumberPixels, nc, nr);
  }

  if (ntotRefMap == 0)
  {
    maps = (myfloat_t *) mallocchk(refMapSize * sizeof(myfloat_t) * ns);
  }
  else
  {
    maps = (myfloat_t *) reallocchk(maps, refMapSize * sizeof(myfloat_t) *
                                              (ntotRefMap + ns));
  }

  if (mode != 2)
  {
    myError("MRC mode: %d. Currently mode 2 is the only one allowed", mode);
  }
  else
  {
    rewind(fin);
    for (count = 0; count < 256; ++count)
      if (read_float_empty(fin) == 0)
      {
        myError("Converting Data: %s", filename);
      }

    for (count = 0; count < (unsigned long) nsymbt; ++count)
      if (read_char_float(&currfloat, fin) == 0)
      {
        myError("Converting Data: %s", filename);
      }

    for (int nmap = 0; nmap < ns; nmap++)
    {
      st = 0.0;
      st2 = 0.0;
      for (int j = 0; j < nr; j++)
        for (int i = 0; i < nc; i++)
        {
          if (read_float(&currfloat, fin, swap) == 0)
          {
            myError("Converting Data: %s", filename);
          }
          else
          {
            maps[(nmap + ntotRefMap) * refMapSize + i * numPixels + j] =
                (myfloat_t) currfloat;
            st += currfloat;
            st2 += currfloat * currfloat;
          }
        }
      // Normaling maps to zero mean and unit standard deviation
      if (!param.notnormmap)
      {
        st /= float(nr * nc);
        st2 = sqrt(st2 / float(nr * nc) - st * st);
        for (int j = 0; j < nr; j++)
          for (int i = 0; i < nc; i++)
          {
            maps[(nmap + ntotRefMap) * refMapSize + i * numPixels + j] =
                maps[(nmap + ntotRefMap) * refMapSize + i * numPixels + j] /
                    st2 -
                st / st2;
            // cout <<"MAP:: " << i << " " << j << " " <<  maps[(nmap +
            // ntotRefMap) * refMapSize + i * numPixels + j]  << "\n";
          }
      }
    }
    ntotRefMap += ns;
    //  cout << ntotRefMap << "\n";
  }
  fclose(fin);

  return (0);
}

int bioem_RefMap::read_one_MRC(const char *filename, bioem_param &param,
                               int offset, int swap, int nc, int nr, int ns,
                               int nsymbt)
{
  // Partially:
  /*     subroutine "read_MRC" of the Situs 2.7.2 program.
         Ref: Willy Wriggers. Using Situs for the Integration of
     Multi-Resolution Structures.
         Biophysical Reviews, 2010, Vol. 2, pp. 21-27.*/

  myfloat_t st, st2;
  FILE *fin;
  float currfloat;
  long start = offset * refMapSize;

  fin = fopen(filename, "rb");
  if (fin == NULL)
  {
    myError("Opening MRC: %s", filename);
  }
  for (int count = 0; count < 256; ++count)
  {
    if (read_float_empty(fin) == 0)
    {
      myError("Converting Data: %s", filename);
    }
  }
  for (long count = 0; count < (long) nsymbt; ++count)
  {
    if (read_char_float(&currfloat, fin) == 0)
    {
      myError("Converting Data: %s", filename);
    }
  }
  if (nr != numPixels || nc != numPixels)
  {
    myError("Inconsistent number of pixels in maps and inputfile "
            "( %d, i %d, j %d)",
            numPixels, nc, nr);
  }

  // Actual reading of the data
  for (int nmap = 0; nmap < ns; nmap++)
  {
    st = 0.0;
    st2 = 0.0;
    for (int j = 0; j < nr; j++)
    {
      for (int i = 0; i < nc; i++)
      {
        if (read_float(&currfloat, fin, swap) == 0)
        {
          myError("Converting Data: %s", filename);
        }
        else
        {
          long address = start + nmap * refMapSize + i * numPixels + j;
          maps[address] = (myfloat_t) currfloat;
          st += currfloat;
          st2 += currfloat * currfloat;
        }
      }
    }
    // Normaling maps to zero mean and unit standard deviation
    if (!param.notnormmap)
    {
      st /= float(nr * nc);
      st2 = sqrt(st2 / float(nr * nc) - st * st);
      for (int j = 0; j < nr; j++)
      {
        for (int i = 0; i < nc; i++)
        {
          long address = start + nmap * refMapSize + i * numPixels + j;
          maps[address] = maps[address] / st2 - st / st2;
        }
      }
    }
  }

  fclose(fin);
  return (0);
}
