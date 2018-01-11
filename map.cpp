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

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "bioem.h"
#include "map.h"
#include "param.h"

using namespace std;

int bioem_RefMap::readRefMaps(bioem_param &param, const char *filemap)
{
  numPixels = param.param_device.NumberPixels;
  refMapSize =
      param.param_device.NumberPixels * param.param_device.NumberPixels;
  // **************************************************************************************
  // ***********************Reading reference Particle
  // Maps************************
  // **************************************************************************************
  int allocsize = 0;
  if (param.loadMap)
  {
    //************** Loading Map from Binary file *******
    FILE *fp = fopen("maps.dump", "rb");
    if (fp == NULL)
    {
      cout << "Error opening dump file\n";
      exit(1);
    }
    size_t elements_read;
    elements_read = fread(&ntotRefMap, sizeof(ntotRefMap), 1, fp);
    if (elements_read != 1)
    {
      cout << "Error reading file\n";
      exit(1);
    }
    maps = (myfloat_t *) mallocchk(ntotRefMap * refMapSize * sizeof(myfloat_t));
    elements_read = fread(maps, sizeof(myfloat_t) * refMapSize, ntotRefMap, fp);
    if (elements_read != (size_t) ntotRefMap)
    {
      cout << "Error reading file\n";
      exit(1);
    }

    fclose(fp);

    cout << "Particle Maps read from Map Dump\n";
  }
  else if (readMRC)
  {
    //************** Reading MRC file *******
    ntotRefMap = 0;

    if (readMultMRC)
    {

      //************** Reading Multiple MRC files *************
      cout << "Opening File with MRC list names: " << filemap << "\n";
      ifstream input(filemap);

      if (!input.good())
      {
        cout << "Failed to open file contaning MRC names: " << filemap << "\n";
        exit(1);
      }

      char line[512] = {0};
      char mapname[100];
      char tmpm[10] = {0};
      const char *indifile;

      while (!input.eof())
      {
        input.getline(line, 511);
        char tmpVals[100] = {0};

        string strline(line);

        //	 cout << "MRC File name:" << strline << "\n";

        strncpy(tmpVals, line, 99);
        sscanf(tmpVals, "%99c", mapname);

        // Check for last line
        strncpy(tmpm, mapname, 3);

        if (strcmp(tmpm, "XXX") != 0)
        {
          indifile = strline.c_str();

          //   size_t foundpos= strline.find("mrc");
          //   size_t endpos = strline.find_last_not_of(" \t");

          // Reading Multiple MRC
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
        cout << "Warining:::: mrc extension NOT dectected in file name::"
             << filemap << " \n";
        cout << "Warining::::  Are you sure you want to read an MRC? \n";
      }

      read_MRC(filemap, param);
      cout << "\n++++++++++++++++++++++++++++++++++++++++++ \n";
      cout << "Particle Maps read from ONE MRC File: " << filemap << "\n";
    }
  }
  else
  {
    //************** Reading Text file *************
    int nummap = -1;
    int lasti = 0;
    int lastj = 0;
    ifstream input(filemap);
    if (!input.good())
    {
      cout << "Particle Maps Failed to open file" << endl;
      exit(1);
    }

    char line[512] = {0};
    char tmpLine[512] = {0};
    bool first = true;

    int countpix = 0;

    while (!input.eof())
    {
      input.getline(line, 511);

      strncpy(tmpLine, line, strlen(line));
      char *token = strtok(tmpLine, " ");

      if (first)
      {
        if (strcmp(token, "PARTICLE") != 0)
        {
          cout << "Missing correct Standard Map Format: PARTICLE HEADER\n"
               << endl;
          exit(1);
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
        if (lasti + 1 != param.param_device.NumberPixels &&
            lastj + 1 != param.param_device.NumberPixels && nummap > 0)
        {
          cout << "PROBLEM INCONSISTENT NUMBER OF PIXELS IN MAPS AND INPUTFILE "
                  "( "
               << param.param_device.NumberPixels << ", i " << lasti << ", j "
               << lastj << ")"
               << "\n";
          exit(1);
        }
      }
      else
      {
        int i, j;
        float z;

        char tmpVals[36] = {0};

        strncpy(tmpVals, line, 8);
        sscanf(tmpVals, "%d", &i);

        strncpy(tmpVals, line + 8, 8);
        sscanf(tmpVals, "%d", &j);

        strncpy(tmpVals, line + 16, 16);
        sscanf(tmpVals, "%f", &z);
        // checking for Map limits
        if (i > -1 && i < param.param_device.NumberPixels && j > -1 &&
            j < param.param_device.NumberPixels)
        {
          countpix++;
          maps[nummap * refMapSize + i * numPixels + j] = (myfloat_t) z;
          lasti = i;
          lastj = j;
          //	 cout << countpix << " " <<
          // param.param_device.NumberPixels*param.param_device.NumberPixels <<
          //"\n";
        }
        else
        {
          cout << "PROBLEM READING MAP (Map number " << nummap << ", i " << i
               << ", j " << j << ")"
               << "\n";
          exit(1);
        }
      }
    }
    if (lasti != param.param_device.NumberPixels - 1 ||
        lastj != param.param_device.NumberPixels - 1 ||
        countpix !=
            param.param_device.NumberPixels * param.param_device.NumberPixels +
                1)
    {
      cout << "PROBLEM INCONSISTENT NUMBER OF PIXELS IN MAPS AND INPUTFILE ( "
           << param.param_device.NumberPixels << ", i " << lasti << ", j "
           << lastj << ")"
           << "\n";
      exit(1);
    }
    cout << ".";
    ntotRefMap = nummap + 1;
    maps = (myfloat_t *) reallocchk(maps, refMapSize * sizeof(myfloat_t) *
                                              ntotRefMap);
    cout << "Particle Maps read from Standard File: " << ntotRefMap << "\n";
  }

  //************* If Dumping Maps *********************
  if (param.dumpMap)
  {
    FILE *fp = fopen("maps.dump", "w+b");
    if (fp == NULL)
    {
      cout << "Error opening dump file\n";
      exit(1);
    }
    fwrite(&ntotRefMap, sizeof(ntotRefMap), 1, fp);
    fwrite(maps, sizeof(myfloat_t) * refMapSize, ntotRefMap, fp);
    fclose(fp);
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
    cout << "ERROR opening MRC: " << filename;
    exit(1);
  }
  n_range_viol0 = test_mrc(filename, 0);
  n_range_viol1 = test_mrc(filename, 1);

  if (n_range_viol0 < n_range_viol1)
  { //* guess endianism
    swap = 0;
    if (n_range_viol0 > 0)
    {
      printf(
          " Warning: %i header field range violations detected in file %s \n",
          n_range_viol0, filename);
    }
  }
  else
  {
    swap = 1;
    if (n_range_viol1 > 0)
    {
      printf("Warning: %i header field range violations detected in file %s \n",
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
    cout << "ERROR reading MRC header: " << filename;
    exit(1);
  }

  if (nr != param.param_device.NumberPixels ||
      nc != param.param_device.NumberPixels)
  {
    cout << "PROBLEM INCONSISTENT NUMBER OF PIXELS IN MAPS AND INPUTFILE ( "
         << param.param_device.NumberPixels << ", i " << nc << ", j " << nr
         << ")"
         << "\n";
    exit(1);
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
    cout << "ERROR with MRC mode " << mode << "\n";
    cout << "Currently mode 2 is the only one allowed"
         << "\n";
    exit(1);
  }
  else
  {
    rewind(fin);
    for (count = 0; count < 256; ++count)
      if (read_float_empty(fin) == 0)
      {
        cout << "ERROR Converting Data: " << filename;
        exit(1);
      }

    for (count = 0; count < (unsigned long) nsymbt; ++count)
      if (read_char_float(&currfloat, fin) == 0)
      {
        cout << "ERROR Converting Data: " << filename;
        exit(1);
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
            cout << "ERROR Converting Data: " << filename;
            exit(1);
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

int bioem_RefMap::read_float(float *currfloat, FILE *fin, int swap)
{
  unsigned char *cptr, tmp;

  if (fread(currfloat, 4, 1, fin) != 1)
    return 0;
  if (swap == 1)
  {
    cptr = (unsigned char *) currfloat;
    tmp = cptr[0];
    cptr[0] = cptr[3];
    cptr[3] = tmp;
    tmp = cptr[1];
    cptr[1] = cptr[2];
    cptr[2] = tmp;
  }
  return 1;
}

int bioem_RefMap::read_int(int *currlong, FILE *fin, int swap)
{
  unsigned char *cptr, tmp;

  if (fread(currlong, 4, 1, fin) != 1)
    return 0;
  if (swap == 1)
  {
    cptr = (unsigned char *) currlong;
    tmp = cptr[0];
    cptr[0] = cptr[3];
    cptr[3] = tmp;
    tmp = cptr[1];
    cptr[1] = cptr[2];
    cptr[2] = tmp;
  }
  return 1;
}
int bioem_RefMap::read_float_empty(FILE *fin)
{
  float currfloat;

  if (fread(&currfloat, 4, 1, fin) != 1)
    return 0;
  return 1;
}

int bioem_RefMap::read_char_float(float *currfloat, FILE *fin)
{
  char currchar;

  if (fread(&currchar, 1, 1, fin) != 1)
    return 0;
  *currfloat = (float) currchar;
  return 1;
}

int bioem_RefMap::test_mrc(const char *vol_file, int swap)
{
  FILE *fin;
  int nc, nr, ns, mx, my, mz;
  int mode, ncstart, nrstart, nsstart;
  float xlen, ylen, zlen;
  int i, header_ok = 1, n_range_viols = 0;
  int mapc, mapr, maps_local;
  float alpha, beta, gamma;
  float dmin, dmax, dmean, dummy, xorigin, yorigin, zorigin;

  fin = fopen(vol_file, "rb");
  if (fin == NULL)
  {
    cout << "ERROR opening MRC: " << vol_file;
    exit(1);
  }

  //* read header info
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
  header_ok *= read_float(&alpha, fin, swap);
  header_ok *= read_float(&beta, fin, swap);
  header_ok *= read_float(&gamma, fin, swap);
  header_ok *= read_int(&mapc, fin, swap);
  header_ok *= read_int(&mapr, fin, swap);
  header_ok *= read_int(&maps_local, fin, swap);
  header_ok *= read_float(&dmin, fin, swap);
  header_ok *= read_float(&dmax, fin, swap);
  header_ok *= read_float(&dmean, fin, swap);
  for (i = 23; i < 50; ++i)
    header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&xorigin, fin, swap);
  header_ok *= read_float(&yorigin, fin, swap);
  header_ok *= read_float(&zorigin, fin, swap);
  fclose(fin);
  if (header_ok == 0)
  {
    cout << "ERROR reading MRC header: " << vol_file;
    exit(1);
  }

  n_range_viols += (nc > 5000);
  n_range_viols += (nc < 0);
  n_range_viols += (nr > 5000);
  n_range_viols += (nr < 0);
  n_range_viols += (ns > 5000);
  n_range_viols += (ns < 0);
  n_range_viols += (ncstart > 5000);
  n_range_viols += (ncstart < -5000);
  n_range_viols += (nrstart > 5000);
  n_range_viols += (nrstart < -5000);
  n_range_viols += (nsstart > 5000);
  n_range_viols += (nsstart < -5000);
  n_range_viols += (mx > 5000);
  n_range_viols += (mx < 0);
  n_range_viols += (my > 5000);
  n_range_viols += (my < 0);
  n_range_viols += (mz > 5000);
  n_range_viols += (mz < 0);
  n_range_viols += (alpha > 360.0f);
  n_range_viols += (alpha < -360.0f);
  n_range_viols += (beta > 360.0f);
  n_range_viols += (beta < -360.0f);
  n_range_viols += (gamma > 360.0f);
  n_range_viols += (gamma < -360.0f);

  return n_range_viols;
}
