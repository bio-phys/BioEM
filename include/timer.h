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

#ifndef TIMER_H
#define TIMER_H

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

class HighResTimer
{

public:
  HighResTimer();
  ~HighResTimer();
  void Start();
  void Stop();
  void Reset();
  void ResetStart();
  double GetElapsedTime();
  double GetCurrentElapsedTime();

private:
  static double Frequency;
  static double GetFrequency();

  double ElapsedTime;
  double StartTime;
  int running;
};

/* Structure for saving a vector of timings */
typedef struct _TimeLog
{
  vector<double> vec;

  double sum;
  double stdev;

  string name;
} TimeLog;
enum TS_NAMES
{
  TS_TPROJECTION,
  TS_PROJECTION,
  TS_CONVOLUTION,
  TS_COMPARISON
};

/* Structure for saving timings of different parts of code and doing basic
 * statistics on them */
class TimeStat
{

public:
  TimeStat(int Angles, int CTFs) : time(0), tl(NULL)
  {
    angles = Angles;
    ctfs = CTFs;
  };
  ~TimeStat() { EmptyTimeStat(); };
  void InitTimeLog(int log, int size, string s);
  void InitTimeStat(int nlogs);
  void EmptyTimeStat();
  void inline Add(int log) { tl[log].vec.push_back(time); };
  void ComputeTimeStat();
  void PrintTimeStat(int mpi_rank);

  /* Variable for storing times during the execution */
  double time;

private:
  TimeLog *tl;
  int total_logs;
  int angles;
  int ctfs;
};

#endif
