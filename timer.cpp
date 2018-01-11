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

#include "timer.h"
#ifdef _WIN32
#include <winbase.h>
#include <windows.h>
#else
#include <time.h>
#endif

HighResTimer::HighResTimer()
{
  ElapsedTime = 0;
  running = 0;
}

HighResTimer::~HighResTimer() {}

void HighResTimer::Start()
{
#ifdef _WIN32
  __int64 istart;
  QueryPerformanceCounter((LARGE_INTEGER *) &istart);
  StartTime = (double) istart;
#else
  timespec tv;
  clock_gettime(CLOCK_REALTIME, &tv);
  StartTime = (double) tv.tv_sec * 1.0E9 + (double) tv.tv_nsec;
#endif
  running = 1;
}

void HighResTimer::ResetStart()
{
  ElapsedTime = 0;
  Start();
}

void HighResTimer::Stop()
{
  if (running == 0)
    return;
  running = 0;
  double EndTime = 0;
#ifdef _WIN32
  __int64 iend;
  QueryPerformanceCounter((LARGE_INTEGER *) &iend);
  EndTime = (double) iend;
#else
  timespec tv;
  clock_gettime(CLOCK_REALTIME, &tv);
  EndTime = (double) tv.tv_sec * 1.0E9 + (double) tv.tv_nsec;
#endif
  ElapsedTime += EndTime - StartTime;
}

void HighResTimer::Reset()
{
  ElapsedTime = 0;
  StartTime = 0;
  running = 0;
}

double HighResTimer::GetElapsedTime() { return ElapsedTime / Frequency; }

double HighResTimer::GetCurrentElapsedTime()
{
  if (running == 0)
    return (GetElapsedTime());
  double CurrentTime = 0;
#ifdef _WIN32
  __int64 iend;
  QueryPerformanceCounter((LARGE_INTEGER *) &iend);
  CurrentTime = (double) iend;
#else
  timespec tv;
  clock_gettime(CLOCK_REALTIME, &tv);
  CurrentTime = (double) tv.tv_sec * 1.0E9 + (double) tv.tv_nsec;
#endif
  return ((CurrentTime - StartTime + ElapsedTime) / Frequency);
}

double HighResTimer::GetFrequency()
{
#ifdef _WIN32
  __int64 ifreq;
  QueryPerformanceFrequency((LARGE_INTEGER *) &ifreq);
  return ((double) ifreq);
#else
  return (1.0E9);
#endif
}

double HighResTimer::Frequency = HighResTimer::GetFrequency();

void TimeStat::InitTimeLog(int log, int size, string s)
{
  tl[log].vec.reserve(size);
  tl[log].name = s;

  tl[log].sum = 0.;
  tl[log].stdev = 0.;
}

void TimeStat::InitTimeStat(int nlogs)
{
  total_logs = nlogs;
  tl = new TimeLog[total_logs];

  InitTimeLog(TS_TPROJECTION, angles, "Total time of projection");
  InitTimeLog(TS_PROJECTION, angles, "Projection");
  InitTimeLog(TS_CONVOLUTION, angles * ctfs, "Convolution");
  InitTimeLog(TS_COMPARISON, angles * ctfs, "Comparison");
}

void TimeStat::EmptyTimeStat()
{
  if (tl == NULL)
    return;

  delete[] tl;
  tl = NULL;
  time = 0.;
}

void TimeStat::ComputeTimeStat()
{
  double mean, sq_sum;
  vector<double> diff;

  for (int i = 0; i < total_logs; i++)
  {
    tl[i].sum = std::accumulate(tl[i].vec.begin(), tl[i].vec.end(), 0.0);
    mean = tl[i].sum / tl[i].vec.size();

    diff.resize(tl[i].vec.size());
    std::transform(tl[i].vec.begin(), tl[i].vec.end(), diff.begin(),
                   std::bind2nd(std::minus<double>(), mean));
    sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    tl[i].stdev = std::sqrt(sq_sum / tl[i].vec.size());
  }
}

void TimeStat::PrintTimeStat(int mpi_rank)
{
  ComputeTimeStat();
  for (int i = 0; i < total_logs; i++)
  {
    printf("SUMMARY -> %s: Total %f sec; Mean %f sec; Std.Dev. %f (rank %d)\n",
           tl[i].name.c_str(), tl[i].sum, tl[i].sum / tl[i].vec.size(),
           tl[i].stdev, mpi_rank);
  }
}
