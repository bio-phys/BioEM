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

#ifndef BIOEM_ALGORITHM_H
#define BIOEM_ALGORITHM_H

__device__ static inline myprob_t
calc_logpro(const bioem_param_device &param, const myfloat_t amp,
            const myfloat_t pha, const myfloat_t env, const myfloat_t sum,
            const myfloat_t sumsquare, const myfloat_t crossproMapConv,
            const myfloat_t sumref, const myfloat_t sumsquareref)
{

  //*** MAIN ROUTINE TO CALCULATE THE LOGPRO FOR ALL KERNELS*************//
  // **** calculate the log posterior of Eq. of Pmw in SI of JSB paper ***//

  // Related to Reference calculated Projection
  const myprob_t ForLogProb = sumsquare * param.Ntotpi - sum * sum;

  // Products of different cross-correlations (first element in formula)
  const myprob_t firstele =
      param.Ntotpi *
          (sumsquareref * sumsquare - crossproMapConv * crossproMapConv) +
      2 * sumref * sum * crossproMapConv - sumsquareref * sum * sum -
      sumref * sumref * sumsquare;

  /// ******* Calculating log of Prob*********
  // As in fortran code:
  // logpro=(3-Ntotpi)*0.5*log(firstele/pConvMap[iOrient].ForLogProbfromConv[iConv])+(Ntotpi*0.5-2)*log(Ntotpi-2)-0.5*log(pConvMap[iOrient].ForLogProbfromConv[iConv])+0.5*log(PI)+(1-Ntotpi*0.5)*(log(2*PI)+1);

  myprob_t logpro =
      (3 - param.Ntotpi) * 0.5 * log(firstele) +
      (param.Ntotpi * 0.5 - 2) * log((param.Ntotpi - 2) * ForLogProb);

  //*************Adding Gaussian Prior to envelope & Defocus
  // parameter******************

  if (not param.tousepsf)
  {
    logpro -= env * env / 2. / param.sigmaPriorbctf / param.sigmaPriorbctf -
              (pha - param.Priordefcent) * (pha - param.Priordefcent) / 2. /
                  param.sigmaPriordefo / param.sigmaPriordefo -
              (amp - param.Priorampcent) * (amp - param.Priorampcent) / 2. /
                  param.sigmaPrioramp / param.sigmaPrioramp;
  }
  else
  {
    myprob_t envF, phaF;
    envF = 4. * M_PI * M_PI * env / (env * env + pha * pha);
    phaF = 4. * M_PI * M_PI * pha / (env * env + pha * pha);
    logpro -= envF * envF / 2. / param.sigmaPriorbctf / param.sigmaPriorbctf -
              (phaF - param.Priordefcent) * (phaF - param.Priordefcent) / 2. /
                  param.sigmaPriordefo / param.sigmaPriordefo -
              (amp - param.Priorampcent) * (amp - param.Priorampcent) / 2. /
                  param.sigmaPrioramp / param.sigmaPrioramp;
  }

  return (logpro);
}

__device__ static inline void
calProb(int iRefMap, int iOrient, int iConv, const myfloat_t amp,
        const myfloat_t pha, const myfloat_t env, const myfloat_t sumC,
        const myfloat_t sumsquareC, myfloat_t value, int disx, int disy,
        bioem_Probability &pProb, const bioem_param_device &param,
        const bioem_RefMap &RefMap)
{
  // IMPORTANT ROUTINE Summation of LogProb using FFTALGO
  // ********************************************************
  // *********** Calculates the BioEM probability ***********
  // ********************************************************

  myfloat_t logpro =
      calc_logpro(param, amp, pha, env, sumC, sumsquareC, value,
                  RefMap.sum_RefMap[iRefMap], RefMap.sumsquare_RefMap[iRefMap]);

#ifdef DEBUG_PROB
  printf("\t\t\tProb: iRefMap %d, iOrient %d, iConv %d, "
         "disx %d, disy %d, address -, value %f, logpro %f\n",
         iRefMap, iOrient, iConv, disx, disy, value, logpro);
#endif

  bioem_Probability_map &pProbMap = pProb.getProbMap(iRefMap);

  if (pProbMap.Constoadd < logpro)
  {
    pProbMap.Total *= exp(-logpro + pProbMap.Constoadd);
    pProbMap.Constoadd = logpro;

    // ********** Getting parameters that maximize the probability ***********
    pProbMap.max.max_prob_cent_x = -disx;
    pProbMap.max.max_prob_cent_y = -disy;
    pProbMap.max.max_prob_orient = iOrient;
    pProbMap.max.max_prob_conv = iConv;
    pProbMap.max.max_prob_norm =
        -(-sumC * RefMap.sum_RefMap[iRefMap] + param.Ntotpi * value) /
        (sumC * sumC - sumsquareC * param.Ntotpi);
    pProbMap.max.max_prob_mu =
        -(-sumC * value + sumsquareC * RefMap.sum_RefMap[iRefMap]) /
        (sumC * sumC - sumsquareC * param.Ntotpi);

#ifdef DEBUG_PROB
    printf("\tProbabilities change: iRefMap %d, iOrient %d, iConv %d, Total "
           "%f, Const %f, bestlogpro %f, sumExp -, bestId -\n",
           iRefMap, iOrient, iConv, pProbMap.Total, pProbMap.Constoadd, logpro);
    printf("\tParameters: iConv %d, myX -, myY -, disx %d, disy %d, probX "
           "%d, probY %d\n",
           iConv, disx, disy, pProbMap.max.max_prob_cent_x,
           pProbMap.max.max_prob_cent_y);
#endif
  }
  pProbMap.Total += exp(logpro - pProbMap.Constoadd);
#ifdef DEBUG_PROB
  printf("\t\tProbabilities after Sum: iRefMap %d, iOrient %d, iConv %d, "
         "Total %f, Const %f, bestlogpro %f, sumExp -, bestId -\n",
         iRefMap, iOrient, iConv, pProbMap.Total, pProbMap.Constoadd, logpro);
#endif

  if (param.writeAngles)
  {
    bioem_Probability_angle &pProbAngle = pProb.getProbAngle(iRefMap, iOrient);

    if (pProbAngle.ConstAngle < logpro)
    {
      pProbAngle.forAngles *= exp(-logpro + pProbAngle.ConstAngle);
      pProbAngle.ConstAngle = logpro;
    }

    pProbAngle.forAngles += exp(logpro - pProbAngle.ConstAngle);
  }
}

__device__ static inline void
doRefMapFFT(const int iRefMap, const int iOrient, const int iConv,
            const myfloat_t amp, const myfloat_t pha, const myfloat_t env,
            const myfloat_t sumC, const myfloat_t sumsquareC,
            const myfloat_t *lCC, bioem_Probability &pProb,
            const bioem_param_device &param, const bioem_RefMap &RefMap)
{
  //******************** Using FFT algorithm **************************
  //******************* Get cross-crollation of Ical to Iobs *******************
  //*********** Routine to get the Cross-Corellation from lCC for the interested
  // center displacement *************

  for (int cent_x = 0; cent_x <= param.maxDisplaceCenter;
       cent_x = cent_x + param.GridSpaceCenter)
  {
    for (int cent_y = 0; cent_y <= param.maxDisplaceCenter;
         cent_y = cent_y + param.GridSpaceCenter)
    {
      calProb(iRefMap, iOrient, iConv, amp, pha, env, sumC, sumsquareC,
              (myfloat_t) lCC[cent_x * param.NumberPixels + cent_y] /
                  (myfloat_t)(param.NumberPixels * param.NumberPixels),
              cent_x, cent_y, pProb, param, RefMap);
    }
    for (int cent_y = param.NumberPixels - param.maxDisplaceCenter;
         cent_y < param.NumberPixels; cent_y = cent_y + param.GridSpaceCenter)
    {
      calProb(iRefMap, iOrient, iConv, amp, pha, env, sumC, sumsquareC,
              (myfloat_t) lCC[cent_x * param.NumberPixels + cent_y] /
                  (myfloat_t)(param.NumberPixels * param.NumberPixels),
              cent_x, cent_y - param.NumberPixels, pProb, param, RefMap);
    }
  }

  for (int cent_x = param.NumberPixels - param.maxDisplaceCenter;
       cent_x < param.NumberPixels; cent_x = cent_x + param.GridSpaceCenter)
  {
    for (int cent_y = 0; cent_y <= param.maxDisplaceCenter;
         cent_y = cent_y + param.GridSpaceCenter)
    {
      calProb(iRefMap, iOrient, iConv, amp, pha, env, sumC, sumsquareC,
              (myfloat_t) lCC[cent_x * param.NumberPixels + cent_y] /
                  (myfloat_t)(param.NumberPixels * param.NumberPixels),
              cent_x - param.NumberPixels, cent_y, pProb, param, RefMap);
    }
    for (int cent_y = param.NumberPixels - param.maxDisplaceCenter;
         cent_y < param.NumberPixels; cent_y = cent_y + param.GridSpaceCenter)
    {
      calProb(iRefMap, iOrient, iConv, amp, pha, env, sumC, sumsquareC,
              (myfloat_t) lCC[cent_x * param.NumberPixels + cent_y] /
                  (myfloat_t)(param.NumberPixels * param.NumberPixels),
              cent_x - param.NumberPixels, cent_y - param.NumberPixels, pProb,
              param, RefMap);
    }
  }
}

#endif
