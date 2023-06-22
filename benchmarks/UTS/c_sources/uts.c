/*
 *         ---- The Unbalanced Tree Search (UTS) Benchmark ----
 *
 *  Copyright (c) 2010 See AUTHORS file for copyright holders
 *
 *  This file is part of the unbalanced tree search benchmark.  This
 *  project is licensed under the MIT Open Source license.  See the LICENSE
 *  file for copyright and licensing information.
 *
 *  UTS is a collaborative project between researchers at the University of
 *  Maryland, the University of North Carolina at Chapel Hill, and the Ohio
 *  State University.  See AUTHORS file for more information.
 *
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#ifdef sgi
#include <time.h>
#else
#include <sys/time.h>
#endif

#include "../c_headers/uts.h"

char const * uts_trees_str[]     = { "Binomial", "Geometric", "Hybrid", "Balanced" };
char const * uts_geoshapes_str[] = { "Linear decrease", "Exponential decrease", "Cyclic",
                                        "Fixed branching factor" };

/***********************************************************
 *                                                         *
 *  FUNCTIONS                                              *
 *                                                         *
 ***********************************************************/

/*
 * wall clock time
 *   for detailed accounting of work, this needs
 *   high resolution
 */
#ifdef sgi
double uts_wctime() {
  timespec_t tv;
  double time;
  clock_gettime(CLOCK_SGI_CYCLE,&tv);
  time = ((double) tv.tv_sec) + ((double)tv.tv_nsec / 1e9);
  return time;
}
#else
double uts_wctime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}
#endif

// Interpret 32 bit positive integer as value on [0,1)
double rng_toProb(int n)
{
  if (n < 0) {
    printf("*** toProb: rand n = %d out of range\n",n);
  }
  return ((n<0)? 0.0 : ((double) n)/2147483648.0);
}

void uts_initRoot(Node_UTS* root, tree_t treeType, int rootId)
{
  root->dist = treeType;
  root->height = 0;
  root->numChildren = -1; // not yet determined
  rng_init(root->state.state, rootId);
}

int uts_numChildren_bin(Node_UTS * parent, int nonLeafBF, double nonLeafProb)
{
  // distribution is identical everywhere below root
  int    v = rng_rand(parent->state.state);
  double d = rng_toProb(v);

  return (d < nonLeafProb) ? nonLeafBF : 0;
}

int uts_numChildren_geo(Node_UTS* parent, double b_0, geoshape_t shape_fn, int gen_mx)
{
  double b_i = b_0;
  int depth = parent->height;
  int numChildren, h;
  double p, u;

  // use shape function to compute target b_i
  if (depth > 0){
    switch (shape_fn) {

      // expected size polynomial in depth
    case EXPDEC:
      b_i = b_0 * pow((double) depth, -log(b_0)/log((double) gen_mx));
      break;

      // cyclic tree size
    case CYCLIC:
      if (depth > 5 * gen_mx){
        b_i = 0.0;
        break;
      }
      b_i = pow(b_0, sin(2.0*3.141592653589793*(double) depth / (double) gen_mx));
      break;

      // identical distribution at all nodes up to max depth
    case FIXED:
      b_i = (depth < gen_mx)? b_0 : 0;
      break;

      // linear decrease in b_i
    case LINEAR:
    default:
      b_i =  b_0 * (1.0 - (double) depth / (double) gen_mx);
      break;
    }
  }

  // given target b_i, find prob p so expected value of
  // geometric distribution is b_i.
  p = 1.0 / (1.0 + b_i);

  // get uniform random number on [0,1)
  h = rng_rand(parent->state.state);
  u = rng_toProb(h);

  // max number of children at this cumulative probability
  // (from inverse geometric cumulative density function)
  numChildren = (int) floor(log(1 - u) / log(1 - p));

  return numChildren;
}


int uts_numChildren(Node_UTS *parent, tree_t treeType, int nonLeafBF, double nonLeafProb,
  double b_0, geoshape_t shape_fn, int gen_mx, double shiftDepth)
{
  int numChildren = 0;

  /* Determine the number of children */
  switch (treeType) {
    case BIN:
      if (parent->height == 0)
        numChildren = (int) floor(b_0);
      else
        numChildren = uts_numChildren_bin(parent, nonLeafBF, nonLeafProb);
      break;

    case GEO:
      numChildren = uts_numChildren_geo(parent, b_0, shape_fn, gen_mx);
      break;

    case HYBRID:
      if (parent->height < shiftDepth * gen_mx)
        numChildren = uts_numChildren_geo(parent, b_0, shape_fn, gen_mx);
      else
        numChildren = uts_numChildren_bin(parent, nonLeafBF, nonLeafProb);
      break;

    case BALANCED:
      if (parent->height < gen_mx)
        numChildren = (int) b_0;
      break;

    default:
      return -1;
  }

  // limit number of children
  // only a BIN root can have more than MAXNUMCHILDREN
  if (parent->height == 0 && parent->dist == BIN) {
    int rootBF = (int) ceil(b_0);
    if (numChildren > rootBF) {
      printf("*** Number of children of root truncated from %d to %d\n",
             numChildren, rootBF);
      numChildren = rootBF;
    }
  }
  else if (treeType != BALANCED) {
    if (numChildren > MAXNUMCHILDREN) {
      printf("*** Number of children truncated from %d to %d\n",
             numChildren, MAXNUMCHILDREN);
      numChildren = MAXNUMCHILDREN;
    }
  }

  return numChildren;
}

int uts_childType(Node_UTS* parent, tree_t treeType, double shiftDepth, int gen_mx)
{
  switch (treeType) {
    case BIN:
      return BIN;
    case GEO:
      return GEO;
    case HYBRID:
      if (parent->height < shiftDepth * gen_mx)
        return GEO;
      else
        return BIN;
    case BALANCED:
      return BALANCED;
    default:
      return -1;
  }
}

// construct string with all parameter settings
// int uts_paramsToStr(char *strBuf, int ind) {
//
//   // random number generator
//   ind += sprintf(strBuf+ind, "Random number generator: ");
//   ind  = rng_showtype(strBuf, ind);
//
//   return ind;
// }

// void uts_showStats(int nPes, int chunkSize, double walltime, counter_t nNodes, counter_t nLeaves, counter_t maxDepth) {
//   // summarize execution info for machine consumption
//   if (verbose == 0) {
//     printf("%4d %7.3f %9llu %7.0llu %7.0llu %d %d %.2f %d %d %1d %f %3d\n",
//         nPes, walltime, nNodes, (long long)(nNodes/walltime), (long long)((nNodes/walltime)/nPes), chunkSize,
//         treeType, b_0, rootId, gen_mx, shape_fn, nonLeafProb, nonLeafBF);
//   }
//
//   // summarize execution info for human consumption
//   else {
//     printf("Tree size = %llu, tree depth = %llu, num leaves = %llu (%.2f%%)\n", nNodes, maxDepth, nLeaves, nLeaves/(float)nNodes*100.0);
//     printf("Wallclock time = %.3f sec, performance = %.0f nodes/sec (%.0f nodes/sec per PE)\n\n",
//         walltime, (nNodes / walltime), (nNodes / walltime / nPes));
//   }
// }

void c_decompose(Node_UTS *parent, Node_UTS children[], tree_t treeType, int numChildren,
  int gen_mx, double shiftDepth, int computeGranularity, int *treeSize, int *maxDepth)
{
  int parentHeight = parent->height;
  int childType;

  childType   = uts_childType(parent, treeType, shiftDepth, gen_mx);

  // construct children and push onto stack
  int i, j;
  Node_UTS child;

  child.dist = childType;
  child.height = parentHeight + 1;

  for (i = 0; i < numChildren; i++) {
    for (j = 0; j < computeGranularity; j++) {
      // computeGranularity controls number of rng_spawn calls per node
      rng_spawn(parent->state.state, child.state.state, i);
    }
     children[i] = child;
  }
  *treeSize += numChildren;
  *maxDepth = MAX(*maxDepth, parentHeight + 1);

  return;
}
