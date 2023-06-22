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

#ifndef _UTS_H
#define _UTS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rng.h"

/***********************************************************
 *  Tree node descriptor and statistics                    *
 ***********************************************************/

#define MAXNUMCHILDREN    100  // cap on children (BIN root is exempt)

struct node_t {
  int dist;          // distribution governing number of children
  int height;        // depth of this node in the tree
  int numChildren;   // number of children, -1 => not yet determined

  /* for statistics (if configured via UTS_STAT) */
#ifdef UTS_STAT
  struct node_t *pp;          // parent pointer
  int sizeChildren;           // sum of children sizes
  int maxSizeChildren;        // max of children sizes
  int ind;
  int size[MAXNUMCHILDREN];   // children sizes
  double unb[MAXNUMCHILDREN]; // imbalance of each child 0 <= unb_i <= 1
#endif

  /* for RNG state associated with this node */
  struct state_t state;
};

typedef struct node_t Node_UTS;

/* Tree type
 *   Trees are generated using a Galton-Watson process, in
 *   which the branching factor of each node is a random
 *   variable.
 *
 *   The random variable can follow a binomial distribution
 *   or a geometric distribution.  Hybrid tree are
 *   generated with geometric distributions near the
 *   root and binomial distributions towards the leaves.
 */
enum uts_trees_e    { BIN = 0, GEO, HYBRID, BALANCED };
enum uts_geoshape_e { LINEAR = 0, EXPDEC, CYCLIC, FIXED };

typedef enum uts_trees_e    tree_t;
typedef enum uts_geoshape_e geoshape_t;

/* Strings for the above enums */
extern char const * uts_trees_str[];
extern char const * uts_geoshapes_str[];

/* For stats generation */
typedef unsigned long long counter_t;

/* Utility Functions */
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b)) // not used

// void   uts_showStats(int nPes, int chunkSize, double walltime, counter_t nNodes, counter_t nLeaves, counter_t maxDepth);
double uts_wctime();

double rng_toProb(int n);

/* Common tree routines */
void  uts_initRoot(Node_UTS *root, tree_t treeType, int rootId);
int   uts_numChildren(Node_UTS *parent, tree_t treeType, int nonLeafBF, double nonLeafProb,
  double b_0, geoshape_t shape_fn, int gen_mx, double shiftDepth);
int   uts_numChildren_bin(Node_UTS *parent, int nonLeafBF, double nonLeafProb);
int   uts_numChildren_geo(Node_UTS *parent, double b_0, geoshape_t shape_fn, int gen_mx);
int   uts_childType(Node_UTS *parent, tree_t treeType, double shiftDepth, int gen_mx);

void c_decompose(Node_UTS *parent, Node_UTS children[], tree_t treeType, int numChildren,
  int gen_mx, double shiftDepth, int computeGranularity, int *treeSize, int *maxDepth);

#ifdef __cplusplus
}
#endif

#endif /* _UTS_H */
