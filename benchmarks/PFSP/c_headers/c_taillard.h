#ifndef C_TAILLARD_H
#define C_TAILLARD_H
// Benchmark instances for permutation flowshop problems
// defined in
//
// E. Taillard, Benchmarks for basic scheduling problems, European Journal of Operational Research, Volume 64, Issue 2, 1993, Pages 278-285, ISSN 0377-2217, https://doi.org/10.1016/0377-2217(93)90182-M.

// #include "../include/c_bound_simple.h"

#ifdef __cplusplus
extern "C" {
#endif

extern long time_seeds[120];

int taillard_get_nb_jobs(const int id);

int taillard_get_nb_machines(const int id);

long unif(long * seed, long low, long high);

void taillard_get_processing_times(int ptm[], const int id);

void taillard_get_instance_data(int *ptm, int *N, int *M, const int id);

#ifdef __cplusplus
}
#endif

#endif
