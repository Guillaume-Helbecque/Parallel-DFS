#ifndef C_BOUND_JOHNSON_H
#define C_BOUND_JOHNSON_H

#ifdef __cplusplus
extern "C" {
#endif

#include "c_bound_simple.h"

//==========2-machine bounds for PFSP==========
//see
// - Lageweg, B. J., Lenstra, J. K., & A. H. G. Rinnooy Kan. (1978). A General Bounding Scheme for the Permutation Flow-Shop Problem. Operations Research, 26(1), 53–67. http://www.jstor.org/stable/169891
// - Jan Gmys, Mohand Mezmaz, Nouredine Melab, Daniel Tuyttens. A computationally eﬀicient Branch-and-Bound algorithm for the permutation flow-shop scheduling problem. European Journal of Operational Research, Elsevier, 2020, 284 (3), pp.814-833.10.1016/j.ejor.2020.01.039

//regroup (constant) bound data
typedef struct johnson_bd_data
{
    int *johnson_schedules;
    int *lags;
    int *machine_pairs[2];
    int *machine_pair_order;

    int nb_machine_pairs;
    int nb_jobs;
    int nb_machines;
} johnson_bd_data;

enum lb2_variant { LB2_FULL, LB2_NABESHIMA, LB2_LAGEWEG, LB2_LEARN };



//-------prepare constant/precomputed data for johnson bound-------
johnson_bd_data* new_johnson_bd_data(const bound_data *const lb1/*, enum lb2_variant lb2_type*/);
void free_johnson_bd_data(johnson_bd_data* b);

void fill_machine_pairs(johnson_bd_data* b/*, enum lb2_variant lb2_type*/);
void fill_lags(const bound_data *const lb1, const johnson_bd_data *const lb2);
int johnson_comp(const void * elem1, const void * elem2);
void fill_johnson_schedules(const bound_data *const lb1, const johnson_bd_data *const lb2);

//helper
void set_flags(const int *const permutation, const int limit1, const int limit2, const int N, int* flags);

//-------------compute lower bounds-------------
int compute_cmax_johnson(const bound_data* const bd, const johnson_bd_data* const jhnsn, const int* const flag, int* tmp0, int* tmp1, int ma0, int ma1, int ind);

int lb_makespan(const bound_data* const bd, const johnson_bd_data* const jhnsn, const int* const flag, const int* const front, const int* const back, const int minCmax);

int lb_makespan_learn(const bound_data* const bd, const johnson_bd_data* const jhnsn, const int* const flag, const int* const front, const int* const back, const int minCmax, const int nb_pairs, int* best_index);

int lb2_bound(const bound_data* const lb1_data, const johnson_bd_data* const lb2_data, const int* const permutation, const int limit1, const int limit2,const int best_cmax);

void lb2_children_bounds(const bound_data* const lb1_data, const johnson_bd_data* const lb2_data, const int* const permutation, const int limit1, const int limit2, int* const lb_begin, int* const lb_end, const int best_cmax, const int direction);

#ifdef __cplusplus
}
#endif

#endif
