#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../c_headers/c_bound_simple.h"
#include "../c_headers/c_bound_johnson.h"

johnson_bd_data* new_johnson_bd_data(const bound_data *const data/*, enum lb2_variant lb2_type*/)
{
    johnson_bd_data *b = (johnson_bd_data*)malloc(sizeof(johnson_bd_data));

    b->nb_jobs = data->nb_jobs;
    b->nb_machines = data->nb_machines;

    enum lb2_variant lb2_type = LB2_FULL; //////////////////////////

    //depends on nb of machine pairs
    if(lb2_type==LB2_FULL)
        b->nb_machine_pairs = (b->nb_machines*(b->nb_machines-1))/2;
    if(lb2_type==LB2_NABESHIMA)
        b->nb_machine_pairs = b->nb_machines-1;
    if(lb2_type==LB2_LAGEWEG)
        b->nb_machine_pairs = b->nb_machines-1;
    if(lb2_type==LB2_LEARN)
        b->nb_machine_pairs = (b->nb_machines*(b->nb_machines-1))/2;

    b->lags = (int*)malloc(b->nb_machine_pairs*b->nb_jobs*sizeof(int));
    b->johnson_schedules = (int*)malloc(b->nb_machine_pairs*b->nb_jobs*sizeof(int));
    b->machine_pairs[0] = (int*)malloc(b->nb_machine_pairs*sizeof(int));
    b->machine_pairs[1] = (int*)malloc(b->nb_machine_pairs*sizeof(int));
    b->machine_pair_order = (int*)malloc(b->nb_machine_pairs*sizeof(int));
    return b;
}

void free_johnson_bd_data(johnson_bd_data* b)
{
    if(b){
        free(b->lags);
        free(b->johnson_schedules);
        free(b->machine_pairs[0]);
        free(b->machine_pairs[1]);
        free(b->machine_pair_order);
        free(b);
    }
}

void fill_machine_pairs(johnson_bd_data* b/*, enum lb2_variant lb2_type*/)
{
    if(!b){
        printf("allocate johnson_bd_data first\n");
        exit(-1);
    }

    enum lb2_variant lb2_type = LB2_FULL; //////////////////////////

    switch(lb2_type){
        case LB2_FULL:
        case LB2_LEARN:
        {
            unsigned c=0;
            for(int i=0; i<b->nb_machines-1; i++){
                for(int j=i+1; j<b->nb_machines; j++){
                    b->machine_pairs[0][c]=i;
                    b->machine_pairs[1][c]=j;
                    b->machine_pair_order[c]=c;
                    c++;
                }
            }
            break;
        }
        case LB2_NABESHIMA:{
            for(int i=0; i<b->nb_machines-1; i++){
                b->machine_pairs[0][i]=i;
                b->machine_pairs[1][i]=i+1;
                b->machine_pair_order[i]=i;
            }
            break;
        }
        case LB2_LAGEWEG:{
            for(int i=0; i<b->nb_machines-1; i++){
                b->machine_pairs[0][i]=i;
                b->machine_pairs[1][i]=b->nb_machines-1;
                b->machine_pair_order[i]=i;
            }
            break;
        }
    }
}

// term q_iuv in [Lageweg'78]
void fill_lags(const bound_data *const lb1, const johnson_bd_data *const lb2){
    int N=lb1->nb_jobs;

    for(int i=0;i<lb2->nb_machine_pairs;i++){
        int m1 = lb2->machine_pairs[0][i];
        int m2 = lb2->machine_pairs[1][i];

        for (int j = 0; j < N; j++) {
            lb2->lags[i * N + j] = 0;
            for (int k = m1 + 1; k < m2; k++) {
                lb2->lags[i * N + j] += lb1->p_times[k * N + j];
            }
        }
    }
}

typedef struct johnson_job{
    int job; //job-id
    int partition; //in partition 0 or 1
    int ptm1; //processing time on m1
    int ptm2; //processing time on m2
} johnson_job;

//(after partitioning) sorting jobs in ascending order with this comparator yield an optimal schedule for the associated 2-machine FSP [Johnson, S. M. (1954). Optimal two-and three-stage production schedules with setup times included.closed access Naval research logistics quarterly, 1(1), 61–68.]
int johnson_comp(const void * elem1, const void * elem2)
{
    johnson_job j1 = *((johnson_job*)elem1);
    johnson_job j2 = *((johnson_job*)elem2);

    //partition 0 before 1
    if(j1.partition == 0 && j2.partition == 1)return -1;
    if(j1.partition == 1 && j2.partition == 0)return 1;

    //in partition 0 increasing value of ptm1
    if(j1.partition == 0){
        if(j2.partition == 1)return -1;
        return j1.ptm1 - j2.ptm1;
    }
    //in partition 1 decreasing value of ptm1
    if(j1.partition == 1){
        if(j2.partition == 0)return 1;
        return j2.ptm2 - j1.ptm2;
    }
    return 0;
}

//for each machine-pair (m1,m2), solve 2-machine FSP with processing times
//  p_1i = PTM[m1][i] + lags[s][i]
//  p_2i = PTM[m2][i] + lags[s][i]
//using Johnson's algorithm [Johnson, S. M. (1954). Optimal two-and three-stage production schedules with setup times included.closed access Naval research logistics quarterly, 1(1), 61–68.]
void fill_johnson_schedules(const bound_data *const lb1, const johnson_bd_data *const lb2)
{
    const int N=lb1->nb_jobs;
    const int* const p_times = lb1->p_times;
    const int* const lags = lb2->lags;

    johnson_job tmp[N];

    //for all machine-pairs
    for(int k=0;k<lb2->nb_machine_pairs;k++){
        int m1 = lb2->machine_pairs[0][k];
        int m2 = lb2->machine_pairs[1][k];

        //partition N jobs into 2 sets {j|p_1j < p_2j} and {j|p_1j >= p_2j}
        for(int i=0; i<N; i++){
            tmp[i].job = i;
            tmp[i].ptm1 = p_times[m1*N + i] + lags[k*N + i];
            tmp[i].ptm2 = p_times[m2*N + i] + lags[k*N + i];

            if(tmp[i].ptm1<tmp[i].ptm2){
                tmp[i].partition=0;
            }else{
                tmp[i].partition=1;
            }
        }
        //sort according to johnson's criterion
        qsort (tmp, sizeof(tmp)/sizeof(*tmp), sizeof(*tmp), johnson_comp);
        //save optimal schedule for 2-machine problem
        for(int i=0; i<N; i++){
            lb2->johnson_schedules[k*N + i] = tmp[i].job;
        }
    }
}

void set_flags(const int *const permutation, const int limit1, const int limit2, const int N, int* flags){
    for(int i=0;i<N;i++)
        flags[i]=0;
    for (int j = 0; j <= limit1; j++)
        flags[permutation[j]] = 1;
    for (int j = limit2; j < N; j++)
        flags[permutation[j]] = 1;
}



inline int compute_cmax_johnson(const bound_data* const bd, const johnson_bd_data* const jhnsn, const int* const flag, int *tmp0, int *tmp1, int ma0, int ma1, int ind)
{
    int nb_jobs = bd->nb_jobs;

    for (int j = 0; j < nb_jobs; j++) {
        int job = jhnsn->johnson_schedules[ind*nb_jobs + j];
        // j-loop is on unscheduled jobs... (==0 if jobCour is unscheduled)
        if (flag[job] == 0) {
            int ptm0 = bd->p_times[ma0*nb_jobs + job];
            int ptm1 = bd->p_times[ma1*nb_jobs + job];
            int lag = jhnsn->lags[ind*nb_jobs + job];
            // add job on ma0 and ma1
            *tmp0 += ptm0;
            *tmp1 = MAX(*tmp1,*tmp0 + lag);
            *tmp1 += ptm1;
        }
    }

    return *tmp1;
}

int lb_makespan(const bound_data* const bd, const johnson_bd_data* const jhnsn, const int* const flag, const int* const front, const int* const back, const int minCmax){
    int lb=0;

    // for all machine-pairs : O(m^2) m*(m-1)/2
    for (int l = 0; l < jhnsn->nb_machine_pairs; l++) {
        int i = jhnsn->machine_pair_order[l];

        int ma0 = jhnsn->machine_pairs[0][i];
        int ma1 = jhnsn->machine_pairs[1][i];

        int tmp0 = front[ma0];
        int tmp1 = front[ma1];

        compute_cmax_johnson(bd,jhnsn,flag,&tmp0,&tmp1,ma0,ma1,i);

        tmp1 = MAX(tmp1 + back[ma1], tmp0 + back[ma0]);

        lb=MAX(lb,tmp1);

        if(lb>minCmax){
            break;
        }
    }

    return lb;
}

//allows variable nb of machine pairs and get machine pair the realized best lb
int lb_makespan_learn(const bound_data* const bd, const johnson_bd_data* const jhnsn, const int* const flag, const int* const front, const int* const back, const int minCmax, const int nb_pairs, int *best_index){
    int lb=0;

    for (int l = 0; l < nb_pairs; l++) {
        int i = jhnsn->machine_pair_order[l];

        int ma0 = jhnsn->machine_pairs[0][i];
        int ma1 = jhnsn->machine_pairs[1][i];

        int tmp0 = front[ma0];
        int tmp1 = front[ma1];

        compute_cmax_johnson(bd,jhnsn,flag,&tmp0,&tmp1,ma0,ma1,i);

        tmp1 = MAX(tmp1 + back[ma1], tmp0 + back[ma0]);

        if(tmp1 > lb){
            *best_index = i;
            lb = tmp1;
        }
        // lb=MAX(lb,tmp1);

        if(lb>minCmax){
            break;
        }
    }

    return lb;
}


int lb2_bound(const bound_data* const lb1_data, const johnson_bd_data* const lb2_data, const int* const permutation, const int limit1, const int limit2,const int best_cmax){
    const int N = lb1_data->nb_jobs;
    const int M = lb1_data->nb_machines;

    int front[M];
    int back[M];

    schedule_front(lb1_data, permutation,limit1,front);
    schedule_back(lb1_data, permutation,limit2,back);

    int flags[N];
    set_flags(permutation,limit1,limit2,N,flags);

    return lb_makespan(lb1_data,lb2_data,flags,front,back,best_cmax);
}

inline void swap(int *a, int *b){
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void lb2_children_bounds(const bound_data* const lb1_data, const johnson_bd_data* const lb2_data, const int* const permutation, const int limit1, const int limit2, int* const lb_begin, int* const lb_end, const int best_cmax, const int direction)
{
    const int N = lb1_data->nb_jobs;

    int tmp_perm[N];
    memcpy(tmp_perm,permutation,N*sizeof(int));

    switch(direction){
        case -1:{
            for (int i = limit1 + 1; i < limit2; i++) {
                int job = tmp_perm[i];

                swap(&tmp_perm[i], &tmp_perm[limit1 + 1]);
                lb_begin[job] = lb2_bound(lb1_data, lb2_data, tmp_perm, limit1+1, limit2, best_cmax);
                swap(&tmp_perm[i], &tmp_perm[limit1 + 1]);
            }
            break;
        }
        case 0:{
            for (int i = limit1 + 1; i < limit2; i++) {
                int job = tmp_perm[i];

                swap(&tmp_perm[i], &tmp_perm[limit1 + 1]);
                lb_begin[job] = lb2_bound(lb1_data, lb2_data, tmp_perm, limit1+1, limit2, best_cmax);
                swap(&tmp_perm[i], &tmp_perm[limit1 + 1]);

                swap(&tmp_perm[i], &tmp_perm[limit2 - 1]);
                lb_end[job] = lb2_bound(lb1_data, lb2_data, tmp_perm, limit1, limit2-1, best_cmax);
                swap(&tmp_perm[i], &tmp_perm[limit2 - 1]);
            }
            break;
        }
        case 1:{
            for (int i = limit1 + 1; i < limit2; i++) {
                int job = tmp_perm[i];

                swap(&tmp_perm[i], &tmp_perm[limit2 - 1]);
                lb_end[job] = lb2_bound(lb1_data, lb2_data, tmp_perm, limit1, limit2-1, best_cmax);
                swap(&tmp_perm[i], &tmp_perm[limit2 - 1]);
            }
            break;
        }
    }

}
