/*
  C+Cuda backtracking algorithm to solve instances of the N-Queens problem.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE 512

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*******************************************************************************
Implementation of N-Queens Nodes.
*******************************************************************************/

#define MAX_QUEENS 21

typedef struct
{
  uint8_t depth;
  uint8_t board[MAX_QUEENS];
} Node;

void initRoot(Node* root, const int N)
{
  root->depth = 0;
  for (uint8_t i = 0; i < N; i++) {
    root->board[i] = i;
  }
}

/*******************************************************************************
Implementation of a dynamic-sized single pool data structure.
Its initial capacity is 1024, and we reallocate a new container with double
the capacity when it is full. Since we perform only DFS, it only supports
'pushBack' and 'popBack' operations.
*******************************************************************************/

#define CAPACITY 1024

typedef struct
{
  Node* elements;
  int capacity;
  int front;
  int size;
} SinglePool;

void initSinglePool(SinglePool* pool)
{
  pool->elements = (Node*)malloc(CAPACITY * sizeof(Node));
  pool->capacity = CAPACITY;
  pool->front = 0;
  pool->size = 0;
}

void pushBack(SinglePool* pool, Node node)
{
  if (pool->front + pool->size >= pool->capacity) {
    pool->capacity *= 2;
    pool->elements = (Node*)realloc(pool->elements, pool->capacity * sizeof(Node));
  }

  pool->elements[pool->front + pool->size] = node;
  pool->size += 1;
}

Node popBack(SinglePool* pool, int* hasWork)
{
  if (pool->size > 0) {
    *hasWork = 1;
    pool->size -= 1;
    return pool->elements[pool->front + pool->size];
  }

  return (Node){0};
}

Node popFront(SinglePool* pool, int* hasWork)
{
  if (pool->size > 0) {
    *hasWork = 1;
    pool->size--;
    return pool->elements[pool->front++];
  }

  return (Node){0};
}

void deleteSinglePool(SinglePool* pool)
{
  free(pool->elements);
}

/*******************************************************************************
Implementation of the single-core single-GPU N-Queens search.
*******************************************************************************/

void parse_parameters(int argc, char* argv[], int* N, int* G, int* m, int* M, int* D)
{
  *N = 14;
  *G = 1;
  *m = 25;
  *M = 50000;
  *D = 1;

  int opt, value;

  while ((opt = getopt(argc, argv, "N:g:m:M:D:")) != -1) {
    value = atoi(optarg);

    if (value <= 0) {
      printf("All parameters must be positive integers.\n");
      exit(EXIT_FAILURE);
    }

    switch (opt) {
      case 'N':
        *N = value;
        break;
      case 'g':
        *G = value;
        break;
      case 'm':
        *m = value;
        break;
      case 'M':
        *M = value;
        break;
      case 'D':
        *D = value;
        break;
      default:
        fprintf(stderr, "Usage: %s -N value -g value -m value -M value -D value\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }
}

void print_settings(const int N, const int G)
{
  printf("\n=================================================\n");
  printf("Resolution of the %d-Queens instance using C+CUDA\n", N);
  printf("  with %d safety check(s) per evaluation\n", G);
  printf("=================================================\n");
}

void print_results(const unsigned long long int exploredTree,
  const unsigned long long int exploredSol, const double timer)
{
  printf("\n=================================================\n");
  printf("Size of the explored tree: %llu\n", exploredTree);
  printf("Number of explored solutions: %llu\n", exploredSol);
  printf("Elapsed time: %.4f [s]\n", timer);
  printf("=================================================\n");
}

void swap(uint8_t* a, uint8_t* b)
{
  uint8_t tmp = *b;
  *b = *a;
  *a = tmp;
}

// Check queen's safety.
uint8_t isSafe(const int G, const uint8_t* board, const uint8_t queen_num, const uint8_t row_pos)
{
  uint8_t isSafe = 1;

  for (int g = 0; g < G; g++) {
    for (int i = 0; i < queen_num; i++) {
      const uint8_t other_row_pos = board[i];

      if (other_row_pos == row_pos - (queen_num - i) ||
          other_row_pos == row_pos + (queen_num - i)) {
        isSafe = 0;
      }
    }
  }

  return isSafe;
}

// Evaluate and generate children nodes on CPU.
void decompose(const int N, const int G, const Node parent,
  unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool* pool)
{
  const uint8_t depth = parent.depth;

  if (depth == N) {
    *num_sol += 1;
  }
  for (int j = depth; j < N; j++) {
    if (isSafe(G, parent.board, depth, parent.board[j])) {
      Node child;
      memcpy(child.board, parent.board, N * sizeof(uint8_t));
      swap(&child.board[depth], &child.board[j]);
      child.depth = depth + 1;
      pushBack(pool, child);
      *tree_loc += 1;
    }
  }
}

// Evaluate a bulk of parent nodes on GPU.
__global__ void evaluate_gpu(const int N, const int G, const Node* parents_d, uint8_t* evals_d, const int size)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId < size) {
    const int parentId = threadId / N;
    const int k = threadId % N;
    const Node parent = parents_d[parentId];
    const uint8_t depth = parent.depth;
    const uint8_t queen_num = parent.board[k];

    uint8_t isSafe = 0;

    // If child 'k' is not scheduled, we evaluate its safety 'G' times, otherwise 0.
    if (k >= depth) {
      isSafe = 1;
      // const int G_notScheduled = G * (k >= depth);
      for (int g = 0; g < G; g++) {
        for (int i = 0; i < depth; i++) {
          isSafe *= (parent.board[i] != queen_num - (depth - i) &&
                     parent.board[i] != queen_num + (depth - i));
        }
      }
      evals_d[threadId] = isSafe;
    }
  }
}

// Generate children nodes (evaluated by GPU) on CPU.
void generate_children(const int N, const Node* parents, const int size, const uint8_t* evals,
  unsigned long long int* exploredTree, unsigned long long int* exploredSol, SinglePool* pool)
{
  for (int i = 0; i < size; i++) {
    const Node parent = parents[i];
    const uint8_t depth = parent.depth;

    if (depth == N) {
      *exploredSol += 1;
    }
    for (int j = depth; j < N; j++) {
      if (evals[j + i * N] == 1) {
        Node child;
        memcpy(child.board, parent.board, N * sizeof(uint8_t));
        swap(&child.board[depth], &child.board[j]);
        child.depth = depth + 1;
        pushBack(pool, child);
        *exploredTree += 1;
      }
    }
  }
}

// Single-core single-GPU N-Queens search.
void nqueens_search(const int N, const int G, const int m, const int M, const int D,
  unsigned long long int* exploredTree, unsigned long long int* exploredSol,
  double* elapsedTime)
{
  Node root;
  initRoot(&root, N);

  SinglePool pool;
  initSinglePool(&pool);

  pushBack(&pool, root);

  int count = 0;
  double startTime, endTime;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  startTime = omp_get_wtime();
  while (pool.size < D * m) {
    int hasWork = 0;
    Node parent = popFront(&pool, &hasWork);
    if (!hasWork) break;

    decompose(N, G, parent, exploredTree, exploredSol, &pool);
  }
  endTime = omp_get_wtime();
  double t = endTime - startTime;
  printf("\nInitial search on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t);

  /*
    Step 2: We continue the search on GPU in a depth-first manner, until there
    is not enough work.
  */
  unsigned long long int eachExploredTree[D], eachExploredSol[D];

  const int poolSize = pool.size;
  const int c = poolSize / D;
  const int l = poolSize - (D-1)*c;
  const int f = pool.front;

  pool.front = 0;
  pool.size = 0;

  #pragma omp parallel for num_threads(D) shared(eachExploredTree, eachExploredSol, pool)
  for (int gpuID = 0; gpuID < D; gpuID++) {
    cudaSetDevice(gpuID);

    unsigned long long int tree = 0, sol = 0;
    SinglePool pool_loc;
    initSinglePool(&pool_loc);

    for (int i = 0; i < c; i++) {
      pool_loc.elements[i] = pool.elements[gpuID+f+i*D];
    }
    pool_loc.size += c;
    if (gpuID == D-1) {
      for (int i = c; i < l; i++) {
        pool_loc.elements[i] = pool.elements[(D*c)+f+i-c];
      }
      pool_loc.size += l-c;
    }

    while (1) {
      int poolSize = pool_loc.size;
      if (poolSize >= m) {
        poolSize = MIN(poolSize, M);
        Node* parents = (Node*)malloc(poolSize * sizeof(Node));
        for (int i = 0; i < poolSize; i++) {
          int hasWork = 0;
          parents[i] = popBack(&pool_loc, &hasWork);
          if (!hasWork) break;
        }

        const int evalsSize = N * poolSize;
        uint8_t* evals = (uint8_t*)malloc(evalsSize * sizeof(uint8_t));

        Node* parents_d;
        uint8_t* evals_d;

        cudaMalloc(&parents_d, poolSize * sizeof(Node));
        cudaMalloc(&evals_d, evalsSize * sizeof(uint8_t));
        cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);

        const int nbBlocks = ceil((double)evalsSize / BLOCK_SIZE);
        count += 1;

        // call the compute kernel on device (deviceID)
        evaluate_gpu<<<nbBlocks, BLOCK_SIZE>>>(N, G, parents_d, evals_d, evalsSize);
        cudaMemcpy(evals, evals_d, evalsSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        cudaFree(parents_d);
        cudaFree(evals_d);

        generate_children(N, parents, poolSize, evals, &tree, &sol, &pool_loc);

        free(parents);
        free(evals);
      }
      else {
        break;
      }
    }

    #pragma omp critical
    {
      for (int i = 0; i < pool_loc.size; i++) {
        int hasWork = 0;
        pushBack(&pool, popBack(&pool_loc, &hasWork));
        if (!hasWork) break;
      }
    }

    eachExploredTree[gpuID] = tree;
    eachExploredSol[gpuID] = sol;

    deleteSinglePool(&pool_loc);
  }
  endTime = omp_get_wtime();
  t = endTime - startTime;

  for (int i = 0; i < D; i++) {
    *exploredTree += eachExploredTree[i];
    *exploredSol += eachExploredSol[i];
  }

  printf("\nSearch on GPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t);

  /*
    Step 3: We complete the depth-first search on CPU.
  */
  while (1) {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork) break;

    decompose(N, G, parent, exploredTree, exploredSol, &pool);
  }
  endTime = omp_get_wtime();
  *elapsedTime = endTime - startTime;
  printf("\nSearch on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", *elapsedTime - t);

  printf("\nExploration terminated.\n");
  printf("Cuda kernel calls: %d\n", count);

  deleteSinglePool(&pool);
}

int main(int argc, char* argv[])
{
  int N, G, m, M, D;
  parse_parameters(argc, argv, &N, &G, &m, &M, &D);
  print_settings(N, G);

  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime;

  nqueens_search(N, G, m, M, D, &exploredTree, &exploredSol, &elapsedTime);

  print_results(exploredTree, exploredSol, elapsedTime);

  return 0;
}
