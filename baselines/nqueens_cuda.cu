/*
  C+Cuda backtracking algorithm to solve instances of the N-Queens problem.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
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
  int size;
} SinglePool;

void initSinglePool(SinglePool* pool)
{
  pool->elements = (Node*)malloc(CAPACITY * sizeof(Node));
  pool->capacity = CAPACITY;
  pool->size = 0;
}

void pushBack(SinglePool* pool, Node node)
{
  if (pool->size >= pool->capacity) {
    pool->capacity *= 2;
    pool->elements = (Node*)realloc(pool->elements, pool->capacity * sizeof(Node));
  }

  pool->elements[pool->size++] = node;
}

Node popBack(SinglePool* pool, int* hasWork)
{
  if (pool->size > 0) {
    *hasWork = 1;
    return pool->elements[--pool->size];
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

void parse_parameters(int argc, char* argv[], int* N, int* G, int* minSize, int* maxSize)
{
  if (argc != 5) {
    printf("Usage: %s <N> <g> <minSize> <maxSize>\n", argv[0]);
    exit(0);
  }

  *N = atoi(argv[1]);
  *G = atoi(argv[2]);
  *minSize = atoi(argv[3]);
  *maxSize = atoi(argv[4]);

  if ((*N <= 0) || (*G <= 0) || (*minSize <= 0) || (*maxSize <= 0)) {
    printf("All parameters must be positive integers.\n");
    exit(0);
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
int isSafe(const int G, const uint8_t* board, const uint8_t queen_num, const uint8_t row_pos)
{
  for (int g = 0; g < G; g++) {
    for (int i = 0; i < queen_num; i++) {
      const uint8_t other_row_pos = board[i];

      if (other_row_pos == row_pos - (queen_num - i) ||
          other_row_pos == row_pos + (queen_num - i)) {
        return 0;
      }
    }
  }

  return 1;
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

    evals_d[threadId] = 1;

    // If child 'k' is not scheduled, we evaluate its safety 'G' times, otherwise 0.
    const int G_notScheduled = G * (k >= depth);
    for (int g = 0; g < G_notScheduled; g++) {
      for (int i = 0; i < depth; i++) {
        const uint8_t other_row_pos = parent.board[i];
        const int isNotSafe = (other_row_pos == parent.board[k] - (depth - i) ||
                               other_row_pos == parent.board[k] + (depth - i));

        evals_d[threadId] *= (1 - isNotSafe);
      }
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
void nqueens_search(const int N, const int G, const int minSize, const int maxSize,
  unsigned long long int* exploredTree, unsigned long long int* exploredSol)
{
  Node root;
  initRoot(&root, N);

  SinglePool pool;
  initSinglePool(&pool);

  pushBack(&pool, root);

  int count = 0;

  while (1) {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork) break;

    decompose(N, G, parent, exploredTree, exploredSol, &pool);

    int poolSize = MIN(pool.size, maxSize);

    // If 'poolSize' is sufficiently large, we offload the pool on GPU.
    if (poolSize >= minSize) {
      Node* parents = (Node*)malloc(poolSize * sizeof(Node));
      for (int i = 0; i < poolSize; i++) {
        int hasWork = 0;
        parents[i] = popBack(&pool, &hasWork);
        if (!hasWork) break;
      }

      const int evalsSize = N * poolSize;
      uint8_t* evals = (uint8_t*)malloc(evalsSize * sizeof(uint8_t));

      Node* parents_d;
      uint8_t* evals_d;

      cudaMalloc(&parents_d, poolSize * sizeof(Node));
      cudaMalloc(&evals_d, evalsSize * sizeof(uint8_t));
      cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);

      int nbBlocks = ceil((double)evalsSize / BLOCK_SIZE);

      count += 1;
      evaluate_gpu<<<nbBlocks, BLOCK_SIZE>>>(N, G, parents_d, evals_d, evalsSize);
      cudaDeviceSynchronize();

      cudaMemcpy(evals, evals_d, evalsSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

      cudaFree(parents_d);
      cudaFree(evals_d);

      generate_children(N, parents, poolSize, evals, exploredTree, exploredSol, &pool);

      free(parents);
      free(evals);
    }
  }

  printf("\nExploration terminated.\n");
  printf("Cuda kernel calls: %d\n", count);

  deleteSinglePool(&pool);
}

int main(int argc, char* argv[])
{
  int N, G, minSize, maxSize;
  parse_parameters(argc, argv, &N, &G, &minSize, &maxSize);
  print_settings(N, G);

  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  clock_t startTime = clock();

  nqueens_search(N, G, minSize, maxSize, &exploredTree, &exploredSol);

  clock_t endTime = clock();
  double totalTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;

  print_results(exploredTree, exploredSol, totalTime);

  return 0;
}
