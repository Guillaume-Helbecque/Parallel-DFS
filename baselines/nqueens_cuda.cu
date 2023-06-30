#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Definition of the N-Queens Node type
#define MAX_QUEENS 15

typedef struct
{
  int depth;
  int board[MAX_QUEENS];
} Node;

// Implementation of a basic single pool
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
  if (pool->size < pool->capacity) {
    pool->elements[pool->size] = node;
    pool->size++;
  } else {
    Node* tmp = (Node*)malloc(pool->capacity * sizeof(Node));
    memcpy(tmp, pool->elements, pool->capacity * sizeof(Node));
    free(pool->elements);
    pool->elements = (Node*)malloc(2 * pool->capacity * sizeof(Node));
    for (int i = 0; i < pool->capacity; i++) {
      pool->elements[i] = tmp[i];
    }
    pool->capacity = 2 * pool->capacity;

    pool->elements[pool->size] = node;
    pool->size++;
  }
}

Node popBack(SinglePool* pool, int* hasWork)
{
  if (pool->size > 0) {
    Node node = pool->elements[pool->size - 1];
    pool->size--;
    *hasWork = 1;
    return node;
  }

  Node node_default;
  return node_default;
}

void clearSinglePool(SinglePool* pool)
{
  free(pool->elements);
  pool->capacity = CAPACITY;
  pool->size = 0;
}

// Implementation of the N-Queens search

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

void print_results(const int exploredTree, const int exploredSol, const double timer)
{
  printf("\n=================================================\n");
  printf("Size of the explored tree: %d\n", exploredTree);
  printf("Number of explored solutions: %d\n", exploredSol);
  printf("Elapsed time: %.4f [s]\n", timer);
  printf("=================================================\n");
}

// Swap two integers
void swap(int* a, int* b)
{
  int tmp = *b;
  *b = *a;
  *a = tmp;
}

// Check queen's safety
int isSafe(const int G, const int* board, const int queen_num, const int row_pos)
{
  for (int g = 0; g < G; g++) {
    for (int i = 0; i < queen_num; i++) {
      const int other_row_pos = board[i];

      if (other_row_pos == row_pos - (queen_num - i) ||
          other_row_pos == row_pos + (queen_num - i)) {
        return 0;
      }
    }
  }

  return 1;
}

void decompose(const int N, const int G, const Node parent, int* tree_loc, int* num_sol, SinglePool* pool)
{
  const int depth = parent.depth;

  if (depth == N) {
    *num_sol += 1;
  }
  for (int j = depth; j < N; j++) {
    if (isSafe(G, parent.board, depth, parent.board[j])) {
      Node child;
      for (int i = 0; i < N; i++) {
        child.board[i] = parent.board[i];
      }
      swap(&child.board[depth], &child.board[j]);
      child.depth = depth + 1;
      pushBack(pool, child);
      *tree_loc += 1;
    }
  }
}

// Evaluate a bulk of parent nodes on GPU
__global__ void evaluate_gpu(const int N, const int G, const Node* parents_d, int* status_d, const int size)
{
  int pid = blockIdx.x * blockDim.x + threadIdx.x;

  if (pid < size) {
    const int parentId = pid / N;
    const int k = pid % N;
    const Node parent = parents_d[parentId];
    const int depth = parent.depth;

    status_d[pid] = 1;

    const int notScheduled = (int)(k >= depth);
    for (int g = 0; g < (notScheduled*G - (1-notScheduled)); g++) {
      for (int i = 0; i < depth; i++) {
        const int other_row_pos = parent.board[i];
        const int isNotSafe = (other_row_pos == parent.board[k] - (depth - i) ||
          other_row_pos == parent.board[k] + (depth - i));

        status_d[pid] = isNotSafe * (-1) + (1-isNotSafe) * status_d[pid];
      }
    }
  }
}

void process_children(const int N, const Node* parents, const int size, const int* evals,
  int* exploredTree, int* exploredSol, SinglePool* pool)
{
  for (int i = 0; i < size; i++) {
    const Node parent = parents[i];
    const int depth = parent.depth;

    if (depth == N) {
      *exploredSol += 1;
    }
    for (int j = depth; j < N; j++) {
      if (evals[j + i * N] == 1) {
        Node child;
        for (int i = 0; i < N; i++) {
          child.board[i] = parent.board[i];
        }
        swap(&child.board[depth], &child.board[j]);
        child.depth = depth + 1;
        pushBack(pool, child);
        *exploredTree += 1;
      }
    }
  }
}

void nqueens_search(const int N, const int G, const int minSize, const int maxSize,
  int* exploredTree, int* exploredSol)
{
  Node root;
  root.depth = 0;
  for (int i = 0; i < N; i++) {
    root.board[i] = i;
  }

  SinglePool pool;
  initSinglePool(&pool);
  pushBack(&pool, root);

  while (1) {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork) {
      break;
    }

    decompose(N, G, parent, exploredTree, exploredSol, &pool);

    int poolSize = MIN(pool.size, maxSize);

    if (poolSize >= minSize) {
      Node* parents = (Node*)malloc(poolSize * sizeof(Node));
      for (int i = 0; i < poolSize; i++) {
        int hasWork = 0;
        parents[i] = popBack(&pool, &hasWork);
        if (!hasWork) {
          break;
        }
      }
      int* evals = (int*)malloc(N * poolSize * sizeof(int));

      Node* parents_d;
      int* status_d;

      // Offload node evaluation on GPU
      cudaMalloc(&parents_d, poolSize * sizeof(Node));
      cudaMalloc(&status_d, N * poolSize * sizeof(int));
      cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);

      int blockSize = 64;
      int nbBlocks = (N * poolSize / blockSize) + (((N * poolSize) % blockSize) == 0 ? 0 : 1);

      evaluate_gpu<<<nbBlocks, blockSize>>>(N, G, parents_d, status_d, N * poolSize);

      cudaMemcpy(evals, status_d, N * poolSize * sizeof(int), cudaMemcpyDeviceToHost);

      cudaFree(parents_d);
      cudaFree(status_d);

      process_children(N, parents, poolSize, evals, exploredTree, exploredSol, &pool);

      free(parents);
      free(evals);
    }
  }
  clearSinglePool(&pool);
}

int main(int argc, char* argv[])
{
  int N, G, minSize, maxSize;
  parse_parameters(argc, argv, &N, &G, &minSize, &maxSize);
  print_settings(N, G);

  int exploredTree = 0;
  int exploredSol = 0;

  clock_t startTime = clock();

  nqueens_search(N, G, minSize, maxSize, &exploredTree, &exploredSol);

  clock_t endTime = clock();
  double totalTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;

  print_results(exploredTree, exploredSol, totalTime);

  return 0;
}
