/*
  Chapel backtracking algorithm to solve instances of the N-Queens problem.
  This version is a variant of nqueens_chpl.chpl exploiting unified memory features.
*/

use Time;
use GpuDiagnostics;

config const BLOCK_SIZE = 512;

/*******************************************************************************
Implementation of N-Queens Nodes.
*******************************************************************************/

config param MAX_QUEENS = 21;

record Node {
  var depth: uint(8);
  var board: MAX_QUEENS*uint(8);

  // default initializer
  proc init() {};

  // root initializer
  proc init(const N: int) {
    init this;
    for i in 0..#N do this.board[i] = i:uint(8);
  }

  /*
    NOTE: This copy-initializer makes the Node type "non-trivial" for `noinit`.
    Perform manual copy in the code instead.
  */
  // copy initializer
  /* proc init(other: Node) {
    this.depth = other.depth;
    this.board = other.board;
  } */
}

/*******************************************************************************
Implementation of a dynamic-sized single pool data structure.
Its initial capacity is 1024, and we reallocate a new container with double
the capacity when it is full. Since we perform only DFS, it only supports
'pushBack' and 'popBack' operations.
*******************************************************************************/

config param CAPACITY = 1024;

record SinglePool {
  var dom: domain(1);
  var elements: [dom] Node;
  var capacity: int;
  var size: int;

  proc init() {
    this.dom = 0..#CAPACITY;
    this.capacity = CAPACITY;
  }

  proc ref pushBack(node: Node){
    if (this.size >= this.capacity) {
      this.capacity *=2;
      this.dom = 0..#this.capacity;
    }

    this.elements[this.size] = node;
    this.size += 1;
  }

  proc ref popBack(ref hasWork: int) {
    if (this.size > 0) {
      hasWork = 1;
      this.size -= 1;
      return this.elements[this.size];
    }

    var default: Node;
    return default;
  }
}

/*******************************************************************************
Implementation of the single-core single-GPU N-Queens search.
*******************************************************************************/

config const N = 14;
config const g = 1;
config const m = 25;
config const M = 50000;

proc check_parameters()
{
  if ((N <= 0) || (g <= 0) || (m <= 0) || (M <= 0)) {
    halt("All parameters must be positive integers.\n");
  }
}

proc print_settings()
{
  writeln("\n=================================================");
  writeln("Resolution of the ", N, "-Queens instance using Chapel");
  writeln("  with ", g, " safety check(s) per evaluation");
  writeln("=================================================");
}

proc print_results(const exploredTree: uint,
  const exploredSol: uint, const timer: real)
{
  writeln("\n=================================================");
  writeln("Size of the explored tree: ", exploredTree);
  writeln("Number of explored solutions: ", exploredSol);
  writeln("Elapsed time: ", timer, " [s]");
  writeln("=================================================\n");
}

// Check queen's safety.
proc isSafe(const board, const queen_num, const row_pos): uint(8)
{
  var isSafe: uint(8) = 1;

  for _g in 0..#g {
    for i in 0..#queen_num {
      const other_row_pos = board[i];

      if (other_row_pos == row_pos - (queen_num - i) ||
          other_row_pos == row_pos + (queen_num - i)) {
        isSafe = 0;
      }
    }
  }

  return isSafe;
}

// Evaluate and generate children nodes on CPU.
proc decompose(const parent: Node, ref tree_loc: uint, ref num_sol: uint, ref pool: SinglePool)
{
  const depth = parent.depth;

  if (depth == N) {
    num_sol += 1;
  }
  for j in depth..(N-1) {
    if isSafe(parent.board, depth, parent.board[j]) {
      var child = new Node();
      child.depth = parent.depth;
      child.board = parent.board;
      child.board[depth] <=> child.board[j];
      child.depth += 1;
      pool.pushBack(child);
      tree_loc += 1;
    }
  }
}

// Evaluate a bulk of parent nodes on GPU.
proc evaluate_gpu(const parents: [] Node, const size: int)
{
  var evals: [0..#size] uint(8) = noinit;

  @assertOnGpu
  foreach threadId in 0..#size {
    const parentId = threadId / N;
    const k = threadId % N;
    const parent = parents[parentId];
    const depth = parent.depth;
    const queen_num = parent.board[k];

    var isSafe: uint(8) = 1;

    // If child 'k' is not scheduled, we evaluate its safety 'G' times, otherwise 0.
    const G_notScheduled = g * (k >= depth);
    for _g in 0..#G_notScheduled {
      for i in 0..#depth {
        isSafe *= (parent.board[i] != queen_num - (depth - i) &&
                   parent.board[i] != queen_num + (depth - i));
      }
      evals[threadId] = isSafe;
    }
  }

  return evals;
}

// Generate children nodes (evaluated by GPU) on CPU.
proc generate_children(const parents: [] Node, const size: int, const evals: [] uint(8),
  ref exploredTree: uint, ref exploredSol: uint, ref pool: SinglePool)
{
  for i in 0..#size  {
    const parent = parents[i];
    const depth = parent.depth;

    if (depth == N) {
      exploredSol += 1;
    }
    for j in depth..(N-1) {
      if (evals[j + i * N] == 1) {
        var child = new Node();
        child.depth = parent.depth;
        child.board = parent.board;
        child.board[depth] <=> child.board[j];
        child.depth += 1;
        pool.pushBack(child);
        exploredTree += 1;
      }
    }
  }
}

// Single-core single-GPU N-Queens search.
proc nqueens_search(ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  const host = here;
  var root = new Node(N);

  var pool: SinglePool;

  pool.pushBack(root);

  var timer: stopwatch;
  timer.start();

  while true {
    var hasWork = 0;
    var parent = pool.popBack(hasWork);
    if !hasWork then break;

    decompose(parent, exploredTree, exploredSol, pool);

    const poolSize = min(pool.size, M);

    // If 'poolSize' is sufficiently large, we offload the pool on GPU.
    if (poolSize >= m) {
      on here.gpus[0] {
        // declaration of buffers on unified memory
        var parents: [0..#poolSize] Node = noinit;
        const evalsSize = N * poolSize;
        var evals: [0..#evalsSize] uint(8) = noinit;

        /*
          Initialization of buffer on CPU memory.
          Not GPU-eligible because the work pool `pool` is not known by the GPU.
        */
        on host {
          for i in 0..#poolSize do
            parents[i] = pool.popBack(hasWork);
        }

        /* GPU kernel - evaluate each children and fill evals */
        evals = evaluate_gpu(parents, evalsSize);

        /*
          On CPU - generate the children and fill the work pool.
          Not GPU-eligible because the work pool `pool` is not known by the GPU.
        */
        on host {
          generate_children(parents, poolSize, evals, exploredTree, exploredSol, pool);
        }
      }
    }
  }

  timer.stop();
  elapsedTime = timer.elapsed();

  writeln("\nExploration terminated.");
}

proc main()
{
  check_parameters();
  print_settings();

  var exploredTree: uint = 0;
  var exploredSol: uint = 0;

  var elapsedTime: real;

  startGpuDiagnostics();

  nqueens_search(exploredTree, exploredSol, elapsedTime);

  stopGpuDiagnostics();

  print_results(exploredTree, exploredSol, elapsedTime);

  writeln("GPU diagnostics:");
  writeln("   kernel_launch: ", getGpuDiagnostics().kernel_launch);
  writeln("   host_to_device: ", getGpuDiagnostics().host_to_device);
  writeln("   device_to_host: ", getGpuDiagnostics().device_to_host);
  writeln("   device_to_device: ", getGpuDiagnostics().device_to_device);

  return 0;
}
