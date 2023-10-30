/*
  Chapel backtracking algorithm to solve instances of the N-Queens problem.
  This version is a variant of nqueens_chpl.chpl using the DistBag_DFS data
  structure as work pool.
*/

use Time;
use GpuDiagnostics;
use DistributedBag_DFS;

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
Implementation of the single-core single-GPU N-Queens search.
*******************************************************************************/

config const N = 14;
config const g = 1;
config const minSize = 25;
config const maxSize = 50000;

proc check_parameters()
{
  if ((N <= 0) || (g <= 0) || (minSize <= 0) || (maxSize <= 0)) {
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
proc decompose(const parent: Node, ref tree_loc: uint, ref num_sol: uint, ref bag: DistBag_DFS(Node))
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
      bag.add(child, 0);
      tree_loc += 1;
    }
  }
}

// Evaluate a bulk of parent nodes on GPU.
proc evaluate_gpu(const parents_d: [] Node, const size: int)
{
  var evals_d: [0..#size] uint(8) = noinit;

  @assertOnGpu
  foreach threadId in 0..#size {
    const parentId = threadId / N;
    const k = threadId % N;
    const parent = parents_d[parentId];
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
      evals_d[threadId] = isSafe;
    }
  }

  return evals_d;
}

// Generate children nodes (evaluated by GPU) on CPU.
proc generate_children(const parents: [] Node, const size: int, const evals: [] uint(8),
  ref exploredTree: uint, ref exploredSol: uint, ref bag: DistBag_DFS(Node))
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
        bag.add(child, 0);
        exploredTree += 1;
      }
    }
  }
}

// Single-core single-GPU N-Queens search.
proc nqueens_search(ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var root = new Node(N);

  var bag = new DistBag_DFS(Node);

  bag.add(root, 0);

  var timer: stopwatch;
  timer.start();

  while true {
    var (hasWork, parent) = bag.remove(0);
    if (hasWork == -1) then break;

    decompose(parent, exploredTree, exploredSol, bag);

    var poolSize = min(bag.size, maxSize);

    // If 'poolSize' is sufficiently large, we offload the pool on GPU.
    if (poolSize >= minSize) {
      var (hasWork, parents) = bag.removeBulk_(poolSize, 0);
      if (hasWork == -1) then break;

      const evalsSize = N * poolSize;
      var evals: [0..#evalsSize] uint(8) = noinit;

      on here.gpus[0] {
        const parents_d = parents; // host-to-device
        evals = evaluate_gpu(parents_d, evalsSize); // device-to-host copy + kernel
      }

      generate_children(parents, poolSize, evals, exploredTree, exploredSol, bag);
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
