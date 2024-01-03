/*
  Chapel backtracking algorithm to solve instances of the N-Queens problem.
*/

use Time;
use GpuDiagnostics;

config param BLOCK_SIZE = 512;

/*******************************************************************************
Implementation of N-Queens Nodes.
*******************************************************************************/

config param MAX_QUEENS = 20;

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
  var front: int;
  var size: int;

  proc init() {
    this.dom = 0..#CAPACITY;
    this.capacity = CAPACITY;
  }

  proc ref pushBack(node: Node) {
    if (this.front + this.size >= this.capacity) {
      this.capacity *=2;
      this.dom = 0..#this.capacity;
    }

    this.elements[this.front + this.size] = node;
    this.size += 1;
  }

  proc ref popBack(ref hasWork: int) {
    if (this.size > 0) {
      hasWork = 1;
      this.size -= 1;
      return this.elements[this.front + this.size];
    }

    var default: Node;
    return default;
  }

  proc ref popFront(ref hasWork: int) {
    if (this.size > 0) {
      hasWork = 1;
      const elt = this.elements[this.front];
      this.front += 1;
      this.size -= 1;
      return elt;
    }

    var default: Node;
    return default;
  }
}

/*******************************************************************************
Implementation of the multi-core multi-GPU N-Queens search.
*******************************************************************************/

config const N = 14;
config const g = 1;
config const m = 25;
config const M = 50000;
config const D = 1;

proc check_parameters()
{
  if ((N <= 0) || (g <= 0) || (m <= 0) || (M <= 0) || (D <= 0)) {
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
proc evaluate_gpu(const parents_d: [] Node, const size)
{
  var evals: [0..#size] uint(8) = noinit;

  @assertOnGpu
  foreach threadId in 0..#size {
    const parentId = threadId / N;
    const k = threadId % N;
    const parent = parents_d[parentId];
    const depth = parent.depth;
    const queen_num = parent.board[k];

    var isSafe: uint(8);

    // If child 'k' is not scheduled, we evaluate its safety 'G' times, otherwise 0.
    if (k >= depth) {
      isSafe = 1;
      /* const G_notScheduled = g * (k >= depth); */
      for _g in 0..#g {//G_notScheduled {
        for i in 0..#depth {
          isSafe *= (parent.board[i] != queen_num - (depth - i) &&
                     parent.board[i] != queen_num + (depth - i));
        }
      }
      evals[threadId] = isSafe;
    }
  }

  return evals;
}

// Generate children nodes (evaluated by GPU) on CPU.
proc generate_children(const ref parents: [] Node, const size: int, const ref evals: [] uint(8),
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
        child.depth = depth + 1;
        child.board = parent.board;
        child.board[depth] <=> child.board[j];
        pool.pushBack(child);
        exploredTree += 1;
      }
    }
  }
}

// Multi-core multi-GPU N-Queens search.
proc nqueens_search(ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var root = new Node(N);

  var pool = new SinglePool();

  pool.pushBack(root);

  var timer: stopwatch;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  timer.start();
  while (pool.size < D*m) {
    var hasWork = 0;
    var parent = pool.popFront(hasWork);
    if !hasWork then break;

    decompose(parent, exploredTree, exploredSol, pool);
  }
  timer.stop();
  var t = timer.elapsed();
  writeln("\nInitial search on CPU completed");
  writeln("Size of the explored tree: ", exploredTree);
  writeln("Number of explored solutions: ", exploredSol);
  writeln("Elapsed time: ", t, " [s]\n");

  /*
    Step 2: We continue the search on GPU in a depth-first manner, until there
    is not enough work.
  */
  timer.start();
  var eachExploredTree, eachExploredSol: [0..#D] uint;

  const poolSize = pool.size;
  const c = poolSize / D;
  const l = poolSize - (D-1)*c;
  const f = pool.front;
  var lock: atomic bool;

  pool.front = 0;
  pool.size = 0;

  coforall (gpuID, gpu) in zip(0..#D, here.gpus) with (ref pool,
    ref eachExploredTree, ref eachExploredSol) {

    var tree, sol: uint;
    var pool_loc = new SinglePool();

    // each task gets its chunk
    pool_loc.elements[0..#c] = pool.elements[gpuID+f.. by D #c];
    pool_loc.size += c;
    if (gpuID == D-1) {
      pool_loc.elements[c..#(l-c)] = pool.elements[(D*c)+f..#(l-c)];
      pool_loc.size += l-c;
    }

    while true {
      /*
        Each task gets its parents nodes from the pool.
      */
      var poolSize = pool_loc.size;
      if (poolSize >= m) {
        poolSize = min(poolSize, M);
        var parents: [0..#poolSize] Node = noinit;
        for i in 0..#poolSize {
          var hasWork = 0;
          parents[i] = pool_loc.popBack(hasWork);
          if !hasWork then break;
        }

        const evalsSize = N * poolSize;
        var evals: [0..#evalsSize] uint(8) = noinit;

        on gpu {
          const parents_d = parents; // host-to-device
          evals = evaluate_gpu(parents_d, evalsSize);
        }

        /*
          Each task 0 generates and inserts its children nodes to the pool.
        */
        generate_children(parents, poolSize, evals, tree, sol, pool_loc);
      }
      else {
        break;
      }
    }

    if lock.compareAndSwap(false, true) {
      const poolLocSize = pool_loc.size;
      for p in 0..#poolLocSize {
        var hasWork = 0;
        pool.pushBack(pool_loc.popBack(hasWork));
        if !hasWork then break;
      }
      lock.write(false);
    }

    eachExploredTree[gpuID] = tree;
    eachExploredSol[gpuID] = sol;
  }
  timer.stop();
  t = timer.elapsed() - t;

  exploredTree += (+ reduce eachExploredTree);
  exploredSol += (+ reduce eachExploredSol);

  writeln("Search on GPU completed");
  writeln("Size of the explored tree: ", exploredTree);
  writeln("Number of explored solutions: ", exploredSol);
  writeln("Elapsed time: ", t, " [s]\n");

  /*
    Step 3: We complete the depth-first search on CPU.
  */
  timer.start();
  while true {
    var hasWork = 0;
    var parent = pool.popBack(hasWork);
    if !hasWork then break;

    decompose(parent, exploredTree, exploredSol, pool);
  }
  timer.stop();
  elapsedTime = timer.elapsed();
  writeln("Search on CPU completed");
  writeln("Size of the explored tree: ", exploredTree);
  writeln("Number of explored solutions: ", exploredSol);
  writeln("Elapsed time: ", elapsedTime - t, " [s]");

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
