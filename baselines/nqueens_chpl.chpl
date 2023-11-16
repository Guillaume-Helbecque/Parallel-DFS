/*
  Chapel backtracking algorithm to solve instances of the N-Queens problem.
*/

use Time;
use GpuDiagnostics;

config const BLOCK_SIZE = 512;
config const numGpus = 1;

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

config param CAPACITY = 500* 1_000_000; //1024;

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
    /* if (this.front + this.size >= this.capacity) {
      this.capacity *=2;
      this.dom = 0..#this.capacity;
    } */

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
proc evaluate_gpu(const parents_d: [] Node, const size)
{
  var children: [0..#size] Node = noinit;

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
        /* evals_d[threadId] = isSafe; */
      }
    }

    children[threadId].depth = 0;

    if isSafe {
      ref child = children[threadId];
      child.depth = parent.depth;
      child.board = parent.board;
      child.board[depth] <=> child.board[k];
      child.depth += 1;
    }
  }

  return children;
}

// Generate children nodes (evaluated by GPU) on CPU.
proc generate_gpu(const parents: [] Node, const size: int, const evals: [] uint(8),
  ref exploredTree: uint, ref exploredSol: uint)
{
  var children: [0..#size] Node = noinit;

  @assertOnGpu
  foreach i in 0..#size {
    const parentId = i / N;
    const k = i % N;
    const parent = parents[parentId];
    const depth = parent.depth;

    children[i].depth = 0;

    /* if (depth == N) {
      exploredSol += 1;
    } */
    if (evals[i] == 1) {
      ref child = children[i];
      child.depth = parent.depth;
      child.board = parent.board;
      child.board[depth] <=> child.board[k];
      child.depth += 1;
      /* exploredTree += 1; */
    }
  }

  return children;
}

// Generate children nodes (evaluated by GPU) on CPU.
/* proc generate_children(const parents: [] Node, const size: int, const evals: [] uint(8),
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
} */

// Single-core single-GPU N-Queens search.
proc nqueens_search(ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var root = new Node(N);

  var pool: SinglePool;

  pool.pushBack(root);

  var timer: stopwatch;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  timer.start();
  while (pool.size < numGpus*m) {
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
  var eachExploredTree, eachExploredSol: [0..#numGpus] uint;

  var chunkSize: [0..#numGpus] int = noinit;
  var poolSize = pool.size;
  const c = pool.size / numGpus;
  for i in 0..#(numGpus-1) do chunkSize[i] = c;
  chunkSize[numGpus-1] = poolSize - (numGpus-1)*c;
  const f = pool.front;

  var lock: atomic bool;

  pool.front = 0;
  pool.size = 0;

  coforall (gpuID, gpu) in zip(0..#numGpus, here.gpus) with (ref pool,
    ref eachExploredTree, ref eachExploredSol) {

    var pool_loc: SinglePool;

    // each task gets its chunk
    pool_loc.elements[0..#c] = pool.elements[gpuID+f.. by numGpus #c];
    if (gpuID == numGpus-1) {
      pool_loc.elements[c..#(chunkSize[gpuID]-c)] = pool.elements[(numGpus*c)+f..#(chunkSize[gpuID]-c)];
    }
    pool_loc.size += chunkSize[gpuID];

    ref exploredTree = eachExploredTree[gpuID];
    ref exploredSol = eachExploredSol[gpuID];

    var timer_s: stopwatch;
    while true {
      /* break; */
      /*
        Each task gets its parents nodes from the pool.
      */
      var poolSize = pool_loc.size;
      if (poolSize >= m) {
        poolSize = min(poolSize, M);
        var parents: [0..#poolSize] Node = noinit;
        for i in 0..#poolSize {
          var hasWork = 0;
          parents[i] = pool_loc.popFront(hasWork);
          if !hasWork then break;
        }

        const evalsSize = N * poolSize;
        var children: [0..#evalsSize] Node = noinit;

        on gpu {
          const parents_d = parents; // host-to-device
          /* const evals_d = evaluate_gpu(parents_d, evalsSize); // device-to-host copy + kernel */
          /* children = generate_gpu(parents_d, evalsSize, evals_d, exploredTree, exploredSol); */
          children = evaluate_gpu(parents_d, evalsSize);
        }

        /*
          Each task 0 generates and inserts its children nodes to the pool.
        */
        timer_s.start();
        for i in 0..#evalsSize {
          if children[i].depth != 0 {
            if (children[i].depth == N) then exploredSol += 1;
            pool_loc.pushBack(children[i]);
            exploredTree += 1;
          }
        }
        timer_s.stop();
      }
      else {
        break;
      }
    }
    writeln("timer on task ", gpuID, " : ", timer_s.elapsed());

    if lock.compareAndSwap(false, true) {
      for p in 0..#pool_loc.size {
        var hasWork = 0;
        pool.pushBack(pool_loc.popBack(hasWork));
        if !hasWork then break;
      }
      lock.write(false);
    }
  }
  timer.stop();
  t = timer.elapsed() - t;

  writeln("eachExploredTree = ", eachExploredTree);

  exploredTree += (+ reduce eachExploredTree);
  exploredSol += (+ reduce eachExploredSol);

  writeln("Search on GPU completed");
  writeln("Size of the explored tree: ", exploredTree);
  writeln("Number of explored solutions: ", exploredSol);
  writeln("Elapsed time: ", t, " [s]\n");

  /*
    Step 3: We complete the depth-first search on CPU.
  */
  writeln("poolSize = ", pool.size);
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
