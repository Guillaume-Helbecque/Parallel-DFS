/*
  Chapel backtracking algorithm to solve instances of the N-Queens problem.
*/

use Time;

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
  var size: int;

  proc init() {
    this.dom = 0..#CAPACITY;
    this.capacity = CAPACITY;
  }

  proc ref pushBack(node: Node) {
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
Implementation of the sequential N-Queens search.
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

// Sequential N-Queens search.
proc nqueens_search(ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var root = new Node(N);

  var pool = new SinglePool();

  pool.pushBack(root);

  var timer: stopwatch;
  timer.start();

  while true {
    var hasWork = 0;
    var parent = pool.popBack(hasWork);
    if !hasWork then break;
    decompose(parent, exploredTree, exploredSol, pool);
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

  nqueens_search(exploredTree, exploredSol, elapsedTime);

  print_results(exploredTree, exploredSol, elapsedTime);

  return 0;
}
