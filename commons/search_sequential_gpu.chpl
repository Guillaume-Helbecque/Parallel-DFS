module search_sequential_gpu
{
  use GPU;
  use List;
  use Time;
  use CTypes;
  use GpuDiagnostics;

  use aux;
  use Problem;

  config const minSize = 25;
  config const maxSize = 50000;

  /*******************************************************************************
  Implementation of a dynamic-sized single pool data structure.
  Its initial capacity is 1024, and we reallocate a new container with double
  the capacity when it is full. Since we perform only DFS, it only supports
  'pushBack' and 'popBack' operations.
  *******************************************************************************/

  config param CAPACITY = 1024;

  record SinglePool {
    type eltType;
    var dom: domain(1);
    var elements: [dom] eltType;
    var capacity: int;
    var size: int;

    proc init(type elt_type) {
      this.eltType = elt_type;
      this.dom = 0..#CAPACITY;
      this.capacity = CAPACITY;
    }

    proc ref pushBack(node){
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

      var default: this.eltType;
      return default;
    }
  }

  proc search_sequential_gpu(type Node, problem, const saveTime: bool): void
  {
    var best: int = problem.getInitBound();
    var best_at: atomic int = best;

    // Statistics
    var exploredTree: int;
    var exploredSol: int;
    var maxDepth: int;
    var globalTimer: stopwatch;

    problem.print_settings();

    // ===============
    // INITIALIZATION
    // ===============

    var pool = new SinglePool(Node);
    var root = new Node(problem);

    pool.pushBack(root);

    startGpuDiagnostics();
    globalTimer.start();

    // =====================
    // PARALLEL EXPLORATION
    // =====================

    // Exploration of the tree
    while true do {

      // Remove an element
      var hasWork = 0;
      var parent = pool.popBack(hasWork);
      if !hasWork then break;

      // Decompose the element
      var children = problem.decompose(Node, parent, exploredTree, exploredSol,
        maxDepth, best_at, best);

      for child in children do pool.pushBack(child);

      var size = min(pool.size, maxSize);

      if (size >= minSize) {
        var parents: [0..#size] Node = noinit;
        for i in 0..#size {
          var hasWork = 0;
          parents[i] = pool.popBack(hasWork);
          if !hasWork then break;
        }

        var evals: [0..#problem.length*parents.size] uint(8) = noinit;

        // Offload on GPUs
        on here.gpus[0] {
          const parents_d = parents; // host-to-device
          evals = problem.evaluate_gpu(parents_d); // device-to-host + kernel
        }

        var children = problem.generate_children(Node, parents, evals, exploredTree,
          exploredSol, maxDepth, best_at, best);

        for child in children do pool.pushBack(child);
      }
    }

    globalTimer.stop();
    stopGpuDiagnostics();

    // ========
    // OUTPUTS
    // ========

    writeln("\nExploration terminated.");
    writeln("kernel_launch: ", getGpuDiagnostics().kernel_launch);
    writeln("host_to_device: ", getGpuDiagnostics().host_to_device);
    writeln("device_to_host: ", getGpuDiagnostics().device_to_host);
    writeln("device_to_device: ", getGpuDiagnostics().device_to_device);

    if saveTime {
      var path = problem.output_filepath();
      save_time(1, globalTimer.elapsed():c_double, path.c_str());
    }

    writeln("\n=================================================");
    writeln("Size of the explored tree: ", exploredTree);
    writeln("Number of explored solutions: ", exploredSol);
    writeln("Optimal makespan: ", best);
    writeln("Elapsed time: ", globalTimer.elapsed(), " [s]");
    writeln("=================================================\n");

    /* problem.print_results(exploredTree, exploredSol, maxDepth, best,
      globalTimer.elapsed()); */
  }
}
