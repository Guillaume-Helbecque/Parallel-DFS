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

    var pool: list(Node);
    var root = new Node(problem);

    pool.pushBack(root);

    startGpuDiagnostics();
    globalTimer.start();

    // =====================
    // PARALLEL EXPLORATION
    // =====================

    // Exploration of the tree
    while !pool.isEmpty() do {

      // Remove an element
      var parent: Node = pool.popBack();

      // Decompose the element
      var children = problem.decompose(Node, parent, exploredTree, exploredSol,
        maxDepth, best_at, best);

      pool.pushBack(children);

      var size = min(pool.size, maxSize);

      if (size >= minSize) {
        var parents: [0..#size] Node;
        for p in parents.domain do parents[p] = pool.popBack();

        var evals: [0..#problem.length*parents.size] uint(8) = noinit;

        // Offload on GPUs
        on here.gpus[0] {
          const parents_d = parents; // host-to-device
          evals = problem.evaluate_gpu(parents_d); // device-to-host + kernel
        }

        var children = problem.generate_children(Node, parents, evals, exploredTree,
          exploredSol, maxDepth, best_at, best);

        pool.pushBack(children);
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
