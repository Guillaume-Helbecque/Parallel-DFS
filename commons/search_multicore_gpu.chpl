module search_multicore_gpu
{
  use GPU;
  use List;
  use Time;
  use CTypes;
  use DistributedBag_DFS;

  use aux;
  use Problem;

  const BUSY: bool = false;
  const IDLE: bool = true;

  proc search_multicore_gpu(type Node, problem, const saveTime: bool, const activeSet: bool): void
  {
    var numTasks = here.maxTaskPar;

    // Global variables (best solution found and termination)
    var best: atomic int = problem.setInitUB();
    var allTasksIdleFlag: atomic bool = false;
    var eachTaskState: [0..#here.maxTaskPar] atomic bool = BUSY;

    // Statistics
    var eachExploredTree: [0..#numTasks] int;
    var eachExploredSol: [0..#numTasks] int;
    var eachMaxDepth: [0..#numTasks] int;
    var globalTimer: stopwatch;

    problem.print_settings();

    // ===============
    // INITIALIZATION
    // ===============

    var bag = new DistBag_DFS(Node, targetLocales = Locales);
    var root = new Node(problem);

    if activeSet {
      /*
        An initial set is sequentially computed and distributed across locales.
        We require at least 2 nodes per task.
      */
      var initSize: int = 2 * numTasks * numLocales;
      var initList: list(Node);
      initList.append(root);

      var best_task: int = best.read();
      ref tree_loc = eachExploredTree[0];
      ref num_sol = eachExploredSol[0];
      ref max_depth = eachMaxDepth[0];

      // Computation of the initial set
      while (initList.size < initSize) {
        var parent = initList.pop();

        {
          var children = problem.decompose(Node, parent, tree_loc, num_sol,
            max_depth, best, best_task);

          for elt in children do initList.insert(0, elt);
        }
      }

      // Static distribution of the set
      var seg, loc: int;
      for elt in initList {
        on Locales[loc % numLocales] do bag.add(elt, seg);
        loc += 1;
        if (loc % numLocales == 0) {
          loc = loc % numLocales;
          seg += 1;
        }
        if (seg == numTasks) then seg = 0;
      }

      initList.clear();
    }
    else {
      /*
        In that case, there is only one node in the bag (task 0 of locale 0).
      */
      bag.add(root, 0);
    }

    globalTimer.start();

    // =====================
    // PARALLEL EXPLORATION
    // =====================

    coforall taskId in 0..#numTasks {

      // Task variables
      var best_task: int = best.read();
      var taskState: bool = false;
      var counter: int = 0;
      ref tree_loc = eachExploredTree[taskId];
      ref num_sol = eachExploredSol[taskId];
      ref max_depth = eachMaxDepth[taskId];

      // Exploration of the tree
      while true do {

        // Try to remove an element
        var (hasWork, parent): (int, Node) = bag.remove(taskId);

        /*
          Check (or not) the termination condition regarding the value of 'hasWork':
            'hasWork' = -1 : remove() fails              -> check termination
            'hasWork' =  0 : remove() prematurely fails  -> continue
            'hasWork' =  1 : remove() succeeds           -> decompose
        */
        if (hasWork == 1) {
          if taskState {
            taskState = false;
            eachTaskState[taskId].write(BUSY);
          }
        }
        else if (hasWork == 0) {
          if !taskState {
            taskState = true;
            eachTaskState[taskId].write(IDLE);
          }
          continue;
        }
        else {
          if !taskState {
            taskState = true;
            eachTaskState[taskId].write(IDLE);
          }
          if allIdle(eachTaskState, allTasksIdleFlag) {
            break;
          }
          continue;
        }

        // Decompose an element
        var children = problem.decompose(Node, parent, tree_loc, num_sol,
          max_depth, best, best_task);

        bag.addBulk(children, taskId);

        // Decompose on GPU
        var size = bag.bag!.segments[taskId].nElems;
        if (size >= 10) {
          writeln("Offloaded on GPU with size ", size);

          // Offload on Gpus
          var parents: [0..#size] Node;
          for i in 0..#size {
            parents[i] = bag.remove(taskId)[1];
          }
          // NOT IMPLEMENTED: 'bag.removeBulk(nElts, taskId)'

          var evals: [0..#problem.N*size] int = noinit; // problem.N breaks genericity... list ?
          coforall gpu in here.gpus with (ref evals, const in parents) do on gpu {
            evals = problem.evaluate_gpu(Node, parents);
          }

          var children = problem.process_children(Node, parents, evals, tree_loc,
            num_sol, max_depth, best, best_task);

          bag.addBulk(children, taskId);
        }

        writeln("======================== ", eachExploredSol);

        // Read the best solution found so far
        if (taskId == 0) {
          counter += 1;
          if (counter % 10000 == 0) then best_task = best.read();
        }

      }
    }

    globalTimer.stop();

    /* bag.clear(); */

    // ========
    // OUTPUTS
    // ========

    writeln("\nExploration terminated.");

    if saveTime {
      var path = problem.output_filepath();
      /* save_time(numTasks:c_int, globalTimer.elapsed():c_double, path.c_str()); */
    }

    problem.print_results(eachExploredTree, eachExploredSol, eachMaxDepth, best.read(), globalTimer);
  }
}
