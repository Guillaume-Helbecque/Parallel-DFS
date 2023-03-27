module search_multicore
{
  use GPU;
  use List;
  use Time;
  use CTypes;
  use DistributedBag_DFS;

  use aux;
  use statistics;

  use Problem;

  const BUSY: bool = false;
  const IDLE: bool = true;

  proc search_multicore(type Node, problem, const saveTime: bool, const activeSet: bool): void
  {
    var numTasks = here.maxTaskPar;

    // Global variables (best solution found and termination)
    var best: atomic int = problem.setInitUB();
    var allTasksIdleFlag: atomic bool = false;
    var eachTaskState: [0..#here.maxTaskPar] atomic bool = BUSY;

    // Counters and timers (for analysis)
    var eachLocalExploredTree: [0..#numTasks] int = 0;
    var eachLocalExploredSol: [0..#numTasks] int = 0;
    var eachMaxDepth: [0..#numTasks] int = 0;
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
      ref tree_loc = eachLocalExploredTree[0];
      ref num_sol = eachLocalExploredSol[0];

      // Computation of the initial set
      while (initList.size < initSize){
        var parent: Node = initList.pop();

        {
          var children = problem.decompose(Node, parent, tree_loc, num_sol,
            best, best_task);

          for elt in children do initList.insert(0, elt);
        }
      }

      // Static distribution of the set
      var seg, loc: int = 0;
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

    coforall tid in 0..#numTasks {

      // Task variables (best solution found)
      var best_task: int = best.read();
      var taskState: bool = false;
      ref tree_loc = eachLocalExploredTree[tid];
      ref num_sol = eachLocalExploredSol[tid];

      // Counters and timers (for analysis)
      var count: int = 0;

      // Exploration of the tree
      while true do {

        // Try to remove an element
        var (hasWork, parent): (int, Node) = bag.remove(tid);

        /*
          Check (or not) the termination condition regarding the value of 'hasWork':
            'hasWork' = -1 : remove() fails              -> check termination
            'hasWork' =  0 : remove() prematurely fails  -> continue
            'hasWork' =  1 : remove() succeeds           -> decompose
        */
        if (hasWork == 1){
          if taskState {
            taskState = false;
            eachTaskState[tid].write(BUSY);
          }
        }
        else if (hasWork == 0){
          if !taskState {
            taskState = true;
            eachTaskState[tid].write(IDLE);
          }
          continue;
        }
        else {
          if !taskState {
            taskState = true;
            eachTaskState[tid].write(IDLE);
          }
          if allIdle(eachTaskState, allTasksIdleFlag){
            break;
          }
          continue;
        }

        // Decompose an element
        {
          var children: list(Node);

          /* writeln("here ", here);
          writeln("here.gpus[0]= ", here.gpus[0]);
          on here.gpus[0] {
            writeln("hello from ", here); */
            children = problem.decompose(Node, parent, tree_loc, num_sol, best, best_task);
          /* } */

          bag.addBulk(children, tid);
        }

        // Read the best solution found so far
        if (tid == 0){
          count += 1;
          if (count % 10000 == 0) then best_task = best.read();
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

    problem.print_results(eachLocalExploredTree, eachLocalExploredSol, eachMaxDepth, best.read(), globalTimer);
  }
}
