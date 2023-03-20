module search_distributed
{
  use List;
  use Time;
  use CTypes;
  use PrivateDist;
  use DistributedBag_DFS;
  use AllLocalesBarriers;

  use aux;
  use statistics;

  use Problem;

  const BUSY: bool = false;
  const IDLE: bool = true;

  proc search_distributed(type Node, problem, const saveTime: bool, const activeSet: bool): void
  {
    // Global variables (best solution found and termination)
    var best: atomic int = problem.setInitUB();
    const PrivateSpace: domain(1) dmapped Private(); // map each index to a locale
    var eachLocaleState: [PrivateSpace] atomic bool = BUSY;
    var allLocalesIdleFlag: atomic bool = false;
    allLocalesBarrier.reset(here.maxTaskPar); // configuration of the global barrier

    // Counters and timers (for analysis)
    var eachExploredTree: [PrivateSpace] int = 0;
    var eachExploredSol: [PrivateSpace] int = 0;
    var eachMaxDepth: [PrivateSpace] int = 0;
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
      var initSize: int = 2 * here.maxTaskPar * numLocales;
      var initList: list(Node);
      initList.append(root);

      var best_task, best_locale: int = best.read();
      ref tree_loc = eachExploredTree[0];
      ref num_sol = eachExploredSol[0];

      // Computation of the initial set
      while (initList.size < initSize) {
        var parent: Node = initList.pop();

        {
          var children = problem.decompose(Node, parent, tree_loc, num_sol, best, best_task);

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
        if (seg == here.maxTaskPar) then seg = 0;
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

    coforall loc in Locales with (const ref problem) do on loc {

      var numTasks = here.maxTaskPar;
      var problem_loc = problem.copy();

      // Local variables (best solution found and termination)
      var best_locale: int = problem.setInitUB();
      var allTasksIdleFlag: atomic bool = false;
      var eachTaskState: [0..#numTasks] atomic bool = BUSY;

      // Counters and timers (for analysis)
      var eachLocalExploredTree: [0..#numTasks] int = 0;
      var eachLocalExploredSol: [0..#numTasks] int = 0;

      coforall tid in 0..#numTasks with (ref best_locale) {

        // Task variables (best solution found)
        var best_task: int = best_locale;
        var taskState, locState: bool = false;
        ref tree_loc = eachLocalExploredTree[tid];
        ref num_sol = eachLocalExploredSol[tid];

        // Counters and timers (for analysis)
        var count, counter: int = 0;

        allLocalesBarrier.barrier(); // synchronization of tasks

        while true do {
          counter += 1;

          // Check if the global termination flag is set or not
          if (counter % 10000 == 0) {
            if allLocalesIdleFlag.read() {
              break;
            }
          }

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
            if locState {
              locState = false;
              eachLocaleState[here.id].write(BUSY);
            }
          }
          else if (hasWork == 0) {
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
            if allIdle(eachTaskState, allTasksIdleFlag) {
              if !locState {
                locState = true;
                eachLocaleState[tid].write(IDLE);
              }
              if allIdle(eachLocaleState, allLocalesIdleFlag) {
                break;
              }
            } else {
              if locState {
                locState = false;
                eachLocaleState[here.id].write(BUSY);
              }
            }
            continue;
          }

          // Decompose an element
          {
            var children = problem_loc.decompose(Node, parent, tree_loc, num_sol, best, best_task);

            bag.addBulk(children, tid);
          }

          // Read the best solution found so far
          if (tid == 0) {
            count += 1;
            if (count % 10000 == 0) then best_locale = best.read();
          }

          best_task = best_locale;
        }
      } // end coforall tasks

      eachExploredTree[here.id] += (+ reduce eachLocalExploredTree);
      eachExploredSol[here.id] += (+ reduce eachLocalExploredSol);
    } // end coforall locales

    globalTimer.stop();

    /* bag.clear(); */

    // ========
    // OUTPUTS
    // ========

    writeln("\nExploration terminated.");

    if saveTime {
      var path = problem.output_filepath();
      /* save_time(numLocales:c_int, globalTimer.elapsed(TimeUnits.seconds):c_double, path.c_str()); */
    }

    problem.print_results(eachExploredTree, eachExploredSol, eachMaxDepth, best.read(), globalTimer);
  }

}
