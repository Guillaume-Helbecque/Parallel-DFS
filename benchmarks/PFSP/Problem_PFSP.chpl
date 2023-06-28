module Problem_PFSP
{
  use aux;
  use GPU;
  use List;
  use Time;
  use CTypes;

  use Problem;
  use Instances;
  use Header_chpl_c_PFSP;

  class Problem_PFSP : Problem
  {
    var name: string;
    var jobs: c_int;
    var machines: c_int;

    var lb_name: string;
    var lbound1: c_ptr(bound_data);
    var lbound2: c_ptr(johnson_bd_data);

    var branching: int;
    var ub_init: string;

    proc init(const fileName: string, const lb: string, const rules: int, const ub: string): void
    {
      this.name = fileName;

      var inst = new Instance();
      if (fileName[0..1] == "ta") then inst = new Instance_Taillard(fileName);
      else if (fileName[0..2] == "VFR") then inst = new Instance_VRF(fileName);
      else halt("Error - Unknown PFSP instance class");

      this.jobs     = inst.get_nb_jobs();
      this.machines = inst.get_nb_machines();

      if (lb == "lb1" || lb == "lb1_d" || lb == "lb2") then this.lb_name = lb;
      else halt("Error - Unsupported lower bound");

      this.lbound1 = new_bound_data(jobs, machines);
      inst.get_data(lbound1.deref().p_times);
      fill_min_heads_tails(lbound1);

      if (lb == "lb2"){
        this.lbound2 = new_johnson_bd_data(lbound1/*, LB2_FULL*/);
        fill_machine_pairs(lbound2/*, LB2_FULL*/);
        fill_lags(lbound1, lbound2);
        fill_johnson_schedules(lbound1, lbound2);
      }

      this.branching = rules;

      if (ub == "opt" || ub == "inf") then this.ub_init = ub;
      else halt("Error - Unsupported upper bound");
    }

    proc deinit()
    {
      free_bound_data(this.lbound1);
      if (this.lb_name == "lb2") then free_johnson_bd_data(this.lbound2);
    }

    // TODO: Implement a copy initializer, to avoid re-computing all the data
    override proc copy()
    {
      return new Problem_PFSP(this.name, this.lb_name, this.branching, this.ub_init);
    }

    proc decompose_lb1(type Node, const parent: Node, ref tree_loc: int, ref num_sol: int,
      best: atomic int, ref best_task: int): list
    {
      var childList: list(Node); // list containing the child nodes

      // Treatment of childs
      for i in parent.limit1+1..parent.limit2-1 {
        var child = new Node(parent);
        swap(child.prmu[child.depth], child.prmu[i]);
        child.depth  += 1;
        child.limit1 += 1;

        var lowerbound: c_int = lb1_bound(lbound1, child.prmu, child.limit1:c_int, jobs);

        if (child.depth == jobs) { // if child leaf
          num_sol += 1;

          if (lowerbound < best_task) { // if child feasible
            best_task = lowerbound;
            best.write(lowerbound);
          }
        } else { // if not leaf
          if (lowerbound < best_task) { // if child feasible
            tree_loc += 1;
            childList.pushBack(child);
          }
        }
      }

      return childList;
    }

    proc decompose_lb1_d(type Node, const parent: Node, ref tree_loc: int, ref num_sol: int,
      best: atomic int, ref best_task: int): list
    {
      var childList: list(Node); // list containing the child nodes

      var lb_begin = allocate(c_int, jobs);
      var BEGINEND: c_int = -1;

      lb1_children_bounds(this.lbound1, parent.prmu, parent.limit1:c_int, parent.limit2:c_int,
        lb_begin, nil, nil, nil, BEGINEND);

      // Treatment of childs
      for i in parent.limit1+1..parent.limit2-1 {

        if (parent.depth + 1 == jobs){ // if child leaf
          num_sol += 1;

          if (lb_begin[parent.prmu[i]] < best_task){ // if child feasible
            best_task = lb_begin[parent.prmu[i]];
            best.write(lb_begin[parent.prmu[i]]);
          }
        } else { // if not leaf
          if (lb_begin[parent.prmu[i]] < best_task){ // if child feasible
            var child = new Node(parent);
            child.depth += 1;

            if (branching == 0){ // if forward
              child.limit1 += 1;
              swap(child.prmu[child.limit1], child.prmu[i]);
            } else if (branching == 1){ // if backward
              child.limit2 -= 1;
              swap(child.prmu[child.limit2], child.prmu[i]);
            }

            childList.pushBack(child);
            tree_loc += 1;
          }
        }

      }

      deallocate(lb_begin);

      return childList;
    }

    proc decompose_lb2(type Node, const parent: Node, ref tree_loc: int, ref num_sol: int,
      best: atomic int, ref best_task: int): list
    {
      var childList: list(Node); // list containing the child nodes

      for i in parent.limit1+1..parent.limit2-1 {
        var child = new Node(parent);
        swap(child.prmu[child.depth], child.prmu[i]);
        child.depth  += 1;
        child.limit1 += 1;

        var lowerbound: c_int = lb2_bound(lbound1, lbound2, child.prmu, child.limit1:c_int, jobs, best_task:c_int);

        if (child.depth == jobs) { // if child leaf
          num_sol += 1;

          if (lowerbound < best_task) { // if child feasible
            best_task = lowerbound;
            best.write(lowerbound);
          }
        } else { // if not leaf
          if (lowerbound < best_task) { // if child feasible
            tree_loc += 1;
            childList.pushBack(child);
          }
        }

      }

      return childList;
    }

    override proc decompose(type Node, const parent: Node, ref tree_loc: int, ref num_sol: int,
      ref max_depth: int, best: atomic int, ref best_task: int): list
    {
      select lb_name {
        when "lb1" {
          return decompose_lb1(Node, parent, tree_loc, num_sol, best, best_task);
        }
        when "lb1_d" {
          return decompose_lb1_d(Node, parent, tree_loc, num_sol, best, best_task);
        }
        when "lb2" {
          return decompose_lb2(Node, parent, tree_loc, num_sol, best, best_task);
        }
        otherwise {
          halt("Error - Unknown lower bound");
        }
      }
    }

    //TODO: implement evaluate_gpu_lb1
    //TODO: implement evaluate_gpu_lb1_d

    proc evaluate_gpu_lb2(type Node, const parents: [] Node): [] int
    {
      const size: int = parents.size;

      var status_loc: [0..#this.jobs*size] int;
      var parents_loc: [0..#size] Node;// = parents; // Github issue #22519
      for i in 0..#size do parents_loc[i] = parents[i]; // WORKAROUND

      foreach pid in 0..#this.jobs*size {
        assertOnGpu();

        //TODO: implement the evaluation of lowerbounds
        // ISSUE: The use of most of extern functions within a GPU eligible loop
        // is not supported (Chapel 1.31.0).
      }

      return status_loc;
    }

    override proc evaluate_gpu(type Node, const parents: [] Node): [] int
    {
      select lb_name {
        /* when "lb1" {
          return evaluate_gpu_lb1(Node, parents);
        } */
        /* when "lb1_d" {
          return evaluate_gpu_lb1_d(Node, parents);
        } */
        when "lb2" {
          return evaluate_gpu_lb2(Node, parents);
        }
        otherwise {
          halt("Error - Unknown lower bound");
        }
      }
    }

    //TODO: implement process_children_lb1
    //TODO: implement process_children_lb1_d

    proc process_children_lb2(type Node, const parents: [] Node, const status: [] int, ref tree_loc: int,
      ref num_sol: int, ref max_depth: int, best: atomic int, ref best_task: int): list
    {
      var children: list(Node);

      //TODO: implement the processing of children

      return children;
    }

    override proc process_children(type Node, const parents: [] Node, const status: [] int, ref tree_loc: int,
      ref num_sol: int, ref max_depth: int, best: atomic int, ref best_task: int): list
    {
      select lb_name {
        /* when "lb1" {
          return process_children_lb1(Node, parents, status, tree_loc, num_sol, max_depth, best, best_task);
        } */
        /* when "lb1_d" {
          return process_children_lb1_d(Node, parents, status, tree_loc, num_sol, max_depth, best, best_task);
        } */
        when "lb2" {
          return process_children_lb2(Node, parents, status, tree_loc, num_sol, max_depth, best, best_task);
        }
        otherwise {
          halt("Error - Unknown lower bound");
        }
      }
    }

    override proc setInitUB(): int
    {
      var inst = new Instance();
      if (this.name[0..1] == "ta") then inst = new Instance_Taillard(this.name);
      else if (this.name[0..2] == "VFR") then inst = new Instance_VRF(this.name);
      else halt("Error - Unknown PFSP instance class");

      if (this.ub_init == "inf"){
        return 999999;
      }
      else {
        return inst.get_ub();
      }
    }

    override proc length
    {
      return this.jobs;
    }

    // =======================
    // Utility functions
    // =======================

    override proc print_settings(): void
    {
      writeln("\n=================================================");
      writeln("PFSP instance: ", name, " (m = ", machines, ", n = ", jobs, ")");
      writeln("Initial upper bound: ", setInitUB());
      writeln("Lower bound function: ", lb_name);
      writeln("Branching rules: ", (1-branching)*"forward" + branching*"backward");
      writeln("=================================================");
    }

    override proc print_results(const subNodeExplored: [] int, const subSolExplored: [] int,
      const subDepthReached: [] int, const best: int, const timer: stopwatch): void
    {
      var treeSize: int = (+ reduce subNodeExplored);
      var nbSol: int = (+ reduce subSolExplored);
      var par_mode: string = if (numLocales == 1) then "tasks" else "locales";

      writeln("\n=================================================");
      writeln("Size of the explored tree: ", treeSize);
      /* writeln("Size of the explored tree per locale: ", sizePerLocale); */
      writeln("% of the explored tree per ", par_mode, ": ", 100 * subNodeExplored:real / treeSize:real);
      writeln("Number of explored solutions: ", nbSol);
      /* writeln("Number of explored solutions per locale: ", numSolPerLocale); */
      writeln("Optimal makespan: ", best);
      writeln("Elapsed time: ", timer.elapsed(), " [s]");
      writeln("=================================================\n");
    }

    override proc output_filepath(): string
    {
      var tup = ("./chpl_pfsp_", name, "_", lb_name, "_",
        ((1-branching)*"forward" + branching*"backward"), ".txt");
      return "".join(tup);
    }

    override proc help_message(): void
    {
      writeln("\n  PFSP Benchmark Parameters:\n");
      writeln("   --inst  str   instance's name");
      writeln("   --lb    str   lower bound function (lb1, lb1_d, lb2)");
      writeln("   --br    int   branching rule (0: forward, 1: backward)");
      writeln("   --ub    str   upper bound initialization (opt, inf)\n");
    }

  } // end class

} // end module
