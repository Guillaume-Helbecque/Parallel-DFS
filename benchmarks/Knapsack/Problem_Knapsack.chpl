use IO;
use List;
use Time;
use CTypes;
use Path;

use Problem;

class Problem_Knapsack : Problem
{
  var name: string;         // file name
  var N: int;               // number of items
  var W: real;              // maximum weight of the bag
  var weight: [0..#N] real; // items' weight
  var profit: [0..#N] real; // items' profit

  // initialisation from a file
  proc init(const fileName: string): void
  {
    this.name = fileName;

    var tup = ("./benchmarks/Knapsack/instances/", fileName);
    var path = "".join(tup);

    var f = open(path, ioMode.r);
    var channel = f.reader();

    this.N = channel.read(int);
    this.W = channel.read(int);
    this.weight = channel.read([0..#this.N] int);
    this.profit = channel.read([0..#this.N] int);

    channel.close();
    f.close();
  }

  // initialisation from parameters
  proc init(const n: int, const w: real, const we: [] real, const pr: [] real): void
  {
    this.N      = n;
    this.W      = w;
    this.weight = we;
    this.profit = pr;
  }

  override proc copy()
  {
    return new Problem_Knapsack(this.N, this.W, this.profit, this.weight);
  }

  inline proc arrMultSom(const c_a: c_array, const chpl_a: [], const depth: int)
  {
    var res: real;
    for i in 0..#depth do res += c_a[i] * chpl_a[i];
    return res;
  }

  override proc decompose(type Node, const parent: Node, ref tree_loc: int, ref num_sol: int,
    best: atomic int, ref best_task: int): list
  {
    var childList: list(Node);

    for i in 0..1 by -1 {
      var child = new Node(parent);
      child.items[parent.depth] = i:c_int;
      child.depth += 1;
      if (arrMultSom(child.items, weight, child.depth) <= W){
        if (child.depth == N - 1){
          num_sol += 1;
          var eval = arrMultSom(child.items, profit, child.depth):int;
          if (best_task <= eval){
            best_task = eval;
            best.write(best_task);
          }
        }
        else {
          childList.append(child);
          tree_loc += 1;
        }
      }
    }

    return childList;
  }

  // No bounding
  override proc setInitUB(): int
  {
    return 0;
  }

  // =======================
  // Utility functions
  // =======================

  override proc print_settings(): void
  {
    writeln("\n=================================================");
    writeln("Resolution of the 0/1-Knapsack instance: ", this.name);
    writeln("  number of items: ", this.N);
    writeln("  capacity of the bag: ", this.W);
    writeln("  items's profit: ", this.profit);
    writeln("  items's weight: ", this.weight);
    writeln("=================================================");
  }

  override proc print_results(const subNodeExplored: [] int, const subSolExplored: [] int,
    const subDepthReached: [] int, const best: int, const timer: stopwatch): void
  {
    var treeSize: int = (+ reduce subNodeExplored);
    var nbSol: int = (+ reduce subSolExplored);
    var par_mode: string = if (numLocales == 1) then "tasks" else "locales";

    writeln("\n=================================================");
    writeln("Optimum found: ", best);
    writeln("Size of the explored tree: ", treeSize);
    /* writeln("Size of the explored tree per locale: ", sizePerLocale); */
    writeln("% of the explored tree per ", par_mode, ": ", 100 * subNodeExplored:real / treeSize:real);
    writeln("Number of explored solutions: ", nbSol);
    /* writeln("Number of explored solutions per locale: ", numSolPerLocale); */
    writeln("Elapsed time: ", timer.elapsed(), " [s]");
    writeln("=================================================\n");
  }

  override proc output_filepath(): string
  {
    var tup = ("./chpl_knapsack_", splitExt(this.name)[0], ".txt");
    return "".join(tup);
  }

  override proc help_message(): void
  {
    writeln("\n  Knapsack Benchmark Parameters:\n");
    writeln("   --name   str   File containing the data\n");
  }

} // end class
