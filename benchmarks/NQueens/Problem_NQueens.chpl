module Problem_NQueens
{
  use aux;
  use GPU;
  use List;
  use Time;
  use CTypes;

  use Problem;

  const SAFE     =  1;
  const NOT_SAFE = -1;

  class Problem_NQueens : Problem
  {
    var N: int; // size of the problem (number of queens)
    var g: int; // number of safety check(s) per evaluation

    proc init(const n: int, const G: int): void
    {
      this.N = n;
      this.g = G;
    }

    override proc copy()
    {
      return new Problem_NQueens(this.N, this.g);
    }

    proc isSafe(const board: c_ptr(c_int), const queen_num: int, const row_pos: c_int): bool
    {
      for 0..#this.g {
        // For each queen before this one
        for i in 0..#queen_num {
          // Get the row position
          const other_row_pos = board[i];

          // Check diagonals
          if (other_row_pos == row_pos - (queen_num - i) ||
              other_row_pos == row_pos + (queen_num - i)) {
            return false;
          }
        }
      }

      return true;
    }

    override proc decompose(type Node, const parent: Node, ref tree_loc: int, ref num_sol: int,
      ref max_depth: int, best: atomic int, ref best_task: int): list
    {
      var children: list(Node);

      const depth = parent.depth;

      if (depth == this.N) { // All queens are placed
        num_sol += 1;
      }
      for j in depth..this.N-1 {
        if isSafe(parent.board, depth, parent.board[j]) {
          var child = new Node(parent);
          swap(child.board[depth], child.board[j]);
          child.depth += 1;
          children.pushBack(child);
          tree_loc += 1;
        }
      }

      return children;
    }

    override proc evaluate_gpu(type Node, const parents: [] Node): [] int
    {
      const size: int = parents.size;
      const G = this.g; // WORKAROUND

      var status_loc: [0..#this.N*size] int = SAFE;
      var parents_loc: [0..#size] Node;// = parents; // Github issue #22519
      for i in 0..#size do parents_loc[i] = parents[i]; // WORKAROUND

      foreach pid in 0..#this.N*size {
        assertOnGpu();

        const parentId = pid / this.N;
        const k = pid % this.N;
        const parent = parents_loc[parentId];
        const depth = parent.depth;

        if (k >= depth) {
          for 0..#G { // ISSUE: Cannot put 'this.g'
            // Check queen's safety
            for i in 0..#depth {
              const other_row_pos = parent.board[i];

              if (other_row_pos == parent.board[k] - (depth - i) ||
                  other_row_pos == parent.board[k] + (depth - i)) {
                status_loc[pid] = NOT_SAFE;
              }
            }
          }
        }
      } // end foreach on GPU

      return status_loc;
    }

    override proc process_children(type Node, const parents: [] Node, const status: [] int, ref tree_loc: int,
      ref num_sol: int, ref max_depth: int, best: atomic int, ref best_task: int): list
    {
      var children: list(Node);

      // Generate children if any
      for parentId in parents.domain {
        const parent = parents[parentId];
        const depth = parent.depth;

        if (depth == this.N) {
          num_sol += 1;
        }
        for j in depth..this.N-1 {
          if (status[j + parentId * this.N] == SAFE) {
            var child = new Node(parent);
            swap(child.board[depth], child.board[j]);
            child.depth += 1;
            children.pushBack(child);
            tree_loc += 1;
          }
        }
      }

      return children;
    }

    // No bounding in NQueens
    override proc setInitUB(): int
    {
      return 0;
    }

    override proc length
    {
      return this.N;
    }

    // =======================
    // Utility functions
    // =======================

    override proc print_settings(): void
    {
      writeln("\n=================================================");
      writeln("Resolution of the ", this.N, "-Queens instance");
      writeln("  with ", this.g, " safety check(s) per evaluation");
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
      writeln("Elapsed time: ", timer.elapsed(), " [s]");
      writeln("=================================================\n");
    }

    override proc output_filepath(): string
    {
      var tup = ("./chpl_nqueens_", this.N:string, "_", this.g:string, ".txt");
      return "".join(tup);
    }

    override proc help_message(): void
    {
      writeln("\n  NQueens Benchmark Parameters:\n");
      writeln("   --N   int   Problem size (number of queens)\n");
      writeln("   --g   int   Number of safety check(s) per evaluation\n");
    }

  } // end class

} // end module
