module Problem_NQueens
{
  /* use aux; */
  use GPU;
  use List;
  use Time;
  use CTypes;

  use Problem;

  class Problem_NQueens : Problem
  {
    var N: int; // size of the problem (number of queens)

    proc init(const n: int): void
    {
      this.N = n;
    }

    override proc copy()
    {
      return new Problem_NQueens(this.N);
    }

    proc isSafe(const board: c_ptr(c_int), const queen_num: int, const row_pos: c_int): bool
    {
      // For each queen before this one
      for i in 0..#queen_num {
        // Get the row position
        const other_row_pos: c_int = board[i];

        // Check diagonals
        if (other_row_pos == row_pos - (queen_num - i) ||
            other_row_pos == row_pos + (queen_num - i)) {
          return false;
        }
      }

      return true;
    }

    override proc decompose(type Node, const parent: Node, ref tree_loc: int, ref num_sol: int,
      best: atomic int, ref best_task: int): list
    {
      var childList: list(Node);

      const depth: int = parent.depth;

      if (depth == this.N) { // All queens are placed
        num_sol += 1;
      }
      for j in depth..this.N-1 {
        if isSafe(parent.board, depth, parent.board[j]) {
          var child = new Node(parent);
          //swap(child.board[depth], child.board[j]);
          var tmp = child.board[depth];
          child.board[depth] = child.board[j];
          child.board[j] = tmp;
          child.depth += 1;
          childList.append(child);
          tree_loc += 1;
        }
      }

      return childList;
    }

    proc decompose_gpu(type Node, const bufNodes: [] Node, ref tree_loc: int, ref num_sol: int,
      best: atomic int, ref best_task: int): [] Node
    {
      var children: [0..#this.N] Node;

      forall parent in bufNodes with (ref num_sol, ref tree_loc) {
        assertOnGpu();

        const depth: int = parent.depth;

        if (depth == this.N) { // All queens are placed
          num_sol += 1;
        }
        for j in depth..this.N-1 {
          // Check queen's safety
          var res = true;

          for i in 0..#depth {
            const other_row_pos: c_int = parent.board[i];

            if (other_row_pos == parent.board[j] - (depth - i) ||
                other_row_pos == parent.board[j] + (depth - i)) {
              res = false;
            }
          }

          // Generate children if any
          ref child = children[j];

          if res {
            for i in 0..#27 do child.board[i] = parent.board[i]; ////////////////////////////////////////
            child.depth = parent.depth;
            //swap(child.board[depth], child.board[j]);
            var tmp = child.board[depth];
            child.board[depth] = child.board[j];
            child.board[j] = tmp;
            child.depth += 1;
            tree_loc += 1;
          }
        }
      }

      var r: [0..#this.N] int = -1;
      var c1: int = -1;
      for i in r.domain {
        if (children[i].depth != 0) {
          c1 += 1;
          r[i] = i;
        }
      }
      var children_true: [0..c1] Node;
      if (c1 != -1) {
        var c2: int;
        for i in r.domain {
          if (r[i] != -1) {
            children_true[c2] = children[i];
            c2 += 1;
          }
        }
      }

      return children_true;
    }

    // No bounding in NQueens
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
      writeln("Resolution of the ", this.N, "-Queens instance");
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
      var tup = ("./chpl_nqueens_", this.N:string, ".txt");
      return "".join(tup);
    }

    override proc help_message(): void
    {
      writeln("\n  NQueens Benchmark Parameters:\n");
      writeln("   --N   int   Problem size (number of queens)\n");
    }

  } // end class

} // end module
