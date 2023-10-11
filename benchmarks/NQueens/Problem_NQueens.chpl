module Problem_NQueens
{
  use aux;
  use List;
  use CTypes;

  use Problem;

  const SAFE = 1;

  class Problem_NQueens : Problem
  {
    var N: int; // number of queens
    var G: int; // number of safety check(s) per evaluation

    proc init(const n: int, const g: int): void
    {
      this.N = n;
      this.G = g;
    }

    override proc copy()
    {
      return new Problem_NQueens(this.N, this.G);
    }

    proc isSafe(const board, const queen_num: int, const row_pos: c_int): bool
    {
      for 0..#this.G {
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
      ref max_depth: int, best: atomic int, ref best_task: int): list(?)
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

    // Evaluate a bulk of parent nodes on GPU.
    override proc evaluate_gpu(type Node, const parents: [] Node): [] int
    {
      const size: int = parents.size;

      var status_loc: [0..#this.N*size] int = SAFE;
      var parents_loc: [0..#size] Node = parents;

      @assertOnGpu
      foreach pid in 0..#this.N*size {

        const parentId = pid / this.N;
        const k = pid % this.N;
        const parent = parents_loc[parentId];
        const depth = parent.depth;

        // If child 'k' is not scheduled, we evaluate its safety 'G' times, otherwise 0.
        const G_notScheduled: int = this.G * (k >= depth);
        for 0..#G_notScheduled {
          // Check queen's safety
          for i in 0..#depth {
            const other_row_pos = parent.board[i];
            const isNotSafe: int = (other_row_pos == parent.board[k] - (depth - i) ||
              other_row_pos == parent.board[k] + (depth - i));

            status_loc[pid] *= (1 - isNotSafe);
          }
        }
      } // end foreach on GPU

      return status_loc;
    }

    // Generate children nodes (evaluated by GPU) on CPU.
    override proc generate_children(type Node, const parents: [] Node, const status: [] int, ref tree_loc: int,
      ref num_sol: int, ref max_depth: int, best: atomic int, ref best_task: int): list(?)
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
    override proc getInitBound(): int
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
      writeln("  with ", this.G, " safety check(s) per evaluation");
      writeln("=================================================");
    }

    override proc print_results(const subNodeExplored: [] int, const subSolExplored: [] int,
      const subDepthReached: [] int, const best: int, const elapsedTime: real): void
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
      writeln("Elapsed time: ", elapsedTime, " [s]");
      writeln("=================================================\n");
    }

    override proc output_filepath(): string
    {
      return "./chpl_nqueens_" + this.N:string + "_" + this.G:string + ".txt";
    }

    override proc help_message(): void
    {
      writeln("\n  N-Queens Benchmark Parameter:\n");
      writeln("   --N   int   problem size (number of queens)\n");
      writeln("   --g   int   Number of safety check(s) per evaluation\n");
    }

  } // end class

} // end module
