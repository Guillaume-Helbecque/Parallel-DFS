module Problem
{
  class Problem
  {
    proc copy()
    {
      halt("Error - copy() not implemented");
    }

    proc decompose(type Node, const parent: Node, ref tree_loc: int, ref num_sol: int,
      ref max_depth: int, best: atomic int, ref best_task: int, ref pool)
    {
      halt("Error - decompose() not implemented");
    }

    proc evaluate_gpu(const parents_d, const size)
    {
      halt("Error - evaluate_gpu() not implemented");
    }

    proc generate_children(type Node, const parents: [] Node, const evals: [] uint(8), ref tree_loc: int,
      ref num_sol: int, ref max_depth: int, best: atomic int, ref best_task: int, ref pool)
    {
      halt("Error - process_children() not implemented");
    }

    proc getInitBound(): int
    {
      halt("Error - getInitBound() not implemented");
    }

    proc length
    {
      halt("Error - length not implemented");
    }

    // =======================
    // Utility functions
    // =======================

    proc print_settings(): void
    {
      halt("Error - print_settings() not implemented");
    }

    proc print_results(const subNodeExplored: [] int, const subSolExplored: [] int,
      const subDepthReached: [] int, const best: int, const elapsedTime: real): void
    {
      halt("Error - print_results() not implemented");
    }

    proc output_filepath(): string
    {
      halt("Error - output_filepath() not implemented");
    }

    proc help_message(): void
    {
      halt("Error - help_message() not implemented");
    }
  } // end class

} // end module
