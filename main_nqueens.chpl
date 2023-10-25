module main_queens
{
  // Common modules
  use aux;
  use search_sequential;
  use search_multicore;
  use search_distributed;
  use search_sequential_gpu;
  use search_multicore_gpu;

  // NQueens-specific modules
  use Node_NQueens;
  use Problem_NQueens;

  // Common options
  config const mode: string    = "multicore"; // sequential, multicore, distributed
  config const activeSet: bool = false;
  config const saveTime: bool  = false;

  // NQueens-specific option
  config const N: int = 13;
  config const g: int = 1;

  proc main(args: [] string): int
  {
    // Initialization of the problem
    var nqueens = new Problem_NQueens(N, g);

    // Helper
    for a in args[1..] {
      if (a == "-h" || a == "--help") {
        common_help_message();
        nqueens.help_message();

        return 1;
      }
    }

    // Search
    select mode {
      when "sequential" {
        if activeSet then warning("Cannot use `activeSet` in sequential mode.");
        /* search_sequential(Node_NQueens, nqueens, saveTime); */
      }
      when "multicore" {
        /* search_multicore(Node_NQueens, nqueens, saveTime, activeSet); */
      }
      when "distributed" {
        /* search_distributed(Node_NQueens, nqueens, saveTime, activeSet); */
      }
      when "sequential-gpu" {
        if activeSet then warning("Cannot use `activeSet` in sequential mode.");
        search_sequential_gpu(Node_NQueens, nqueens, saveTime);
      }
      when "multicore-gpu" {
        /* search_multicore_gpu(Node_NQueens, nqueens, saveTime, activeSet); */
      }
      otherwise {
        halt("ERROR - Unknown execution mode");
      }
    }

    return 0;
  }
}
