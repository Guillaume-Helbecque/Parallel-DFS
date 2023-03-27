module main_knapsack
{
  // Common modules
  use CTypes;

  use aux;
  use search_multicore;
  use search_distributed;

  // Knapsack-specific modules
  use Node_Knapsack;
  use Problem_Knapsack;

  // Common options
  config const mode: string = "multicore"; // multicore, distributed
  config const activeSet: bool = false;
  config const saveTime: bool = false;

  // Knapsack-specific option
  config const name: string = "default.txt";

  proc main(args: [] string): int
  {
    // Initialization of the problem
    var knapsack = new Problem_Knapsack(name);

    // Helper
    for a in args[1..] {
      if (a == "-h" || a == "--help") {
        common_help_message();
        knapsack.help_message();

        return 1;
      }
    }

    // Parallel search
    select mode {
      when "multicore" {
        search_multicore(Node_Knapsack, knapsack, saveTime, activeSet);
      }
      when "distributed" {
        search_distributed(Node_Knapsack, knapsack, saveTime, activeSet);
      }
      otherwise {
        halt("ERROR - Unknown parallel execution mode");
      }
    }

    return 0;
  }
}
