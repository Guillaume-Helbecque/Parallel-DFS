module Node_NQueens
{
  use CTypes;

  /*
    Maximum size of the problem. The exact number of N-Queens solutions is only
    known for N < 28.
  */
  config param NMax: int = 27;

  record Node_NQueens
  {
    /*
      NOTE: Using tuple instead of c_array allows bulk transfer between host and
      device, and highly improves the performance using GPUs.
      See Chapel Github issue #22519.
    */
    var board: NMax*c_int;
    var depth: int;

    // default-initializer
    proc init()
    {}

    // root-initializer
    proc init(problem)
    {
      init this;
      for i in 0..#problem.N do this.board[i] = i:c_int;
    }

    // copy-initializer
    proc init(other: Node_NQueens)
    {
      this.board = other.board;
      this.depth = other.depth;
    }

    proc deinit()
    {}
  }

}
