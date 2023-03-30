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
    var board: c_array(c_int, NMax);
    var depth: int;

    // default-initializer
    proc init()
    {}

    // root-initializer
    proc init(problem)
    {
      this.complete();
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
