module Node_NQueens
{
  /*
    Maximum size of the problem. The exact number of N-Queens solutions is only
    known for N < 28.
  */
  config param NMax: int = 21;

  record Node_NQueens
  {
    /*
      NOTE: Using tuple instead of c_array allows bulk transfer between host and
      device, and highly improves the performance using GPUs.
      See Chapel Github issue #22519.
    */
    var board: NMax*uint(8);
    var depth: uint(8);

    // default-initializer
    proc init()
    {}

    // root-initializer
    proc init(problem)
    {
      init this;
      for i in 0..#problem.N do this.board[i] = i:uint(8);
    }

    /*
      NOTE: This copy-initializer makes the Node type "non-trivial" for `noinit`.
      Perform manual copy in the code instead.
    */
    // copy-initializer
    /* proc init(other: Node_NQueens)
    {
      this.board = other.board;
      this.depth = other.depth;
    } */
  }

}
