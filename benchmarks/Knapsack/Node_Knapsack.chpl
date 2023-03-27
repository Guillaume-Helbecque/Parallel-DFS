module Node_Knapsack
{
  use CTypes;

  config param maxItems: int = 23;

  record Node_Knapsack
  {
    var depth: int;
    var items: c_array(c_int, maxItems);

    // default-initializer
    proc init()
    {}

    // root-initializer
    proc init(problem)
    {
      this.init();
    }

    // copy-initializer
    proc init(other: Node_Knapsack)
    {
      this.depth = other.depth;
      this.items = other.items;
    }

    proc deinit()
    {}
  }
}
