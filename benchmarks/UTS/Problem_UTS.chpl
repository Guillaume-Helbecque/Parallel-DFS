module Problem_UTS
{
  use Time;
  use CTypes;

  use Problem;
  use Header_chpl_c_UTS;

  const BIN: c_int      = 0;
  const GEO: c_int      = 1;
  const HYBRID: c_int   = 2;
  const BALANCED: c_int = 3;

  class Problem_UTS : Problem
  {
    /* Tree type
     *   Trees are generated using a Galton-Watson process, in
     *   which the branching factor of each node is a random
     *   variable.
     *
     *   The random variable can follow a binomial distribution
     *   or a geometric distribution. Hybrid tree are
     *   generated with geometric distributions near the
     *   root and binomial distributions towards the leaves.
     */
    var treeType: c_int;
    var b_0: c_double;
    var rootId: c_int;

    /*  Tree type BIN (BINOMIAL)
     *  The branching factor at the root is specified by b_0.
     *  The branching factor below the root follows an
     *     identical binomial distribution at all nodes.
     *  A node has m children with prob q, or no children with
     *     prob (1-q).  The expected branching factor is q * m.
     */
    var nonLeafBF: c_int; // m
    var nonLeafProb: c_double; // q

    /*  Tree type GEO (GEOMETRIC)
     *  The branching factor follows a geometric distribution with
     *     expected value b.
     *  The probability that a node has 0 <= n children is p(1-p)^n for
     *     0 < p <= 1. The distribution is truncated at MAXNUMCHILDREN.
     *  The expected number of children b = (1-p)/p.  Given b (the
     *     target branching factor) we can solve for p.
     *
     *  A shape function computes a target branching factor b_i
     *     for nodes at depth i as a function of the root branching
     *     factor b_0 and a maximum depth gen_mx.
     */
    var gen_mx: c_int;
    var shape_fn: c_int;

    /*  In type HYBRID trees, each node is either type BIN or type
     *  GEO, with the generation strategy changing from GEO to BIN
     *  at a fixed depth, expressed as a fraction of gen_mx
     */
    var shiftDepth: c_double;

    /* compute granularity - number of rng evaluations per tree node */
    var computeGranularity: c_int;

    proc init(const tree_type: c_int, const bf_0: c_double, const rootIdx: c_int, const nonLeafBFact: c_int,
      const nonLeafProba: c_double, const gen: c_int, const shape_fct: c_int, const shiftD: c_double,
      const gran: c_int): void
    {
      this.treeType = tree_type;
      this.b_0 = bf_0;
      this.rootId = rootIdx;
      this.nonLeafBF = nonLeafBFact;
      this.nonLeafProb = nonLeafProba;
      this.gen_mx = gen;
      this.shape_fn = shape_fct;
      this.shiftDepth = shiftD;
      this.computeGranularity = gran;
    }

    override proc copy()
    {
      return new Problem_UTS(treeType, b_0, rootId, nonLeafBF, nonLeafProb, gen_mx,
        shape_fn, shiftDepth, computeGranularity);
    }

    override proc decompose(type Node, const parent: Node, ref tree_loc: int, ref num_sol: int,
      best: atomic int, ref best_task: int): [] Node
    {
      var numChildren: c_int = uts_numChildren(parent, treeType, nonLeafBF, nonLeafProb,
        b_0, shape_fn, gen_mx, shiftDepth);

      var children: [0..#numChildren] Node;

      if (numChildren > 0) {
        c_decompose(parent, c_ptrTo(children), treeType, numChildren, gen_mx,
        shiftDepth, computeGranularity, tree_loc, best_task);
      }
      else {
        num_sol += 1;
      }

      return children;
    }

    // No bounding in UTS
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
      writeln("UTS - Unbalanced Tree Search");
      writeln("Tree type: ", treeType, " (", uts_trees_str[treeType]:string, ")");
      writeln("Tree shape parameters:");
      writeln("  root branching factor b_0 = ", b_0, ", root seed r = ", rootId);
      if (treeType == GEO || treeType == HYBRID) {
        writeln("  GEO parameters: gen_mx = ", gen_mx, ", shape function = ", shape_fn, " (", uts_geoshapes_str[shape_fn]:string, ")");
      }
      if (treeType == BIN || treeType == HYBRID) {
        var q: c_double = nonLeafProb;
        var m: c_int = nonLeafBF;
        var es: c_double = (1.0 / (1.0 - q * m));
        writeln("  BIN parameters: q = ", q, ", m = ", m, ", E(n) = ", q * m, ", E(s) = ", es);
      }
      if (treeType == HYBRID) {
        writeln("  HYBRID:  GEO from root to depth ", ceil(shiftDepth * gen_mx): c_int, ", then BIN");
      }
      if (treeType == BALANCED) {
        var exp_nodes: uint(64) = ((b_0**(gen_mx+1) - 1.0) / (b_0 - 1.0)):uint(64); /* geometric series */
        var exp_leaves: uint(64) = (b_0**gen_mx):uint(64);
        writeln("  BALANCED parameters gen_mx = ", gen_mx);
        writeln("    Expected size: ", exp_nodes, ", ", exp_leaves);
      }
      writeln("Random number generator: "); // TO COMPLETE
      writeln("Compute granularity: ", computeGranularity);
      writeln("=================================================");
    }

    override proc print_results(const subNodeExplored: [] int, const subSolExplored: [] int,
      const subDepthReached: [] int, const best: int, const timer: stopwatch): void
    {
      var treeSize: int = (+ reduce subNodeExplored);
      var nbLeaf: int   = (+ reduce subSolExplored);
      var maxDepth: int = (max reduce subDepthReached);
      var par_mode: string = if (numLocales == 1) then "tasks" else "locales";

      writeln("\n=================================");
      writeln("Size of the explored tree: ", treeSize);
      /* writeln("Size of the explored tree per locale: ", sizePerLocale); */
      writeln("% of the explored tree per ", par_mode, ": ", 100 * subNodeExplored:real / treeSize:real);
      writeln("Number of leaves explored: ", nbLeaf, " (", 100 * nbLeaf:real / treeSize:real, "%)");
      /* writeln("Number of explored solutions per locale: ", numSolPerLocale); */
      writeln("Tree depth: ", maxDepth);
      /* writeln("Optimal makespan: ", best); */
      writeln("Elapsed time: ", timer.elapsed(), " [s]");
      writeln("=================================\n");
    }

    // TO COMPLETE
    override proc output_filepath(): string
    {
      var tup = ("./chpl_uts.txt");
      return "".join(tup);
    }

    override proc help_message(): void
    {
      writeln("\n  UTS Benchmark Parameters:\n");
      writeln("   --t   int       tree type (0: BIN, 1: GEO, 2: HYBRID, 3: BALANCED)");
      writeln("   --b   double    root branching factor");
      writeln("   --r   int       root seed 0 <= r < 2^31");
      writeln("   --a   int       GEO: tree shape function");
      writeln("   --d   int       GEO, BALANCED: tree depth");
      writeln("   --q   double    BIN: probability of non-leaf node");
      writeln("   --m   int       BIN: number of children for non-leaf node");
      writeln("   --f   double    HYBRID: fraction of depth for GEO -> BIN transition");
      writeln("   --g   int       granularity: number of rng_spawns per node");
    }

  } // end class

} // end module
