# P3D-DFS
Chapel-based implementation of our Productivity- and Performance-aware Parallel Distributed Depth-First Search algorithm, namely P3D-DFS. The latter is a generic and general algorithm that can be instantiated on numerous tree-based problems. It is based on the `DistBag_DFS` distributed data structure, which employs an underlying Work Stealing (WS) mechanism to balance workload across multiple Chapel's locales.

## Chapel
[Chapel](https://chapel-lang.org/) is a programming language designed for productive parallel computing on large-scale systems.
Chapel supports a multi-threaded execution model via high-level abstractions for data parallelism, task parallelism, concurrency, and nested parallelism. Chapel's locale type enables users to specify and reason about the placement of data and tasks on a target architecture in order to tune for locality and affinity. Chapel supports global-view data aggregates with user-defined implementations, permitting operations on distributed data structures to be expressed in a natural manner.

Our Chapel codes rely on version $1.31.0$. <br/>
The corresponding Chapel's version is downloadable at: https://github.com/chapel-lang/chapel/releases.

## The `DistBag_DFS` distributed data structure
The `DistBag_DFS` distributed data structure is a parallel-safe distributed multi-pool implementation that is unordered and incorporates a WS mechanism that balances workload across multiple locales, transparently to the user. It can contain either predefined-Chapel types, user-defined types or external ones (*e.g.* C structures).
This data structure has been derived from the `DistBag` data structure supplied is the `DistributedBag` Chapel's module, and revised in two different ways: (1) we propose a new scheduling policy of its elements as well as a new synchronization mechanism using non-blocking split-deques, and (2) we redefine the underlying WS mechanism.

## Compilation & Execution
- **Step 1:** [Set up your Chapel environment](https://chapel-lang.org/docs/usingchapel/chplenv.html) according to the machine on which your code is expected to run, and [build Chapel](https://chapel-lang.org/docs/usingchapel/building.html).
- **Step 2:** Compile with `make` and execute with:
```
./main.o --mode=distributed ${problem-specific options} -nl 2
```
where:
- `--mode` is the parallel execution mode, *i.e.* `multicore` or `distributed`;
- For the list of supported problems and options, see below;
- `-nl` is the number of Chapel's process(es), typically the number of computer nodes in distributed setting.

## Supported problems

### Branch-and-Bound algorithms (B&B)
B&B are exact optimization algorithms constructing implicity trees by successively applying *branching*, *bounding* and *pruning* operators. Each tree node corresponds to a subproblem (the initial problem defined on a restricted domain) and children nodes are obtained by further restricting the search space.

- [The Permutation Flow-shop Scheduling problem](./benchmarks/PFSP) (PFSP)
- [The 0/1-Knapsack problem](./benchmarks/Knapsack)

### Backtracking algorithms
Backtracking is an algorithmic technique for solving problems recursively by trying to build a solution incrementally, one piece at a time, removing those solutions that fail to satisfy the constraints of the problem at any point in time.

- [The Unbalanced Tree-Search benchmark](./benchmarks/UTS) (UTS)
- [The N-Queens problem](./benchmarks/NQueens)

## Future improvements
- Perform low-level investigations and optimizations of the `DistBag-DFS` distributed data structure, as well as the underlying WS mechanism.
- Make PFSP more flexible, *e.g.* considering advanced B&B branching techniques.
- Extend the list of supported problems, *e.g.* the Quadratic Assignment Problem (QAP), the Traveling Salesman Problem (TSP).

## Contributors
- Guillaume Helbecque, Université de Lille, CNRS/CRIStAL UMR 9189, Centre Inria de l'Université de Lille, France
- Jan Gmys, Centre Inria de l'Université de Lille, France
- Tiago Carneiro, Université du Luxembourg, FSTM, Luxembourg
- Nouredine Melab, Université de Lille, CNRS/CRIStAL UMR 9189, Centre Inria de l'Université de Lille, France
- Pascal Bouvry, Université du Luxembourg, DCS-FSTM/SnT, Luxembourg
