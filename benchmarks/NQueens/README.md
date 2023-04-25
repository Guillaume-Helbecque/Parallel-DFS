# The N-Queens problem

The problem consists in placing `N` chess queens on a $N \times N$ chessboard so that no two queens attack each other; thus, a solution requires that no two queens share the same row, column, or diagonal. We also introduce an artificial granularity parameter, `g`, which controls the number of safety check(s) per evaluation.

### Launch & Command-line parameters

```chapel
./main_nqueens.o --N=15 --g=10
```
where:
- `N` (`int`): number of queens ($13$ by default);
- `g` (`int`): number of safety check(s) per evaluation ($1$ by default).

### Instances

The exact number of N-Queens solutions is only known for $N < 28$ [1]. In the following table, we report the exact number of solutions for these instances, as well as the size of the explored tree using P3D-DFS.

<table><tr><td>

| $N$|  Number of solutions    | Size of the explored tree |
|----|-------------------------|---------------------------|
| 1  | 1 	                     | 1                         |
| 2  | 0 	                     | 2                         |
| 3  | 0 	                     | 5                         |
| 4  | 2                       | 16                        |
| 5  | 10 	                   | 53                        |
| 6  | 4 	                     | 152                       |
| 7  | 40 	                   | 551                       |
| 8  | 92 	                   | 2,056                     |
| 9  | 352 	                   | 8,393                     |
| 10 | 724 	                   | 35,538                    |
| 11 | 2,680 	                 | 166,925                   |
| 12 | 14,200 	               | 856,188                   |
| 13 | 73,712 	               | 4,674,889                 |
| 14 | 365,596 	               | 27,358,552                |

</td><td>

| $N$|  Number of solutions    | Size of the explored tree |
|----|-------------------------|---------------------------|
| 15 | 2,279,184 	             | 171,129,071               |
| 16 | 14,772,512 	           | 1,141,190,302             |
| 17 | 95,815,104 	           | 8,017,021,931             |
| 18 | 666,090,624 	           | 59,365,844,490            |
| 19 | 4,968,057,848 	         | 461,939,618,823           |
| 20 | 39,029,188,884 	       | ...                       |
| 21 | 314,666,222,712 	       | ...                       |
| 22 | 2,691,008,701,644 	     | ...                       |
| 23 | 24,233,937,684,440 	   | ...                       |
| 24 | 227,514,171,973,736 	   | ...                       |
| 25 | 2,207,893,435,808,350 	 | ...                       |
| 26 | 22,317,699,616,364,000  | ...                       |
| 27 | 234,907,967,154,122,528 | ...                       |

</td></tr></table>

### References

[1] Preußer, T. (2019). A Brute-Force Solution to the 27-Queens Puzzle Using a Distributed Computation. In: Hinze, T., Rozenberg, G., Salomaa, A., Zandron, C. (eds) Membrane Computing. CMC 2018. Lecture Notes in Computer Science(), vol 11399. Springer, Cham. DOI: [10.1007/978-3-030-12797-8_3](https://doi.org/10.1007/978-3-030-12797-8_3)
