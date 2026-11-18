# Column Generation for Max Flow

A small, readable Python implementation of column generation for the **path-based
maximum flow problem**, plus short write-ups (PDF and LaTeX) that explain the math.

## What it does

Instead of one variable per arc, the LP has one variable per s-t path, and there can
be exponentially many paths. Column generation avoids listing them:

1. Solve a **restricted master problem (RMP)** over a small set of paths P'.
2. Read the **dual** value `y_a` of each arc's capacity constraint (the shadow price of
   one more unit of capacity on that arc).
3. **Pricing:** find the shortest s-t path using the duals as arc lengths (Dijkstra).
4. If that path has length **< 1**, its reduced cost `1 - sum(y_a)` is positive, so add it
   to P' and go back to step 1. If its length is **>= 1**, no missing path can help and
   the RMP solution is a maximum flow.

At the end, the duals mark a minimum cut and `sum(u_a * y_a)` equals the max flow value.

## Requirements

- Python 3.8+
- `numpy`
- `scipy` (1.6 or newer, for the HiGHS solver)

```
pip install numpy scipy
```

**No Gurobi (or any commercial solver) is needed.** The LP is solved by
`scipy.optimize.linprog(method="highs")`, and Dijkstra is written by hand with `heapq`.

## Run it

```
python column_generation_maxflow.py
```

This runs two examples (a 4-node and a 6-node network). For each iteration it prints
the paths in the RMP, the flows, the nonzero duals, and the pricing result. At the end
it checks the answer against an ordinary arc-based max flow LP and asserts they match.

Expected final results: max flow **5** (4-node) and **19** (6-node).

## How the code maps to the math

| Function | Role |
|---|---|
| `solve_rmp(arcs, paths)` | Builds and solves the RMP. Rows = arcs, columns = paths. Returns the objective, path flows `f`, and arc duals `y`. |
| `dijkstra(arcs, length, s, t)` | The pricing problem: shortest path with arc lengths `y_a`. |
| `bfs_path(arcs, s, t)` | Finds a first path to seed the RMP (the fewest-arc path). |
| `column_generation(...)` | The main loop: solve RMP, price, add a path, repeat. |
| `max_flow_arc_lp(...)` | Independent check using the standard arc-flow LP. |

### A note on the duals from scipy

`linprog` always minimizes, so the code maximizes `sum(f)` by minimizing `-sum(f)`.
For that minimization, `res.ineqlin.marginals` are `<= 0`, so the code flips the sign:

```python
y = -res.ineqlin.marginals     # dual y_a >= 0 of the original max problem
```

## Using your own network

Arcs are a dict of `(from, to): capacity`:

```python
from column_generation_maxflow import column_generation

arcs = {("s", "a"): 3, ("s", "b"): 2, ("a", "t"): 2, ("b", "t"): 3}
value, paths, flows, duals = column_generation(arcs, "s", "t")
```

If you leave out `initial_path`, a BFS path is used as the starting column.

## Good to know

- **Duals are not always unique.** When several optimal dual solutions exist, the solver
  returns one of them. Intermediate duals (and so the order paths are added) can differ
  between solvers or versions. The final max flow value is always the same.
- **Ties in pricing.** Many paths can tie at cost 0 early on. Dijkstra returns one of
  them, which changes the order of the iterations but not the final answer.
- **Not for speed.** Plain max flow is solved much faster by augmenting-path or
  push-relabel algorithms. Column generation is useful when paths have side constraints
  (hop limits, resource limits, multi-commodity flow). Then only the pricing step changes,
  for example to a constrained shortest path.

## Files

| File | Contents |
|---|---|
| `column_generation_maxflow.py` | The implementation and examples. |
| `colgen_maxflow_simple.pdf` / `.tex` | Short explanation: formulation, duals and reduced cost, algorithm in words, 6-node example with full dual and path-cost tables. |
| `colgen_maxflow_iter.pdf` / `.tex` | Formulation with defined parameters and variables, and a 4-node example broken down by iteration. |
| `colgen_maxflow.pdf` / `.tex` | Fuller write-up with the optimality proof and constrained-path extensions. |
