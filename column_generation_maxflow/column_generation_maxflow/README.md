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
- `numpy` and `scipy` (1.6 or newer, for the HiGHS solver) -- **always required**

```
pip install numpy scipy
```

The restricted master LP can be solved by any of three back ends. Only the one you
choose needs to be installed:

| Back end | Install | License |
|---|---|---|
| `scipy` (default) | nothing extra | None needed (free HiGHS solver) |
| `gurobi` | `pip install gurobipy` | The pip install includes a free size-limited license (2,000 variables / 2,000 constraints, non-commercial). Students and staff at academic institutions can get a free full license from Gurobi. |
| `ampl` | `pip install amplpy` then `python -m amplpy.modules install highs` | Free AMPL Community Edition license (open-source solvers, no size limit). Activate it with `python -m amplpy.modules activate <license-uuid>`. |

Dijkstra (the pricing step) is written by hand with `heapq` in every case.

## Run it

```
python column_generation_maxflow.py                    # scipy (default)
python column_generation_maxflow.py --solver gurobi    # Gurobi
python column_generation_maxflow.py --solver ampl      # AMPL with HiGHS
python column_generation_maxflow.py --solver ampl --ampl-solver gurobi
python column_generation_maxflow.py --solver all       # run all three
```

This runs two examples (a 4-node and a 6-node network). For each iteration it prints
the paths in the RMP, the flows, the nonzero duals, and the pricing result. At the end
it checks the answer against an ordinary arc-based max flow LP and asserts they match.

Expected final results: max flow **5** (4-node) and **19** (6-node).

**Testing status:** the `scipy` back end was run and verified. The `gurobi` and `ampl`
back ends were written from those libraries' documented APIs but have **not been run**
(the packages could not be installed where this was written). Every back end checks that
the dual objective `sum(u_a * y_a)` equals the primal objective and raises an error if it
does not, so a sign or setting problem will show up as an error rather than a wrong answer.
If you hit one, please fix the corresponding `_solve_rmp_<name>` function; each is short.

## How the code maps to the math

| Function | Role |
|---|---|
| `solve_rmp(arcs, paths, solver)` | Solves the RMP with the chosen back end (`_solve_rmp_scipy`, `_solve_rmp_gurobi`, `_solve_rmp_ampl`). Rows = arcs, columns = paths. Returns the objective, path flows `f`, and arc duals `y`, after a strong-duality check. |
| `dijkstra(arcs, length, s, t)` | The pricing problem: shortest path with arc lengths `y_a`. |
| `bfs_path(arcs, s, t)` | Finds a first path to seed the RMP (the fewest-arc path). |
| `column_generation(...)` | The main loop: solve RMP, price, add a path, repeat. |
| `max_flow_arc_lp(...)` | Independent check using the standard arc-flow LP. |

### Where the duals come from

`linprog` always minimizes, so the code maximizes `sum(f)` by minimizing `-sum(f)`.
For that minimization, `res.ineqlin.marginals` are `<= 0`, so the code flips the sign:

```python
y = -res.ineqlin.marginals     # dual y_a >= 0 of the original max problem
```

Gurobi and AMPL model the problem as a maximization directly, so they report the shadow
price as is: `constr.Pi` in gurobipy and `constraint.dual()` in amplpy.

## Using your own network

Arcs are a dict of `(from, to): capacity`:

```python
from column_generation_maxflow import column_generation

arcs = {("s", "a"): 3, ("s", "b"): 2, ("a", "t"): 2, ("b", "t"): 3}
value, paths, flows, duals = column_generation(arcs, "s", "t")                  # scipy
value, paths, flows, duals = column_generation(arcs, "s", "t", solver="gurobi")   # Gurobi
value, paths, flows, duals = column_generation(arcs, "s", "t", solver="ampl",
                                                ampl_solver="highs")           # AMPL
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
