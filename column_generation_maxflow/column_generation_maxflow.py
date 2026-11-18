"""
Column generation for the path-based maximum flow problem.

Master problem:     max  sum_p f_p
                    s.t. sum_{p containing a} f_p <= u_a   for every arc a   (dual y_a >= 0)
                         f_p >= 0

Restricted master:  the same LP, but only over the paths found so far (P').
Pricing problem:    shortest s-t path using the duals y_a as arc lengths.
                    A path is worth adding if its length is < 1
                    (reduced cost = 1 - sum of duals on the path > 0).

Only numpy and scipy are needed (the LP is solved by scipy's HiGHS solver).
"""

import heapq
from collections import deque

import numpy as np
from scipy.optimize import linprog


# ----------------------------------------------------------------------------
# Graph helpers
# ----------------------------------------------------------------------------
def build_adjacency(arcs):
    """arcs: {(u, v): capacity}. Returns {u: [v, ...]} in the order arcs were given."""
    adj = {}
    for (u, v) in arcs:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, [])
    return adj


def bfs_path(arcs, s, t):
    """Fewest-arc s-t path, returned as a list of arcs. Used to seed the RMP."""
    adj = build_adjacency(arcs)
    parent = {s: None}
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                queue.append(v)
    if t not in parent:
        raise ValueError("No s-t path exists.")
    nodes = [t]
    while nodes[-1] != s:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    return list(zip(nodes, nodes[1:]))


def dijkstra(arcs, length, s, t):
    """Shortest s-t path where length[(u, v)] >= 0 is the cost of each arc.

    Returns (path_length, path), with path as a list of arcs.
    """
    adj = build_adjacency(arcs)
    dist = {s: 0.0}
    parent = {}
    heap = [(0.0, s)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v in adj[u]:
            nd = d + length[(u, v)]
            if nd < dist.get(v, float("inf")) - 1e-12:   # strict improvement only
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    nodes = [t]
    while nodes[-1] != s:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    return dist[t], list(zip(nodes, nodes[1:]))


# ----------------------------------------------------------------------------
# Restricted master problem (RMP) -- three interchangeable solver back ends
# ----------------------------------------------------------------------------
# Each back end returns (objective, f, y):
#     f : list/array, flow on each path in `paths`
#     y : {arc: dual value >= 0}, the shadow price of each arc's capacity
SOLVERS = ("scipy", "gurobi", "ampl")


def _solve_rmp_scipy(arcs, paths, **_):
    """scipy / HiGHS (free, no license needed)."""
    arc_list = list(arcs)
    # A[a, p] = 1 if arc a is on path p, else 0 (one row per arc, one column per path)
    A = np.array([[1.0 if a in p else 0.0 for p in paths] for a in arc_list])
    u = np.array([arcs[a] for a in arc_list], dtype=float)

    # linprog minimizes, so maximize sum(f) by minimizing -sum(f).
    res = linprog(c=-np.ones(len(paths)), A_ub=A, b_ub=u,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"RMP failed: {res.message}")

    # For a minimization, ineqlin.marginals are <= 0; flip the sign to get the
    # dual values y_a >= 0 of the original maximization problem.
    y = dict(zip(arc_list, np.clip(-res.ineqlin.marginals, 0.0, None)))
    return -res.fun, list(res.x), y


def _solve_rmp_gurobi(arcs, paths, **_):
    """Gurobi via gurobipy. Needs `pip install gurobipy` and a license
    (the pip install includes a size-limited one; academic licenses are free)."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:
        raise ImportError("gurobipy is not installed: pip install gurobipy") from e

    m = gp.Model("rmp")
    m.Params.OutputFlag = 0
    f = m.addVars(len(paths), lb=0.0, name="f")
    m.setObjective(gp.quicksum(f[i] for i in range(len(paths))), GRB.MAXIMIZE)
    cons = {}
    for a, cap in arcs.items():
        used = gp.quicksum(f[i] for i, p in enumerate(paths) if a in p)
        cons[a] = m.addConstr(used <= cap, name=f"cap_{a[0]}_{a[1]}")
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi RMP not optimal (status {m.Status})")

    # Pi = shadow price of the constraint = d(objective)/d(rhs), which is >= 0 here.
    y = {a: max(cons[a].Pi, 0.0) for a in arcs}
    return m.ObjVal, [f[i].X for i in range(len(paths))], y


def _solve_rmp_ampl(arcs, paths, ampl_solver="highs", **_):
    """AMPL via amplpy. Needs `pip install amplpy`, a solver module
    (python -m amplpy.modules install highs) and an AMPL license
    (the free Community Edition works with open-source solvers)."""
    try:
        from amplpy import AMPL
    except ImportError as e:
        raise ImportError("amplpy is not installed: pip install amplpy") from e

    ampl = AMPL()
    ampl.eval(r"""
        set ARCS;
        set PATHS;
        param cap{ARCS} >= 0;
        param on{ARCS, PATHS} binary default 0;   # 1 if the arc is on the path
        var f{PATHS} >= 0;
        maximize Flow: sum{p in PATHS} f[p];
        s.t. Cap{a in ARCS}: sum{p in PATHS} on[a, p] * f[p] <= cap[a];
    """)
    labels = {a: f"{a[0]}_{a[1]}" for a in arcs}              # AMPL set members
    ampl.set["ARCS"] = list(labels.values())
    ampl.set["PATHS"] = list(range(len(paths)))
    ampl.param["cap"] = {labels[a]: float(c) for a, c in arcs.items()}
    ampl.param["on"] = {(labels[a], i): 1 for i, p in enumerate(paths) for a in p}
    ampl.option["solver"] = ampl_solver
    ampl.solve()

    obj = ampl.obj["Flow"].value()
    f = [ampl.var["f"][i].value() for i in range(len(paths))]
    # AMPL's dual of a <= constraint in a max problem is >= 0 (d objective / d rhs).
    y = {a: max(ampl.con["Cap"][labels[a]].dual(), 0.0) for a in arcs}
    return obj, f, y


def solve_rmp(arcs, paths, solver="scipy", **options):
    """Solve the RMP over `paths` with the chosen back end.

    Returns (objective, f, y). The duals are checked against the objective
    (strong duality: sum(u_a * y_a) must equal the objective), which catches
    any sign-convention or solver-setting mistakes in a back end.
    """
    backends = {"scipy": _solve_rmp_scipy,
                "gurobi": _solve_rmp_gurobi,
                "ampl": _solve_rmp_ampl}
    if solver not in backends:
        raise ValueError(f"solver must be one of {SOLVERS}")
    obj, f, y = backends[solver](arcs, paths, **options)

    dual_obj = sum(arcs[a] * y[a] for a in arcs)
    if abs(dual_obj - obj) > 1e-6 * max(1.0, abs(obj)):
        raise RuntimeError(f"[{solver}] dual objective {dual_obj} != primal {obj}; "
                           "check the dual sign convention / solver settings.")
    return obj, f, y


# ----------------------------------------------------------------------------
# Column generation loop
# ----------------------------------------------------------------------------
def fmt_path(path):
    return "-".join([path[0][0]] + [v for (_, v) in path])


def column_generation(arcs, s, t, initial_path=None, tol=1e-9, verbose=True,
                      solver="scipy", **solver_options):
    """Max flow by column generation. Returns (max_flow, paths, f, y).

    solver: "scipy" (default), "gurobi" or "ampl" -- which LP back end solves the RMP.
    solver_options: passed to the back end, e.g. ampl_solver="highs".
    """
    paths = [initial_path if initial_path is not None else bfs_path(arcs, s, t)]

    it = 0
    while True:
        it += 1
        obj, f, y = solve_rmp(arcs, paths, solver, **solver_options)

        # Pricing: shortest s-t path with the duals as arc lengths.
        cost, new_path = dijkstra(arcs, y, s, t)
        reduced_cost = 1.0 - cost

        if verbose:
            nonzero = {f"{u}{v}": round(float(val), 4) for (u, v), val in y.items() if val > tol}
            flows = {fmt_path(p): round(float(x), 4) + 0.0 for p, x in zip(paths, f)}
            print(f"Iteration {it}")
            print(f"  paths in RMP : {[fmt_path(p) for p in paths]}")
            print(f"  RMP flows    : {flows}   objective = {obj:.4f}")
            print(f"  nonzero duals: {nonzero}")
            print(f"  pricing      : shortest path {fmt_path(new_path)}, "
                  f"cost = {cost:.4f}, reduced cost = {reduced_cost:.4f}")

        if cost >= 1.0 - tol:        # no path has positive reduced cost -> optimal
            if verbose:
                print("  -> shortest path cost >= 1: stop.\n")
            return obj, paths, f, y

        if new_path in paths:        # safety net against numerical cycling
            raise RuntimeError("Pricing returned a path already in the RMP.")
        paths.append(new_path)
        if verbose:
            print(f"  -> cost < 1: add {fmt_path(new_path)}.\n")


# ----------------------------------------------------------------------------
# Independent check: the ordinary arc-based max flow LP (no column generation)
# ----------------------------------------------------------------------------
def max_flow_arc_lp(arcs, s, t):
    """Standard arc-flow LP: maximize flow out of s with flow conservation."""
    arc_list = list(arcs)
    nodes = sorted({n for a in arc_list for n in a})
    inner = [n for n in nodes if n not in (s, t)]

    c = np.zeros(len(arc_list))                 # objective: net flow leaving s
    for j, (u, v) in enumerate(arc_list):
        if u == s:
            c[j] -= 1.0
        if v == s:
            c[j] += 1.0

    A_eq = np.zeros((len(inner), len(arc_list)))  # conservation at every inner node
    for i, n in enumerate(inner):
        for j, (u, v) in enumerate(arc_list):
            if v == n:
                A_eq[i, j] += 1.0
            if u == n:
                A_eq[i, j] -= 1.0

    bounds = [(0, arcs[a]) for a in arc_list]
    res = linprog(c, A_eq=A_eq, b_eq=np.zeros(len(inner)), bounds=bounds, method="highs")
    return -res.fun


# ----------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------
# 4-node network (step-by-step example)
ARCS_4 = {("s", "a"): 3, ("s", "b"): 2, ("a", "b"): 1, ("a", "t"): 2, ("b", "t"): 3}

# 6-node network
ARCS_6 = {("s", "a"): 10, ("s", "b"): 10, ("a", "b"): 2, ("a", "c"): 4, ("a", "d"): 8,
          ("b", "d"): 9, ("d", "c"): 6, ("c", "t"): 10, ("d", "t"): 10}


def run_example(name, arcs, s, t, initial_path=None, solver="scipy", **solver_options):
    print("=" * 70)
    print(f"{name}   [RMP solver: {solver}]")
    print("=" * 70)
    value, paths, f, y = column_generation(arcs, s, t, initial_path=initial_path,
                                           solver=solver, **solver_options)
    check = max_flow_arc_lp(arcs, s, t)

    print(f"Column generation max flow : {value:.4f}")
    print(f"Arc-based LP max flow      : {check:.4f}")
    assert abs(value - check) < 1e-6, "Column generation disagrees with the arc LP!"

    dual_obj = sum(arcs[a] * y[a] for a in arcs)
    cut = [a for a in arcs if y[a] > 1e-9]
    print(f"Dual objective sum(u_a*y_a): {dual_obj:.4f}")
    print(f"Arcs with y_a > 0 (min cut): {cut}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Column generation for max flow")
    parser.add_argument("--solver", choices=SOLVERS + ("all",), default="scipy",
                        help="LP solver for the restricted master problem")
    parser.add_argument("--ampl-solver", default="highs",
                        help="solver AMPL should call (e.g. highs, cbc, gurobi)")
    args = parser.parse_args()
    chosen = SOLVERS if args.solver == "all" else (args.solver,)

    # The starting paths below match the documents; leave initial_path out
    # to seed with a BFS path instead.
    for solver in chosen:
        run_example("4-node example", ARCS_4, "s", "t",
                    initial_path=[("s", "a"), ("a", "t")],
                    solver=solver, ampl_solver=args.ampl_solver)
        run_example("6-node example", ARCS_6, "s", "t",
                    initial_path=[("s", "a"), ("a", "c"), ("c", "t")],
                    solver=solver, ampl_solver=args.ampl_solver)
